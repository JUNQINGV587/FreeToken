"""Disk tier: NVMe-backed MoE experts (VRAM <- RAM <- NVMe).

Lets the offload backend serve experts that do NOT fit in pinned RAM: the RAM
bank holds only the first ``ram_experts`` experts per layer (pinned), the rest
stay on disk in the original checkpoint. When the GPU slot cache misses a
disk-resident expert, :class:`DiskTier` fetches its rows with O_DIRECT preadv
into a small pinned staging buffer and H2D-copies them into the slot the LRU
kernel already assigned, then shrinks the miss list so the existing PCIe
``copy_missing`` path only moves the RAM-resident misses.

v0 scope (prototype):
* native NVFP4 layout only (the "triton" backend banks -- what sm_120 picks);
* ``decode_target == "gpu"`` (offload) only -- the CPU executor reads banks
  directly and would read released pages. The RAM-resident miss tier
  (``--moe-cpu-tier``, moe/cpu_tier.py, dsv41 M2) is the deliberate exception:
  its CPU workers read ONLY the RAM-pinned bank rows [0, ram_experts) -- the
  rows this tier never releases -- never disk rows (those keep the fetch path);
* synchronous fetch (the layer waits for its disk misses); no CUDA-graph
  capture (the miss-list D2H/H2D round trip is host-side and variable);
* prefill_overlap off (the double-buffer prefill path bypasses the slot cache).

The bank rows are read from the ORIGINAL safetensors shards: every expert
tensor is a contiguous per-expert tensor, so a bank row is one (or two, for
the gate|up-fused banks) aligned super-block preads. No FTW conversion needed.
"""

from __future__ import annotations

import ctypes
import json
import os
import struct
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import torch

from freetoken.moe.host_banks import HostBank

_ALIGN = 4096


@dataclass(frozen=True)
class DiskTierSpec:
    """Engine -> loader: how many experts per layer stay pinned in RAM.

    Experts ``[0, ram_experts)`` are pinned as usual; ``[ram_experts, E)`` keep
    their bank rows allocated but their pages are released after load and are
    served from disk by :class:`DiskTier`.

    ``pin_doc`` (optional, from ``--moe-ram-pin-file``): a validated
    ``freetoken.ram_pin_set.v1`` document choosing WHICH local experts fill the
    pinned rows, per layer per EP rank (a learned hot set instead of the
    contiguous prefix). The bank layout is unchanged -- the pinned rows are
    still the first ``ram_experts`` bank rows; only the local-id -> bank-row
    mapping changes (see :func:`pin_rows_to_row_map`)."""

    ram_experts: int
    pin_doc: dict | None = None


def load_ram_pin_doc(path: str) -> dict:
    """Load and validate a ``freetoken.ram_pin_set.v1`` pin-set document.

    Structural validation only (schema/shape/window per rank); the caller
    cross-checks num_layers/num_experts/ep/budgets against the model and the
    resolved RAM budget. Raises ValueError on any malformed content -- a bad
    pin file must fail the boot, never serve a wrong row."""
    with open(path) as f:
        doc = json.load(f)
    try:
        if doc["format"] != "freetoken.ram_pin_set.v1":
            raise ValueError(f"unknown format {doc['format']!r}")
        L = int(doc["num_layers"])
        E = int(doc["num_experts"])
        ep = int(doc["ep"])
        budgets = [int(b) for b in doc["budgets"]]
        ranks = doc["ranks"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{path}: malformed ram_pin_set document: {exc}") from exc
    if not (L > 0 and E > 0 and ep > 0):
        raise ValueError(f"{path}: num_layers/num_experts/ep must be positive")
    if len(budgets) != ep:
        raise ValueError(f"{path}: budgets has {len(budgets)} entries for ep={ep}")
    if E % ep:
        raise ValueError(f"{path}: num_experts {E} not divisible by ep {ep}")
    local = E // ep
    for rank in range(ep):
        key = str(rank)
        rows = ranks.get(key)
        if not isinstance(rows, list) or len(rows) != L:
            raise ValueError(f"{path}: ranks[{key!r}] must have {L} layer lists")
        for layer, ids in enumerate(rows):
            if (not isinstance(ids, list) or len(ids) != budgets[rank]
                    or ids != sorted(set(ids))
                    or any(not isinstance(e, int) or e < 0 or e >= local for e in ids)):
                raise ValueError(
                    f"{path}: ranks[{key!r}][{layer}] must be {budgets[rank]} unique "
                    f"sorted local ids in [0, {local})")
    return doc


def validate_ram_pin_doc(doc: dict, num_experts: int, num_layers: int, ep: int,
                         ram_experts: int) -> list[str]:
    """Cross-check a loaded pin document against the model and the resolved RAM
    budget. Returns a list of human-readable problems (empty = OK); the engine
    reports them with the other unmet disk-tier preconditions, so a stale pin
    file fails the boot with the full diagnosis instead of serving wrong rows.

    The EP split of the pin budget is per-rank host RAM, so the doc's budgets
    must match ``local_ram_experts`` per rank (validated here without the
    ownership object: budget[rank] = ram budget that rank's host actually pins)."""
    problems: list[str] = []
    if doc["num_experts"] != num_experts:
        problems.append(
            f"--moe-ram-pin-file: num_experts {doc['num_experts']} != model {num_experts}")
    if doc["num_layers"] != num_layers:
        problems.append(
            f"--moe-ram-pin-file: num_layers {doc['num_layers']} != model {num_layers}")
    if doc["ep"] != ep:
        problems.append(
            f"--moe-ram-pin-file: ep {doc['ep']} != --moe-ep-size {ep}")
    if not problems:
        # Replicate local_ram_experts' per-rank split of the host-wide budget
        # (the ownership object is not built yet at validation time; the EP
        # group IS the owner world here).
        local = num_experts // ep
        share, rem = divmod(ram_experts, ep)
        if share >= _AUTO_ALIGN:
            share = share // _AUTO_ALIGN * _AUTO_ALIGN
            rem = ram_experts - share * ep
            expected = [min(local, share + (_AUTO_ALIGN if r < rem // _AUTO_ALIGN else 0))
                        for r in range(ep)]
        else:
            expected = [min(local, share + (1 if r < rem else 0)) for r in range(ep)]
        budgets = doc["budgets"]
        if budgets != expected:
            problems.append(
                f"--moe-ram-pin-file: budgets {budgets} != resolved per-rank RAM "
                f"budget {expected} (regenerate the pin file for this host)")
    return problems


def pin_rows_to_row_map(pin_rows: list[list[int]], local_num: int) -> list[list[int]]:
    """Per-layer permutation ``local id -> bank row`` packing the pinned experts
    into bank rows ``[0, len(pin_rows[layer]))``.

    Bank row ``r`` of layer ``L`` holds local expert ``pin_rows[L][r]`` for
    ``r < len(pin_rows[L])``; the remaining (disk-resident) experts fill rows
    ``[len(pin_rows[L]), local_num)`` in ascending id order. The result is a
    bijection, so ``row_map[e] >= len(pin_rows[L])`` is exactly "e is
    disk-resident" -- the same shape of test the prefix layout gets from
    ``e >= ram``. ``pin_rows`` must be sorted unique ids in range; raises
    ValueError otherwise (a malformed pin set would serve wrong rows)."""
    row_map: list[list[int]] = []
    for layer, ids in enumerate(pin_rows):
        if ids != sorted(set(ids)) or any(e < 0 or e >= local_num for e in ids):
            raise ValueError(
                f"pin_rows[{layer}] must be unique sorted local ids in [0, {local_num})")
        pinned = set(ids)
        rest = [e for e in range(local_num) if e not in pinned]
        row = [0] * local_num
        for r, e in enumerate(list(ids) + rest):
            row[e] = r
        row_map.append(row)
    return row_map


def release_bank_tails(banks_by_name: dict[str, list[HostBank]], num_experts: int,
                       ram_experts: int) -> None:
    """MADV_DONTNEED the unpinned tail rows of every bank layer (post-load).

    The release is an optimization, not an invariant: the tail rows were never
    written at load, so when a row boundary is not page-aligned (the small scale
    banks) we warn and skip that bank instead of failing the boot."""
    _PAGE = 4096
    for layer_banks in banks_by_name.values():
        for bank in layer_banks:
            row_bytes = bank.nbytes // num_experts
            offset = ram_experts * row_bytes
            size = bank.nbytes - offset
            if offset % _PAGE or size % _PAGE:
                print(f"[disk-tier] WARNING: bank row boundary not page-aligned "
                      f"(ram_experts={ram_experts}, row_bytes={row_bytes}); skipping "
                      f"the release for this bank -- the tail rows were never written, "
                      f"so nothing is lost", flush=True)
                continue
            bank.release_range(offset, size)


def tail_resident_bytes(bank: HostBank, num_experts: int, ram_experts: int) -> int:
    """Bytes the kernel currently backs in the released tail rows ``[ram_experts, E)``.

    mincore(2) over the tail's byte range: one syscall, per-page residency.
    Conservative -- mincore also reports private-anon pages mapped from the
    shared zero page (a plain READ of a tail row), so this overcounts what
    actually costs RAM (cgroup memory.stat shmem is the real number).
    Returns -1 if mincore itself fails."""
    row_bytes = bank.nbytes // num_experts
    off = ram_experts * row_bytes
    size = bank.nbytes - off
    if size <= 0:
        return 0
    _PAGE = 4096
    vec = (ctypes.c_ubyte * (size // _PAGE))()
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_ubyte)]
    if libc.mincore(ctypes.c_void_p(bank.addr + off), size, vec) != 0:
        return -1
    return sum(vec) * _PAGE


def check_tail_unbacked(banks_by_name: dict[str, list[HostBank]], num_experts: int,
                        ram_experts: int) -> None:
    """Startup sanity check for the lazy-tail invariant, right after
    ``release_bank_tails``: nothing reads or writes the tail rows, so the kernel
    should be backing ~none of them. The expected worst case is one 2 MiB THP
    huge page per bank layer (shmem_enabled=always|force can back the
    prefix/tail boundary as a huge page); more than that means something touched
    the tail and the disk-tier RAM math no longer holds. Always logs, warns
    above the bound."""
    _HUGE = 2 << 20
    resident = tail = n_banks = 0
    for layer_banks in banks_by_name.values():
        for bank in layer_banks:
            row_bytes = bank.nbytes // num_experts
            t = bank.nbytes - ram_experts * row_bytes
            if t <= 0:
                continue
            r = tail_resident_bytes(bank, num_experts, ram_experts)
            if r < 0:
                continue  # mincore failed: skip rather than warn on our own probe
            resident += r
            tail += t
            n_banks += 1
    if n_banks == 0:
        return
    from freetoken.distributed import try_get_tp_info
    tp = try_get_tp_info()
    rank = getattr(tp, "rank", "?")
    size_ = getattr(tp, "size", "?")
    bound = n_banks * _HUGE
    print(f"[disk-tier] tail check rank={rank}/{size_}: resident {resident >> 20} MiB "
          f"of {tail >> 20} MiB (warn bound {bound >> 20} MiB = 1x2MiB per bank layer)",
          flush=True)
    if resident > bound:
        print(f"[disk-tier] WARNING: tail rows more resident than the THP bound -- "
              f"something is reading the released rows; the disk-tier RAM math no longer holds",
              flush=True)

# Native NVFP4 bank order (== _BANK_SCHEMAS["nvfp4"]) and, per bank, the
# checkpoint segments that make up one expert row: (proj, kind, dst_row_start,
# dst_row_end). The gate|up-fused banks splice gate rows then up rows on the
# output-row axis; down banks are a single segment. Row ends are None = rest.
_NVP4_BANK_SEGS = (
    (("gate_proj", "weight", 0, None), ("up_proj", "weight", None, None)),
    (("gate_proj", "weight_scale", 0, None), ("up_proj", "weight_scale", None, None)),
    (("gate_proj", "weight_scale_2", 0, None), ("up_proj", "weight_scale_2", None, None)),
    (("down_proj", "weight", 0, None),),
    (("down_proj", "weight_scale", 0, None),),
    (("down_proj", "weight_scale_2", 0, None),),
)



def _preadv_error(tier, staging, shard_idx: int, off: int, a0: int, slen: int,
                  direct: bool) -> OSError:
    vma = "?"
    try:
        for line in open("/proc/self/maps"):
            lo, hi = line.split()[0].split("-")
            if int(lo, 16) <= staging.addr < int(hi, 16):
                vma = line.strip()[:120]
                break
    except OSError:
        pass
    return OSError(
        f"disk-tier preadv failed: shard={shard_idx} off={off} a0={a0} "
        f"slen={slen} direct={direct} buf={hex(staging.addr)} "
        f"staging_size={tier._staging_size} thread={threading.current_thread().name} "
        f"vma={vma}"
    )


def _read_safetensors_offsets(path: str) -> dict[str, tuple[int, int]]:
    """{tensor_name: (start, end)} from a shard's safetensors header, as ABSOLUTE
    file offsets (data_offsets are relative to the data section, i.e. after the
    8-byte length + header JSON)."""
    with open(path, "rb") as f:
        (hlen,) = struct.unpack("<Q", f.read(8))
        meta = json.loads(f.read(hlen))
    base = 8 + hlen
    return {
        k: (v["data_offsets"][0] + base, v["data_offsets"][1] + base)
        for k, v in meta.items() if k != "__metadata__"
    }



class Nvfp4DiskIndex:
    """(bank, layer, expert) -> per-segment (shard_idx, offset, nbytes) locations.

    Built from the original checkpoint: the HF index json (name -> shard) plus
    each referenced shard's safetensors header (name -> byte range). Expert
    tensors are per-expert and contiguous, so a row is exactly one byte range
    per segment.
    """

    # Banks whose row content is a per-expert fp32 SCALAR (weight_scale_2 ->
    # fp16 broadcast). A 4-byte scalar must never cost a 4 KiB-aligned disk
    # read per expert per fetch: DiskTier preloads them once (P0-3 port of
    # colibri's read-batch economics).
    scalar_banks = frozenset({2, 5})

    def __init__(self, model_dir: str, config, spec) -> None:
        from freetoken.models.nvfp4_banks import _num_moe_layers
        from freetoken.utils.hf import download_hf_weight

        model_dir = download_hf_weight(model_dir)  # hub id -> local cache dir; no-op if local
        index_path = os.path.join(model_dir, "model.safetensors.index.json")
        with open(index_path, encoding="utf-8") as f:
            weight_map = json.load(f)["weight_map"]

        num_layers = _num_moe_layers(config)
        # (bank_layer, expert, proj, kind) -> (tensor_name, shard). The projection is keyed by
        # CANONICAL role (gate_proj/up_proj/down_proj, what _NVP4_BANK_SEGS names), so a
        # checkpoint that spells its experts w1/w2/w3 (V4.1, minimax) resolves through the
        # same spec.proj_to_role the in-RAM bank path uses.
        loc: dict[tuple[int, int, str, str], tuple[str, str]] = {}
        for name, shard in weight_map.items():
            m = spec.key_pattern.match(name)
            if m is None:
                continue
            bank_layer = spec.layer_to_bank(int(m.group("layer")), config)
            if bank_layer is None:
                continue
            role = spec.proj_to_role.get(m.group("proj"))
            if role is None:
                raise ValueError(
                    f"{spec.desc}: unknown NVFP4 expert projection {m.group('proj')!r} in {name}"
                )
            loc[(bank_layer, int(m.group("expert")), f"{role}_proj", m.group("kind"))] = (
                name, shard)

        shards = sorted(set(shard for _, shard in loc.values()))
        self.shard_paths = [os.path.join(model_dir, s) for s in shards]
        offsets = {s: _read_safetensors_offsets(os.path.join(model_dir, s)) for s in shards}
        shard_idx = {s: i for i, s in enumerate(shards)}

        E = config.num_experts
        self.num_layers = num_layers
        self.num_experts = E
        seg_size = struct.calcsize("<iqq")  # (shard_idx, offset, nbytes)
        self.entries: list[list[bytes]] = []  # [bank][layer] -> packed segments per expert
        for bank_idx in range(len(_NVP4_BANK_SEGS)):
            per_layer = []
            for layer in range(num_layers):
                rows = bytearray()
                for e in range(E):
                    for proj, kind, _, _ in _NVP4_BANK_SEGS[bank_idx]:
                        key = (layer, e, proj, kind)
                        entry = loc.get(key)
                        if entry is None:
                            raise KeyError(
                                f"disk tier: no {proj}.{kind} tensor for layer {layer} expert {e} "
                                f"(bank {bank_idx}) in {index_path}"
                            )
                        name, shard = entry
                        start, end = offsets[shard][name]
                        rows += struct.pack("<iqq", shard_idx[shard], start, end - start)
                per_layer.append(bytes(rows))
            self.entries.append(per_layer)
        self._seg_size = seg_size

    def row_segments(self, bank_idx: int, layer: int, expert: int) -> list[tuple[int, int, int]]:
        """[(shard_idx, offset, nbytes)] for one expert row, in segment order."""
        base = expert * self._seg_size * len(_NVP4_BANK_SEGS[bank_idx])
        raw = self.entries[bank_idx][layer][base:base + self._seg_size * len(_NVP4_BANK_SEGS[bank_idx])]
        return [
            struct.unpack_from("<iqq", raw, i * self._seg_size)
            for i in range(len(_NVP4_BANK_SEGS[bank_idx]))
        ]

    def row_bytes_per_expert(self) -> int:
        """On-disk bytes of ONE expert across all banks (layer 0).

        Native NVFP4 checkpoint rows are fixed-size, so expert 0 of layer 0 is
        representative; this equals the per-expert pinned-RAM cost per layer."""
        total = 0
        for bank_idx in range(len(_NVP4_BANK_SEGS)):
            total += sum(nbytes for _, _, nbytes in self.row_segments(bank_idx, 0, 0))
        return total


# --moe-disk-tier auto: capacity-adaptive activation -------------------------
# Every rank on the host resolves independently from an early-boot
# /proc/meminfo snapshot (neither rank has allocated its banks yet), so the
# dormant/active decision is quantized with wide hysteresis: dormancy requires
# the FULL expert set to fit with _AUTO_DORMANT_HEADROOM x headroom, and an
# active K leaves (1 - _AUTO_BUDGET_USE) of the budget untouched and aligns
# down to _AUTO_ALIGN. Rank divergence would need the snapshots to differ by
# more than that band within the same boot minute — impossible before either
# rank has allocated. PLE's disk backend is a separate subsystem
# (ple_backend="disk") and is not affected by this resolution.
_AUTO_DORMANT_HEADROOM = 1.25
_AUTO_BUDGET_USE = 0.95
_AUTO_ALIGN = 8  # per-model page-alignment rule (see --expert-ram-experts)


def mem_available_bytes(path: str = "/proc/meminfo") -> int:
    """MemAvailable from /proc/meminfo, in bytes."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024  # kB -> bytes
    raise RuntimeError(f"MemAvailable not found in {path}")


def auto_ram_experts(row_bytes: int, num_layers: int, num_experts: int,
                     available_bytes: int, reserve_bytes: int) -> int | None:
    """Pure resolution math for --moe-disk-tier auto.

    Returns None when the full expert set fits in host RAM (tier dormant: run
    exactly like tier off); otherwise the largest aligned RAM prefix K in
    (0, num_experts) whose pinned cost fits the budget. The host cost of the
    full set spans ALL ranks (single-host TP: every rank's banks together hold
    the global expert set), so ``num_experts`` is the global count.
    """
    budget = available_bytes - reserve_bytes
    if budget <= 0:
        raise ValueError(
            "--moe-disk-tier auto: no RAM budget left after the "
            f"{reserve_bytes / (1 << 30):.0f} GiB reserve "
            f"(MemAvailable={available_bytes / (1 << 30):.1f} GiB)")
    layer_cost = row_bytes * num_layers
    total = layer_cost * num_experts
    if total * _AUTO_DORMANT_HEADROOM <= budget:
        return None
    k = int((budget * _AUTO_BUDGET_USE) // layer_cost)
    k = min(k, num_experts - 1) // _AUTO_ALIGN * _AUTO_ALIGN
    if k <= 0:
        raise ValueError(
            "--moe-disk-tier auto: RAM budget fits fewer than "
            f"{_AUTO_ALIGN} experts per layer "
            f"({budget * _AUTO_BUDGET_USE / (1 << 30):.1f} GiB budget vs "
            f"{layer_cost / (1 << 30):.2f} GiB per expert across "
            f"{num_layers} layers); use --moe-disk-tier on with an explicit "
            "--expert-ram-experts instead")
    return k


def resolve_auto_ram_experts(model_path: str, model_config,
                             reserve_bytes: int) -> tuple[int | None, int, int]:
    """IO wrapper around auto_ram_experts: reads the checkpoint index for the
    per-expert row bytes and /proc/meminfo for MemAvailable.

    Returns (ram_experts_or_None, row_bytes, available_bytes) so the caller can
    log exact numbers. Raises NotImplementedError for non-NVFP4 layouts (same
    rule as the disk-tier loader path)."""
    from freetoken.moe.expert_pieces import nvfp4_expert_spec_of
    from freetoken.models.nvfp4_banks import _num_moe_layers

    spec = nvfp4_expert_spec_of(model_path, model_config)
    if spec is None:
        raise NotImplementedError(
            "--moe-disk-tier auto: expert layers are not in NVFP4 layout; "
            "auto mode supports the NVFP4 checkpoint format only")
    index = Nvfp4DiskIndex(model_path, model_config, spec)
    row_bytes = index.row_bytes_per_expert()
    available = mem_available_bytes()
    resolved = auto_ram_experts(row_bytes, _num_moe_layers(model_config),
                                model_config.num_experts, available, reserve_bytes)
    return resolved, row_bytes, available


def local_ram_experts(ram_experts: int, ownership) -> int:
    """This rank's pinned prefix for a host-wide ``ram_experts`` budget.

    ``ram_experts`` counts the experts per layer the host can pin across ALL
    owner-local ranks (see ``auto_ram_experts``), while the banks it sizes are
    rank-local. Aliasing that count as one global prefix ``[0, ram_experts)``
    leaves every rank past the prefix with zero pinned experts: with the EP=2
    auto budget of 120 on this host, rank 0 pinned 120 experts/ep and rank 1
    pinned none (measured 93.2 GiB vs 4.1 GiB RSS), which is neither what the
    flag documents nor a safe host-memory layout. Split the same budget instead,
    on the alignment grid the tail release needs.

    This balances host RAM; it does NOT shorten a cold prefill. Measured on the
    same 24k-token cold request: 270.1 s with 120/0 against 273.2 s with 64/56,
    because the split leaves the total fetch volume (264 expert rows per layer
    either way) and the per-batch fixed cost unchanged.
    """
    if ownership is None:
        return ram_experts
    world = max(1, getattr(ownership, "world_size", 1))
    local_num = ownership.local_num_experts
    share, rem = divmod(ram_experts, world)
    if share >= _AUTO_ALIGN:
        # Stay on the page-alignment grid; leftover whole units go to the
        # lowest ranks so the host-wide total is preserved.
        share = share // _AUTO_ALIGN * _AUTO_ALIGN
        rem = ram_experts - share * world
        if ownership.rank < rem // _AUTO_ALIGN:
            share += _AUTO_ALIGN
        return min(local_num, share)
    # Budgets below one alignment unit cannot be split on the grid.
    return min(local_num, share + (1 if ownership.rank < rem else 0))


class DiskTier:
    """Runtime fetcher: disk-resident slot-cache misses -> staging -> GPU slot."""

    def __init__(self, index: Nvfp4DiskIndex, cache, ram_experts: int, workers: int = 8,
                 ownership=None, pin_rows: list[list[int]] | None = None) -> None:
        self._index = index
        # Owner-local EP: the slot cache (and therefore every miss/fetch/slot
        # identifier below) lives in the LOCAL expert namespace
        # [0, ownership.local_num_experts), while the disk index spans the GLOBAL
        # checkpoint rows [0, index.num_experts). ``_g0`` is the local->global
        # offset applied at every index access; ``_ram`` is how many of THIS
        # rank's local experts are pinned, in the LOCAL namespace.
        # ``ram_experts`` arrives already split: the loader resolves the
        # host-wide budget into a per-rank share with ``local_ram_experts`` and
        # hands that same number to ``release_bank_tails`` and to
        # ``attach_disk_tier``. Splitting again here would divide the share a
        # second time (measured on the EP=2 auto budget: 120 -> 64/56 -> 32/24).
        # With ownership=None both collapse to the identity (global) mapping.
        self._g0 = ownership.global_start if ownership is not None else 0
        self._local_num = (ownership.local_num_experts if ownership is not None
                           else index.num_experts)
        self._ram = min(self._local_num, ram_experts)
        # ---- pinned-row layout: local id -> bank row ---------------------------
        # Default (pin_rows=None): the identity map -- local expert e sits in bank
        # row e, so "RAM-resident" is the contiguous prefix ``e < _ram`` and every
        # consumer below degenerates to exactly the pre-pin-set behavior.
        # With a learned pin set (--moe-ram-pin-file), the loader wrote pinned
        # expert ``pin_rows[L][r]`` into bank row ``r < _ram`` (see
        # expert_banks), and every residency test here goes through the per-layer
        # permutation ``_row_map``: RAM-resident iff ``_row_map[L][e] < _ram``.
        # ``_pin_ids``/``_nonpin`` are the inverse map restricted to the two row
        # ranges (identity slot of the expert in bank row r / the slots the
        # prefill phantom cleanup must clear).
        num_layers = index.num_layers
        if pin_rows is None:
            row_map = None
        else:
            if len(pin_rows) != num_layers or any(len(r) != self._ram for r in pin_rows):
                raise ValueError(
                    f"pin_rows must be {num_layers} layer lists of exactly "
                    f"{self._ram} local ids")
            row_map = pin_rows_to_row_map(pin_rows, self._local_num)
        if row_map is None:
            rm = torch.arange(self._local_num, dtype=torch.int32).expand(num_layers, -1)
        else:
            rm = torch.tensor(row_map, dtype=torch.int32)
        self._row_map_cpu = rm.contiguous()
        # A pin set that IS the contiguous prefix yields the identity map: keep
        # the no-pin-file fast paths byte-identical for it (``_remapped`` gates
        # the scatter/lookup branches everywhere below).
        self._remapped = bool(
            (self._row_map_cpu != torch.arange(self._local_num, dtype=torch.int32))
            .any().item())
        inv = torch.empty_like(self._row_map_cpu)
        inv.scatter_(1, self._row_map_cpu.long(),
                     torch.arange(self._local_num, dtype=torch.int32).expand(num_layers, -1))
        self._ids_by_row_cpu = inv.to(torch.int32).contiguous()
        self._pin_ids_cpu = self._ids_by_row_cpu[:, :self._ram].contiguous()
        device = cache.banks[0][1].device  # == self._banks, rebound below
        self._row_map_dev = self._row_map_cpu.to(device)
        self._pin_ids_dev = self._pin_ids_cpu.to(device)
        self._pin_ids_dev_i64 = self._pin_ids_dev.long()
        self._nonpin_dev = self._ids_by_row_cpu[:, self._ram:].contiguous().to(device).long()
        self._graph_bridge = None
        self._banks = list(cache.banks)  # [(per_layer_host, gpu_cache)] in schema order
        # ---- DS-FP4 conversion mode (triton_dsfp4 kernel): the slot cache and the
        # pinned RAM banks hold DS-FP4 rows (e2m1 + per-32 e8m0, global folded) while
        # the checkpoint rows the index addresses stay native NVFP4. Packed banks map
        # 1:1 (same bytes); scale extents are read whole and repacked on the host by
        # nvfp4_to_dsfp4.convert_scale_rows before the H2D. ``_disk_bank`` translates
        # cache-bank index -> NVFP4 disk-bank index; ``_scale_convert`` names, per
        # cache scale bank, (its disk scale bank, the scalar-preload bank with the
        # per-expert globals the fold needs).
        self._convert = getattr(cache, "quant_format", None) == "ds_fp4"
        self._disk_bank = (0, 1, 3, 4) if self._convert else None
        self._scale_convert = {1: (1, 2), 3: (4, 5)} if self._convert else {}
        self._row_bytes = [
            b[0][0][0].numel() * b[0][0][0].element_size() for b in self._banks
        ]  # full expert-row bytes per bank (staging must hold the biggest one)
        # Per-bank destination row slices (gate|up split at the row midpoint), keyed
        # by the DISK bank's segment count (identical to the cache bank's outside
        # conversion mode; inside it the cache banks are the DS-FP4 four and the disk
        # banks the NVFP4 six, so the lookup must go through ``_disk_bank``).
        self._dst_slices: list[list[tuple[int, int]]] = []
        for bank_idx, (host_layer, _gpu) in enumerate(self._banks):
            row = host_layer[0][0]
            disk_bank = self._disk_bank[bank_idx] if self._convert else bank_idx
            if len(_NVP4_BANK_SEGS[disk_bank]) == 2:
                mid = row.shape[0] // 2
                self._dst_slices.append([(0, mid), (mid, row.shape[0])])
            else:
                self._dst_slices.append([(0, row.shape[0])])
        if os.environ.get("FT_DISK_TIER_VERIFY"):
            # TP=2 debug: prove per-rank whether the host bank rows are full or
            # TP-sharded, and that the disk index's full-row segments match them.
            from freetoken.distributed import try_get_tp_info
            tp = try_get_tp_info()
            host_shapes = [tuple(b[0][0][0].shape) for b in self._banks]
            disk_bytes = [
                sum(nb for _, _, nb in self._index.row_segments(bi, 0, 0))
                for bi in range(len(self._banks))
            ]
            print(f"[disk-tier-init] tp_rank={getattr(tp, 'rank', '?')}/{getattr(tp, 'size', '?')} "
                  f"ram={self._ram} g0={self._g0} local_num={self._local_num} host_row_shapes={host_shapes} "
                  f"host_row_bytes={self._row_bytes} disk_row_bytes={disk_bytes}", flush=True)
        max_row = max(self._row_bytes)
        self._fd_lock = threading.Lock()
        self._fds: dict[int, tuple[int, bool]] = {}
        if not self._convert:
            # R1x: cross-bank merging makes the worst-case READ extent bigger
            # than any single bank row (v41: the contiguous w1|w2|3 run is
            # ~17.7 MiB vs the biggest bank's 5.9). Every staging/bounce
            # buffer derives from _staging_size, and the layout is uniform
            # across rows/layers, so one probe row sizes them all. Runs after
            # _fds init: _row_groups -> _group_runs -> _fd.
            max_row = max(max_row, max(
                a1 - a0 for _s, a0, a1, _m, _e in self._row_groups(0, 0)))
        self._staging_size = ((max_row + _ALIGN - 1) // _ALIGN + 2) * _ALIGN
        self._staging = threading.local()
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="disk-tier")
        self._fetches = 0
        self._fetch_bytes = 0
        self._preadv_calls = 0
        # Doorbell (graph_fetch) telemetry: the doorbell path never touches
        # _fetches/_fetch_bytes (those are eager-only), so decode fetch volume
        # was a blind spot (W22). The bridge reports each served request here.
        # _db_row_bytes = disk bytes per expert row across all cache banks.
        # Convert mode counts each scale bank's full native row (the doorbell
        # actually reads half) -- known ~7% over-count in the experimental
        # ds_fp4 mode; exact for native.
        self._db_row_bytes = 0
        for _bi in range(len(self._banks)):
            _db = self._disk_bank[_bi] if self._convert else _bi
            self._db_row_bytes += sum(
                nb for _, _, nb in self._index.row_segments(_db, 0, 0))
        self._db_requests = 0
        self._db_rows = 0
        self._db_bytes = 0
        self._db_host_us = 0
        self._scalar_preload_reads = 0
        self._scalars: dict | None = None
        self._scalars_lock = threading.Lock()
        self._decode_verify_steps = {}  # layer_id -> verified step count
        self._map_verify_steps = {}  # layer_id -> verified step count
        self._dbg_fetched = {}  # layer_id -> set of experts disk-fetched on the latest call
        self._cache = cache
        # ---- PILOT prefetch (P0-4 port of colibri's router-guided cross-layer
        # prefetch): at layer L the decode path hands us L's raw routing; we read
        # the predicted experts for L+1 into pinned host SLABS. A stash entry never
        # touches a GPU slot, so it cannot evict the current demand set (P0-1) and
        # its (layer, expert) key suppresses duplicate reads vs the on-demand path
        # (P0-2 reservation). Layer L+1's fetch_pending consumes stash hits with a
        # fast pinned->slot H2D instead of an NVMe round trip.
        # Default OFF (2026-10-03 measurement). The 2026-09-28 decision turned this on
        # because it is *output-neutral*, not because it pays: the identity-map predictor
        # is only 1.01% precise on DeepSeek-V4.1-Flash-NVFP4 (adjacent-layer top-k overlap,
        # measured over two real decode route traces / 12k+ samples; an offline learned
        # per-layer co-occurrence table -- colibri's COUPLE -- reaches 20.1%), and colibri's
        # cited 62.3% top-8 overlap is a DSv4 figure that does not transfer. An eager A/B
        # confirmed the consequence: prefetch_issued 6799 / prefetch_hits 27 / wasted 6772,
        # i.e. ~99% of the issued reads are NVMe bandwidth stolen from the demand path on a
        # device that is already saturated. Set FT_DISK_TIER_PREFETCH=1 to opt back in.
        self._prefetch_window = int(os.environ.get("FT_DISK_TIER_PREFETCH", "0"))
        if self._convert and self._prefetch_window > 0:
            # The PILOT slab machinery is written against the native 6-bank NVFP4
            # layout (scalar-bank skip, no scale repack). It is default-OFF and
            # measured NO-GO on v41 (1.01% predictor precision); rather than teach
            # dead code the conversion, refuse the combination.
            print("[disk-tier] DS-FP4 conversion mode: PILOT prefetch unsupported, "
                  "forcing FT_DISK_TIER_PREFETCH=0", flush=True)
            self._prefetch_window = 0
        self._prefetch_pool = (
            ThreadPoolExecutor(max_workers=2, thread_name_prefix="disk-tier-pf")
            if self._prefetch_window > 0 else None
        )
        self._stash: dict = {}            # (layer, expert) -> _StashEntry
        self._stash_lock = threading.Lock()
        self._slab_free: list = []        # [(slab tensor, cuda event|None)]
        self._slabs_allocated = 0
        # P1 telemetry (EMAP): decayed per-(layer, expert) routing mass. Observed
        # at prefetch_from_routing, which sees every layer's RAW routing in decode.
        self._route_hist = torch.zeros(
            (index.num_layers, index.num_experts), dtype=torch.float64)
        # LFRU recency clock (per decode token at layer 0) + last-access stamps.
        self._route_last = torch.zeros(
            (index.num_layers, index.num_experts), dtype=torch.int64)
        self._route_clock = 0
        self._hist_since_decay = 0
        self._slab_count = int(os.environ.get("FT_DISK_TIER_PREFETCH_SLABS", "32"))
        # Host mirror of slot occupancy: (layer, slot) -> expert. fetch_pending sees
        # every miss->slot assignment, which is enough to keep this exact; used to
        # skip prefetching experts that are already slot-resident.
        self._slot_owner: dict = {}
        self._resident_disk: list[set] = [set() for _ in range(index.num_layers)]
        self._pf_issued = 0
        self._pf_hits = 0
        self._pf_wasted = 0
        self._pf_skipped_resident = 0
        # HITS telemetry (colibri 7.7): per-turn bitmap of which experts were
        # served and from WHERE -- 0=RAM resident, 1=stash hit, 2=disk read.
        # -1 = not routed this turn. end_turn() snapshots+resets it and, when
        # FT_DISK_TIER_TELEMETRY=<jsonl path> is set, appends the turn record.
        self._turn_src = torch.full(
            (index.num_layers, self._local_num), -1, dtype=torch.int8)
        self._turn = 0
        self._telemetry_path = os.environ.get("FT_DISK_TIER_TELEMETRY") or None

    # ------------------------------------------------------------------ fds
    def _fd(self, shard_idx: int) -> tuple[int, bool]:
        """(fd, o_direct) for a shard; O_DIRECT falls back to plain preadv where the
        filesystem refuses it (tmpfs/overlayfs -- tests)."""
        ent = self._fds.get(shard_idx)
        if ent is None:
            with self._fd_lock:
                ent = self._fds.get(shard_idx)
                if ent is None:
                    path = self._index.shard_paths[shard_idx]
                    try:
                        fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
                        direct = True
                    except OSError:
                        fd = os.open(path, os.O_RDONLY)
                        direct = False
                    ent = (fd, direct)
                    self._fds[shard_idx] = ent
        return ent

    # --------------------------------------------------------------- staging
    # Staging ring depth per worker thread. A buffer must not be overwritten by the
    # next preadv until the async H2D copy that read it has finished (pinned-memory
    # reuse race -- the copy is DMA, still reading host bytes after copy_ returns).
    # The depth only needs to cover one copy's DMA time in host-side preadv time;
    # the per-slot CUDA event below makes any shallower lap correct, just slower.
    _STAGING_RING = 8

    def init_graph_bridge(self, cache, k_max: int) -> None:
        """Allocate the graph-doorbell fetch bridge (CUDA-graph decode with the
        tier on). k_max bounds disk misses per layer = max graph bs x topk."""
        from .graph_fetch import GraphFetchBridge
        self._graph_bridge = GraphFetchBridge(self, cache, k_max)

    def graph_stage_fetch(self, cache, layer_id: int) -> None:
        """Capture-time only: record request+spin+install kernels so replay can
        fetch disk-resident experts via the host service thread."""
        self._graph_bridge.stage_fetch(cache, layer_id)

    def _staging_ring(self) -> list:
        ring = getattr(self._staging, "ring", None)
        if ring is None:
            # Fresh worker threads default to CUDA device 0, but the rank may
            # live on another device (TP>1). The H2D copies land on the
            # destination tensor's device stream, while ev.record() below uses
            # the thread's CURRENT stream -- without this, the ring's reuse
            # guard waits on an idle stream and a preadv can overwrite the
            # buffer mid-DMA (corrupted slot rows on TP=2 rank 1).
            dev = self._banks[0][1].device
            if dev.type == "cuda":
                torch.cuda.set_device(dev)
            ring = []
            for _ in range(self._STAGING_RING):
                buf = HostBank((self._staging_size,), torch.uint8)
                buf.pin()  # pin once per worker thread
                ev = torch.cuda.Event() if torch.cuda.is_available() else None
                if ev is not None:
                    ev.record()  # start "complete"; re-recorded after each copy
                ring.append([buf, ev])
            self._staging.ring = ring
        return ring

    def _scalar_blob(self, layer: int, bank_idx: int) -> bytes:
        sc = self._scalars
        if sc is None:
            with self._scalars_lock:
                if self._scalars is None:
                    self._scalars = self._preload_scalars()
                sc = self._scalars
        return sc[(layer, bank_idx)]

    def _preload_scalars(self) -> dict:
        """Read the scalar-bank (weight_scale_2) values for THIS rank's local
        expert window ``[_g0, _g0 + _local_num)`` ONCE, merging exactly-adjacent
        file runs to keep startup syscalls low. One fp32 scalar per segment;
        blob layout per (layer, bank): local-expert-major, each expert's segments
        concatenated in index order. With ``ownership=None`` the window is the
        whole global range, which is byte-identical to reading every expert."""
        scalar_banks = getattr(self._index, "scalar_banks", ())
        out: dict = {}
        if not scalar_banks:
            return out
        fds = [os.open(p, os.O_RDONLY) for p in self._index.shard_paths]
        try:
            for layer in range(self._index.num_layers):
                for bank_idx in scalar_banks:
                    stride = sum(nb for _, _, nb in
                                 self._index.row_segments(bank_idx, layer, 0))
                    blob = bytearray(stride * self._local_num)
                    flat = []  # (shard_idx, file_off, nbytes, blob_pos)
                    for e in range(self._g0, self._g0 + self._local_num):
                        pos = (e - self._g0) * stride
                        for (shard_idx, off, nb) in self._index.row_segments(
                                bank_idx, layer, e):
                            flat.append((shard_idx, off, nb, pos))
                            pos += nb
                    flat.sort()
                    i = 0
                    while i < len(flat):
                        shard_idx, off, nb, dst = flat[i]
                        end = off + nb
                        run = [(dst, nb)]
                        j = i + 1
                        while (j < len(flat) and flat[j][0] == shard_idx
                               and flat[j][1] == end):
                            run.append((flat[j][3], flat[j][2]))
                            end += flat[j][2]
                            j += 1
                        data = os.pread(fds[shard_idx], end - off, off)
                        self._scalar_preload_reads += 1
                        p = 0
                        for dst, nb in run:
                            blob[dst:dst + nb] = data[p:p + nb]
                            p += nb
                        i = j
                    out[(layer, bank_idx)] = bytes(blob)
        finally:
            for fd in fds:
                os.close(fd)
        return out

    # ---------------------------------------------------------------- fetch
    def _fill_scalar_row(self, bank_idx: int, layer: int, expert: int,
                         row: torch.Tensor) -> None:
        """Global-scale banks (weight_scale_2): fill the row from the preloaded
        per-expert fp32 scalar blob -- no disk read, no staging. ``expert`` is a
        LOCAL (slot-cache) id; the blob is keyed by that same local index (see
        ``_preload_scalars``), while the disk layout lookup needs the global row.
        The assert guards the callers that trust ``cache.src_indices``'s local
        namespace without an explicit runtime re-check (``fetch_pending``)."""
        assert 0 <= expert < self._local_num, (
            f"scalar fill outside the local expert window: expert={expert} "
            f"local_num={self._local_num}")
        segs = self._index.row_segments(bank_idx, layer, expert + self._g0)
        stride = sum(nb for _, _, nb in segs)
        blob = self._scalar_blob(layer, bank_idx)
        base = expert * stride  # blob is local-expert-major, not global
        for k, (d0, d1) in enumerate(self._dst_slices[bank_idx]):
            val = struct.unpack_from("<f", blob, base + 4 * k)[0]
            row[d0:d1].fill_(val)

    def _group_runs(self, bank_idx: int, layer: int, expert: int,
                    disk_bank: int | None = None):
        """Merged preadv groups for one non-scalar bank: [(shard, a0, a1,
        [(d0, d1, off, nbytes)])]. Sorts by file position and merges EXACTLY
        adjacent segments into one read (P0-3). ``disk_bank`` (DS-FP4 conversion
        mode) decouples the checkpoint row being read from the cache bank whose
        ``_dst_slices`` the members target."""
        expert = expert + self._g0  # local (slot-cache) id -> global checkpoint row
        segs = self._index.row_segments(bank_idx if disk_bank is None else disk_bank,
                                        layer, expert)
        runs = []
        for (d0, d1), (shard_idx, off, nbytes) in zip(
                self._dst_slices[bank_idx], segs):
            fd, direct = self._fd(shard_idx)
            if direct:
                a0 = off & ~(_ALIGN - 1)
                a1 = (off + nbytes + _ALIGN - 1) & ~(_ALIGN - 1)
            else:
                a0, a1 = off, off + nbytes
            runs.append((shard_idx, a0, a1, off, nbytes, d0, d1))
        runs.sort(key=lambda t: (t[0], t[3]))
        groups = []  # [shard_idx, a0, a1, [(d0, d1, off, nbytes)], exact_end]
        for shard_idx, a0, a1, off, nbytes, d0, d1 in runs:
            if (groups and groups[-1][0] == shard_idx
                    and groups[-1][4] == off):
                groups[-1][2] = max(groups[-1][2], a1)
                groups[-1][4] = off + nbytes
                groups[-1][3].append((d0, d1, off, nbytes))
            else:
                groups.append([shard_idx, a0, a1,
                               [(d0, d1, off, nbytes)], off + nbytes])
        return groups

    def _row_groups(self, layer: int, expert: int):
        """Cross-bank merged read plan for one row -- NATIVE mode only (R1x).

        Same [shard, a0, a1, members, exact_end] shape as _group_runs, but the
        members carry their bank: (bank_idx, d0, d1, off, nbytes). The v41
        checkpoint lays each expert's w1|w2|w3 weights down as one contiguous
        run (and the three per-16 scales as another), while _group_runs only
        merges within a bank -- so a row costs 6 preadv. Merging exactly
        adjacent runs ACROSS banks (same shard, prev exact_end == next off,
        the within-bank rule) cuts that to 2 syscalls with identical bytes;
        non-adjacent layouts simply don't merge. Scalar banks stay blob fills
        and are not part of the plan."""
        scalar_banks = getattr(self._index, "scalar_banks", ())
        runs = []
        for bank_idx in range(len(self._banks)):
            if bank_idx in scalar_banks:
                continue
            for shard_idx, a0, a1, members, _end in self._group_runs(
                    bank_idx, layer, expert):
                for (d0, d1, off, nbytes) in members:
                    runs.append((shard_idx, a0, a1, off, nbytes,
                                 bank_idx, d0, d1))
        runs.sort(key=lambda t: (t[0], t[3]))
        groups = []  # [shard, a0, a1, [(bank_idx, d0, d1, off, nbytes)], exact_end]
        for shard_idx, a0, a1, off, nbytes, bank_idx, d0, d1 in runs:
            if (groups and groups[-1][0] == shard_idx
                    and groups[-1][4] == off):
                g = groups[-1]
                g[2] = max(g[2], a1)
                g[4] = off + nbytes
                g[3].append((bank_idx, d0, d1, off, nbytes))
            else:
                groups.append([shard_idx, a0, a1,
                               [(bank_idx, d0, d1, off, nbytes)],
                               off + nbytes])
        return groups

    def _fetch_expert(self, layer: int, expert: int, slot: int,
                      dst_buffers: list | None = None, buffer_id: int = 0) -> None:
        # The server runs under inference_mode; the fetch pool threads do not,
        # so the H2D writes into the (inference) slot cache need their own scope.
        with torch.inference_mode():
            self._fetch_expert_inner(layer, expert, slot, dst_buffers, buffer_id)

    def _fetch_expert_inner(self, layer: int, expert: int, slot: int,
                            dst_buffers: list | None = None,
                            buffer_id: int = 0) -> None:
        ring = self._staging_ring()
        ri = getattr(self._staging, "ri", 0)
        scalar_banks = getattr(self._index, "scalar_banks", ())
        disk_bytes = 0
        if not self._convert:
            # R1x native path: blob-fill the scalar banks, then serve every
            # disk bank from ONE cross-bank merged plan (6 -> 2 preadv/row on
            # the v41 layout; identical bytes, see _row_groups).
            rows = {}
            for bank_idx, (_host_layer, gpu_cache) in enumerate(self._banks):
                if dst_buffers is None:
                    row = gpu_cache[slot]
                else:
                    row = dst_buffers[bank_idx][buffer_id][expert]
                if bank_idx in scalar_banks:
                    self._fill_scalar_row(bank_idx, layer, expert, row)
                else:
                    rows[bank_idx] = row
            for shard_idx, a0, a1, members, _end in self._row_groups(
                    layer, expert):
                staging, ev = ring[ri]
                if ev is not None:
                    ev.synchronize()
                ri = (ri + 1) % len(ring)
                fd, direct = self._fd(shard_idx)
                slen = a1 - a0
                mv = (ctypes.c_char * slen).from_address(staging.addr)
                try:
                    os.preadv(fd, [mv], a0)
                except OSError:
                    raise _preadv_error(self, staging, shard_idx,
                                        members[0][3], a0, slen, direct)
                self._preadv_calls += 1
                for bank_idx, d0, d1, off, nbytes in members:
                    src = staging.tensor[off - a0:off - a0 + nbytes]
                    dst = rows[bank_idx][d0:d1]
                    dst.copy_(src.view(dst.dtype).view(dst.shape),
                              non_blocking=True)
                    disk_bytes += nbytes
                if ev is not None:
                    ev.record()
            self._staging.ri = ri
            self._fetches += 1
            self._fetch_bytes += sum(self._row_bytes)
            return
        for bank_idx, (_host_layer, gpu_cache) in enumerate(self._banks):
            if dst_buffers is None:
                row = gpu_cache[slot]
            else:
                # Overlap prefill: write directly into the borrowed buffer row
                # (position == expert id); the identity slots are untouched.
                row = dst_buffers[bank_idx][buffer_id][expert]
            if not self._convert and bank_idx in scalar_banks:
                # Native mode only: cache bank == disk bank, and the global-scale
                # banks are blob fills. In conversion mode the (0..3) cache banks
                # would collide with the NVFP4 scalar ids {2, 5} -- every cache row
                # here is a real disk read.
                self._fill_scalar_row(bank_idx, layer, expert, row)
                continue
            if bank_idx in self._scale_convert:
                ri, disk_bytes = self._fetch_scale_convert(
                    layer, expert, row, bank_idx, ring, ri, disk_bytes)
                continue
            groups = self._group_runs(
                bank_idx, layer, expert,
                disk_bank=self._disk_bank[bank_idx] if self._convert else None)
            for shard_idx, a0, a1, members, _exact_end in groups:
                staging, ev = ring[ri]
                if ev is not None:
                    # This buffer's last async H2D copy must be done before the
                    # preadv below overwrites it (pinned-memory reuse race).
                    ev.synchronize()
                ri = (ri + 1) % len(ring)
                fd, direct = self._fd(shard_idx)
                slen = a1 - a0
                mv = (ctypes.c_char * slen).from_address(staging.addr)
                try:
                    os.preadv(fd, [mv], a0)
                except OSError:
                    raise _preadv_error(self, staging, shard_idx,
                                        members[0][2], a0, slen, direct)
                self._preadv_calls += 1
                for d0, d1, off, nbytes in members:
                    src = staging.tensor[off - a0:off - a0 + nbytes]
                    dst = row[d0:d1]
                    dst.copy_(src.view(dst.dtype).view(dst.shape),
                              non_blocking=True)
                    disk_bytes += nbytes
                if ev is not None:
                    # Arm: the next reuse of this buffer waits for this copy.
                    ev.record()
        self._staging.ri = ri
        self._fetches += 1
        # Conversion mode reports the NVMe-side bytes actually read (the NVFP4
        # extents incl. the dropped odd scale columns); the native path keeps the
        # cache-row accounting it always had.
        self._fetch_bytes += disk_bytes if self._convert else sum(self._row_bytes)

    def _fetch_scale_convert(self, layer: int, expert: int, row: torch.Tensor,
                             cache_bank: int, ring: list, ri: int,
                             disk_bytes: int) -> tuple[int, int]:
        """DS-FP4 conversion mode: read one NVFP4 scale extent (per-16 e4m3) and
        repack it to per-32 e8m0 with the per-expert global folded into the
        exponent (moe.nvfp4_to_dsfp4), then hand the smaller row to the H2D copy.
        The repack runs on the host between the preadv and the copy; the sync
        copy from the pageable converter output costs ~100us per row, noise
        against the ~19MB row's disk time, and adds zero buffer-reuse races."""
        from freetoken.moe.nvfp4_to_dsfp4 import convert_scale_rows, global_exponent

        disk_bank, blob_bank = self._scale_convert[cache_bank]
        segs = self._index.row_segments(disk_bank, layer, expert + self._g0)
        slices = self._dst_slices[cache_bank]
        if len(segs) != len(slices):
            raise RuntimeError(
                f"scale convert: {len(segs)} disk segments vs {len(slices)} dst "
                f"slices (cache bank {cache_bank}, disk bank {disk_bank})")
        # Per-expert globals from the scalar preload blob: fp32, local-expert-major,
        # one value per segment (gate, up) or a single one (down).
        blob = self._scalar_blob(layer, blob_bank)
        vals = struct.unpack_from(f"<{len(segs)}f", blob, expert * 4 * len(segs))
        for k, (shard_idx, off, nbytes) in enumerate(segs):
            d0, d1 = slices[k]
            rows = d1 - d0
            if nbytes % rows:
                raise RuntimeError(
                    f"scale convert: segment {nbytes}B not divisible by {rows} rows")
            staging, ev = ring[ri]
            if ev is not None:
                ev.synchronize()
            ri = (ri + 1) % len(ring)
            fd, direct = self._fd(shard_idx)
            if direct:
                a0 = off & ~(_ALIGN - 1)
                a1 = (off + nbytes + _ALIGN - 1) & ~(_ALIGN - 1)
            else:
                a0, a1 = off, off + nbytes
            slen = a1 - a0
            mv = (ctypes.c_char * slen).from_address(staging.addr)
            try:
                os.preadv(fd, [mv], a0)
            except OSError:
                raise _preadv_error(self, staging, shard_idx, off, a0, slen, direct)
            self._preadv_calls += 1
            src = (staging.tensor[off - a0:off - a0 + nbytes].numpy()
                   .reshape(rows, nbytes // rows))
            conv = convert_scale_rows(src, global_exponent(vals[k]))
            dst = row[d0:d1]
            dst.copy_(torch.from_numpy(conv).view(dst.dtype).view(dst.shape))
            disk_bytes += nbytes
        return ri, disk_bytes

    def _sync_fetches(self) -> None:
        """Wait for the pool threads' async H2D copies to land.

        The copies are enqueued on the pool threads' default stream; f.result() only
        waits for them to be ENQUEUED. The GEMM's stream is not ordered with that
        stream, so sync the default stream before the GEMM reads the slots."""
        # Key this on where the BANKS live, not on whether the machine has a GPU: the
        # disk-tier unit tests build CPU banks on a CUDA box, and default_stream() rejects
        # a CPU device outright. There is nothing to order for a CPU->CPU copy anyway.
        device = self._banks[0][1].device
        if device.type != "cuda":
            return  # CPU banks: the copies are synchronous CPU->CPU
        torch.cuda.default_stream(device).synchronize()

    def _verify_slot(self, cache, layer: int, expert: int, slot: int | None = None,
                     phase: str = "prefill") -> None:
        """One-shot debug: read back an expert's slot rows and compare against the
        checkpoint bytes (ground truth). Gated on FT_DISK_TIER_VERIFY."""
        import torch
        if slot is None:
            slot = expert  # identity mapping (prefill)
        for bank_idx, (_host_layer, gpu_cache) in enumerate(self._banks):
            slot_row = gpu_cache[slot].contiguous()
            flat = slot_row.view(torch.uint8).reshape(-1)
            ref = self._ref_row(bank_idx, layer, expert, flat.numel(),
                                slot_row.element_size(),
                                slot_row.numel() // slot_row.shape[0] if slot_row.dim() > 1 else 1)
            try:
                ref = ref.to(flat.device)
                match = bool(torch.equal(flat, ref))
                if match:
                    print(f"[verify] {phase} L{layer} bank={bank_idx} expert={expert} "
                          f"slot={slot} match=True", flush=True)
                    continue
                diff = (flat != ref)
                nz = torch.nonzero(diff).flatten()
                print(f"[verify] {phase} L{layer} bank={bank_idx} expert={expert} "
                      f"slot={slot} match=False n_diff={int(diff.sum())}/{flat.numel()} "
                      f"first_off={int(nz[0])} last_off={int(nz[-1])} "
                      f"slot_norm={slot_row.float().norm().item():.4f} "
                      f"ref_norm={ref.view(slot_row.dtype).view(slot_row.shape).float().norm().item():.4f} "
                      f"slot_head={flat[:8].tolist()} ref_head={ref[:8].tolist()}", flush=True)
                self._identify_overwriter(layer, bank_idx, expert, flat)
            except Exception as exc:  # never crash the server in debug
                print(f"[verify] bank={bank_idx} expert={expert} ERROR {exc!r}", flush=True)

    def _ref_row(self, bank_idx: int, layer: int, expert: int, row_bytes: int,
                 row_el: int, row_leading: int) -> torch.Tensor:
        """Reference row bytes for (bank, layer, expert) straight from the checkpoint.
        DS-FP4 conversion mode: packed banks read the identical NVFP4 extent; scale
        banks are repacked (per-16 e4m3 + per-expert fp32 global) -> per-32 e8m0 so
        the reference matches the DS-FP4 row the slot is expected to hold."""
        expert = expert + self._g0  # local (slot-cache) id -> global checkpoint row
        disk_bank = self._disk_bank[bank_idx] if self._convert else bank_idx
        ref = torch.zeros(row_bytes, dtype=torch.uint8)
        segs = self._index.row_segments(disk_bank, layer, expert)
        gexp = None
        if bank_idx in self._scale_convert:
            from freetoken.moe.nvfp4_to_dsfp4 import convert_scale_rows, global_exponent
            import numpy as _np2
            blob = self._scalar_blob(layer, self._scale_convert[bank_idx][1])
            # NOTE: `expert` was already globalized above; the blob is local-major.
            vals = struct.unpack_from(f"<{len(segs)}f", blob,
                                      (expert - self._g0) * 4 * len(segs))
            gexp = [global_exponent(v) for v in vals]
        for k, ((d0, d1), (shard_idx, off, nbytes)) in enumerate(
                zip(self._dst_slices[bank_idx], segs)):
            fd, direct = self._fd(shard_idx)
            a0 = off if not direct else (off & ~(_ALIGN - 1))
            slen = nbytes if not direct else (off + nbytes - a0 + _ALIGN - 1) & ~(_ALIGN - 1)
            buf = os.pread(fd, slen, a0)
            row_off = off - a0
            seg = buf[row_off:row_off + nbytes]
            if gexp is not None:
                src = _np2.frombuffer(seg, dtype=_np2.uint8).reshape(d1 - d0, -1)
                seg = convert_scale_rows(src, gexp[k]).tobytes()
            elif not self._convert and bank_idx in (2, 5):
                import struct as _st
                import numpy as _np
                f16 = _np.float16(_st.unpack("<f", seg[:4])[0]).tobytes()
                for r in range(d0, d1):
                    ref[r * row_el:(r + 1) * row_el] = torch.frombuffer(f16, dtype=torch.uint8)
                continue
            dst_off = d0 * row_leading * row_el
            ref[dst_off:dst_off + len(seg)] = torch.frombuffer(seg, dtype=torch.uint8)
        return ref

    def _identify_overwriter(self, layer: int, bank_idx: int, expert: int,
                             flat: torch.Tensor) -> None:
        """Debug: on a verify mismatch, find whose row the slot actually holds.

        Compares the slot's first 64 bytes against (a) every other expert of the
        same layer and (b) the same expert in every other layer, straight from the
        checkpoint. A hit names the overwriter (e.g. a staging-buffer reuse race
        landing a neighbour's preadv); no hit means a partial mix. Only runs on
        mismatch, so the ~300 extra preads are free otherwise."""
        head = flat[:64].cpu()
        host_row = self._banks[bank_idx][0][0][0]  # expert-0 row (all rows share its shape)
        row_el = host_row.element_size()
        row_leading = host_row.numel() // host_row.shape[0] if host_row.dim() > 1 else 1
        num_experts = self._cache.num_experts
        num_layers = len(self._banks[bank_idx][0])
        hits = []
        for e in range(num_experts):
            if e == expert:
                continue
            ref = self._ref_row(bank_idx, layer, e, flat.numel(), row_el, row_leading)
            if bool(torch.equal(ref[:64], head)):
                hits.append(f"L{layer}_e{e}")
        for L in range(num_layers):
            if L == layer:
                continue
            ref = self._ref_row(bank_idx, L, expert, flat.numel(), row_el, row_leading)
            if bool(torch.equal(ref[:64], head)):
                hits.append(f"L{L}_e{expert}")
        print(f"[overwriter] L{layer} B{bank_idx} e{expert} head64 matches: "
              f"{hits if hits else 'NONE (partial mix?)'}", flush=True)

    def verify_ram(self, cache, layer: int) -> None:
        """One-shot debug: after the PCIe copy, check a RAM-resident expert's slot rows
        against the checkpoint reference. Gated on FT_DISK_TIER_VERIFY."""
        pin_ids = self._pin_ids_cpu[layer]
        expert = int(pin_ids[min(10, self._ram - 1)])  # a RAM-resident expert
        print(f"[verify-ram] layer={layer} expert={expert} (pinned row "
              f"{min(10, self._ram - 1)})", flush=True)
        self._verify_slot(cache, layer, expert)
        # Check ALL RAM experts: slot vs host row (host correctness established
        # separately). Count mismatches; identify the source of the first one.
        # Bank row ``r`` holds expert ``pin_ids[r]`` (identity map without a pin
        # file); its identity slot is the expert id.
        n_bad = 0
        identified = False
        for r in range(self._ram):
            e = int(pin_ids[r])
            for bank_idx, (host_layer, gpu_cache) in enumerate(self._banks):
                slot_row = gpu_cache[e].contiguous()
                flat = slot_row.view(torch.uint8).reshape(-1)
                hflat = host_layer[layer][r].contiguous().view(torch.uint8).reshape(-1)
                if flat.numel() != hflat.numel() or not bool(torch.equal(flat.cpu(), hflat)):
                    n_bad += 1
                    if n_bad <= 12:
                        print(f"[verify-ram] MISMATCH e={e} row={r} bank={bank_idx} "
                              f"slot_head={flat[:8].tolist()} host_head={hflat[:8].tolist()}",
                              flush=True)
                    if not identified:
                        identified = True
                        self._identify_source(layer, bank_idx, flat)
        print(f"[verify-ram] layer={layer} mismatches={n_bad}/{self._ram * len(self._banks)}",
              flush=True)

    def _identify_source(self, layer: int, bank_idx: int, flat: torch.Tensor) -> None:
        """Debug: find where a corrupted slot's bytes came from (GPU slot, host row,
        or checkpoint row)."""
        flat_cpu = flat.cpu()
        head = flat_cpu[:16]
        found = []
        _host_layer, gpu_cache = self._banks[bank_idx]
        gpu_flat = gpu_cache.view(torch.uint8).reshape(gpu_cache.shape[0], -1)
        if gpu_flat.shape[1] == flat_cpu.numel():
            cand = torch.nonzero(
                (gpu_flat[:, :16].cpu() == head.unsqueeze(0)).all(dim=1)).flatten().tolist()
            for s in cand[:16]:
                if bool(torch.equal(gpu_flat[s].cpu(), flat_cpu)):
                    found.append(f"gpu_slot={s}(L{layer},B{bank_idx})")
        for e in range(self._ram):
            hflat = _host_layer[layer][e].contiguous().view(torch.uint8).reshape(-1)
            if hflat.numel() == flat_cpu.numel() and bool(torch.equal(hflat, flat_cpu)):
                found.append(f"host_row_e{e}(L{layer},B{bank_idx})")
        import itertools
        num_layers = len(self._banks[bank_idx][0])
        targets = sorted(set(itertools.product([layer], range(256)))
                         | set(itertools.product(range(num_layers), [10])))
        for (L, e) in targets:
            row_el, row_leading = None, None
            hrow = _host_layer[layer][0]
            row_el = hrow.element_size()
            row_leading = hrow.numel() // hrow.shape[0] if hrow.dim() > 1 else 1
            try:
                ref = self._ref_row(bank_idx, L, e, flat_cpu.numel(), row_el, row_leading)
            except Exception:
                continue
            if bool(torch.equal(ref, flat_cpu)):
                found.append(f"checkpoint_L{L}_e{e}")
        print(f"[identify] L{layer} B{bank_idx} n={flat_cpu.numel()} "
              f"source={found if found else 'UNKNOWN'}", flush=True)

    def verify_decode_mapping(self, cache, layer_id: int, topk_ids: torch.Tensor,
                              routed_experts: torch.Tensor | None = None) -> None:
        """Debug: after the LRU rewrite + fetch/copy, check that every slot the GEMM
        will read actually holds the expert the bookkeeping says it holds. Gated on
        FT_DISK_TIER_VERIFY; FT_DISK_TIER_VERIFY_STEPS caps steps (default 4);
        FT_DISK_TIER_VERIFY_ALL_LAYERS=1 checks every layer, else layer 0 callers only.

        When ``routed_experts`` (the local expert ids aligned with ``topk_ids``) is
        given, the check is route satisfaction: the bytes at each slot must equal the
        ROUTED expert's reference row, and ``id_of_slot`` is diagnostic only. Without
        it (legacy), the reference expert is taken from ``id_of_slot`` itself -- that
        only proves bookkeeping<->bytes consistency, which a wrong-but-consistent
        assignment passes."""
        cap = int(os.environ.get("FT_DISK_TIER_VERIFY_STEPS", "4"))
        steps_done = self._map_verify_steps.get(layer_id, 0)
        if steps_done >= cap:
            return
        self._map_verify_steps[layer_id] = steps_done + 1
        map_step = steps_done + 1
        banks = self._banks if os.environ.get(
            "FT_DISK_TIER_VERIFY_ALL_BANKS") else self._banks[:1]
        if routed_experts is not None:
            pairs = torch.stack(
                (topk_ids.reshape(-1).long(), routed_experts.reshape(-1).long()),
                dim=1).unique(dim=0)
            checks = [(int(s), int(e)) for s, e in pairs.tolist()]
        else:
            checks = [(int(s), None)
                      for s in torch.unique(topk_ids.reshape(-1)).tolist()]
        nbad = 0
        for s, routed in checks:
            flat_id = int(cache.id_of_slot[s].item())
            if routed is None and flat_id < 0:
                print(f"[verify-map] step={map_step} L{layer_id} slot={s} id_of_slot=-1",
                      flush=True)
                nbad += 1
                continue
            expert = routed if routed is not None else flat_id % cache.num_experts
            fetched = (routed is not None
                       and int(routed) in self._dbg_fetched.get(layer_id, set()))
            diag = (f" routed={routed} id_of_slot={flat_id} fetched={fetched}"
                    if routed is not None else "")
            for bank_idx, (_host_layer, gpu_cache) in enumerate(banks):
                slot_row = gpu_cache[s].contiguous()
                flat = slot_row.view(torch.uint8).reshape(-1)
                ref = self._ref_row(bank_idx, layer_id, expert, flat.numel(),
                                    slot_row.element_size(),
                                    slot_row.numel() // slot_row.shape[0]
                                    if slot_row.dim() > 1 else 1)
                if not bool(torch.equal(flat.cpu(), ref)):
                    nbad += 1
                    if nbad <= 8:
                        print(f"[verify-map] step={map_step} L{layer_id} slot={s} "
                              f"expert={expert} bank={bank_idx} MISMATCH{diag} "
                              f"slot_head={flat[:8].tolist()} ref_head={ref[:8].tolist()}",
                              flush=True)
        print(f"[verify-map] step={map_step} L{layer_id} slots={len(checks)} bad={nbad}",
              flush=True)

    def materialize_layer(self, cache, layer_id: int, expert_ids: torch.Tensor) -> None:
        """Disk-tier prefill: materialize the RAM-resident prefix into identity slots
        (the normal kernel restricted to K experts; the following ``copy_missing``
        streams it over PCIe), then fetch the routed disk-resident experts into
        THEIR identity slots. The identity mapping (position == expert id) is
        preserved, so the prefill GEMM is unchanged."""
        from freetoken.moe.offload_kernels import _materialize_layer_gpu

        _materialize_layer_gpu(cache, layer_id, materialize_count=self._ram,
                               pin_ids=self._pin_ids_dev[layer_id])
        self.fetch_routed(cache, layer_id, expert_ids)

    def fetch_routed(self, cache, layer_id: int,
                     expert_ids: torch.Tensor) -> torch.Tensor:
        """Fetch the ROUTED disk-resident experts of one layer into their identity
        slots (slot row == expert id) and return them as a device int32 tensor.

        Shared by the legacy prefill (``materialize_layer``) and the overlap
        prefill, whose ring already streamed the RAM prefix into a borrowed
        buffer and only needs the disk rows patched in. ``expert_ids`` is the
        layer's routing (GLOBAL ids under owner-EP; renumbered to local rows
        here). Bookkeeping (slot_for_id/id_of_slot/usage) matches the
        materialize kernel so the decode LRU sees the fetched experts.
        """
        # Prefill identity mapping owns ALL of slots [0, E) for this layer, but
        # only the pinned experts' slots are materialized, so the non-pinned
        # slots that still hold a previous layer's experts (previous prefill
        # layer or decode LRU) would keep their slot_for_id entries -- phantom
        # decode hits that read another layer's weights. Clear them first
        # (device-side, no sync). With the default prefix layout the non-pinned
        # slots are exactly [ram, E) (the old slice); a learned pin set scatters
        # them, so the clear goes through the inverse row map either way.
        idx = self._nonpin_dev[layer_id]
        seg = cache.id_of_slot.index_select(0, idx)
        valid = seg >= 0
        cache.slot_for_id.view(-1)[seg[valid].long()] = -1
        cache.id_of_slot[idx] = torch.where(valid, -1, seg)
        cache.usage[idx] = torch.where(valid, 0, cache.usage[idx])

        routed = self._routed_disk(layer_id, expert_ids)
        disk = routed
        if os.environ.get("FT_DISK_TIER_DEBUG") and layer_id < 3:
            print(f"[disk-tier dbg] layer={layer_id} routed={expert_ids.numel()} "
                  f"unique_disk={disk.numel()} disk={disk.tolist()[:12]}", flush=True)
        if disk.numel() == 0:
            return disk.to(dtype=torch.int32)
        # cache.step was already incremented by the kernel; assign the 0-d tensor
        # device-side (same dtype/device as usage) instead of .item()-ing it, which
        # would sync the stream once per layer on the prefill/decode path.
        futures = [
            self._pool.submit(self._fetch_expert, layer_id, int(e), int(e))
            for e in disk.tolist()
        ]
        for f in futures:
            f.result()
        self._sync_fetches()
        if os.environ.get("FT_DISK_TIER_VERIFY") and disk.numel() > 0 and (
                layer_id in (0, 20)
                or os.environ.get("FT_DISK_TIER_VERIFY_ALL_LAYERS")):
            limit = disk.numel() if layer_id == 0 else 6  # layer 0: ALL experts (race hunt)
            for e in disk.tolist()[:limit]:
                self._verify_slot(cache, layer_id, int(e), phase="prefill")
        # Same bookkeeping the materialize kernel writes, per fetched expert.
        flat = layer_id * cache.num_experts + disk
        cache.slot_for_id[layer_id, disk] = disk
        cache.id_of_slot[disk] = flat
        cache.usage[disk] = cache.step
        return disk.to(dtype=torch.int32)

    def _routed_disk(self, layer_id: int, expert_ids: torch.Tensor) -> torch.Tensor:
        """Unique disk-resident rows in this layer's routing (GLOBAL ids under
        owner-EP; renumbered to local rows, unowned experts dropped).

        "Disk-resident" goes through the pinned-row layout: with the default
        identity map ``_row_map[layer][e] == e`` this is the prefix test
        ``e >= _ram``; with a learned pin set it is whatever the pin file left
        out."""
        routed = expert_ids.reshape(-1)
        if self._g0 or self._local_num != self._index.num_experts:
            # Owner-local EP: routing arrives GLOBAL; renumber to local rows and
            # drop the experts this rank does not own.
            routed = routed - self._g0
            routed = routed[(routed >= 0) & (routed < self._local_num)]
        rows = self._row_map_dev[layer_id][routed.long()]
        return torch.unique(routed[rows >= self._ram])

    def fetch_routed_into(self, cache, layer_id: int,
                          expert_ids: torch.Tensor, buffer_id: int) -> int:
        """Overlap-prefill fetch: write the routed disk-resident experts directly
        into the borrowed buffer rows (position == expert id), bypassing the
        identity slots. The overlap ring borrows the slot cache's first
        ``depth * E`` rows as buffers, so the identity slots ARE buffer rows --
        fetching into them would scribble on a neighbouring buffer whose GEMM
        may still be reading it. The buffer's tail rows were never copied by
        the ring (prefix-only copy), so no ring-copy/pool-write race either.

        Slot bookkeeping is untouched: the buffer rows' map entries are owned
        by the overlap ring (``_invalidate_prefill_buffer``), and decode never
        sees a phantom hit. Returns the fetched-expert count.
        """
        disk = self._routed_disk(layer_id, expert_ids)
        if disk.numel() == 0:
            return 0
        device = self._banks[0][1].device
        if device.type == "cuda":
            # Order the pool threads' H2D writes (default stream) behind ALL
            # compute enqueued so far -- this layer's ring-copy wait
            # (wait_prefill_layer) and the previous occupants' GEMMs. The NVMe
            # preadv below still overlaps the in-flight compute; only the PCIe
            # writes wait.
            gate = torch.cuda.Event()
            gate.record()
            torch.cuda.default_stream(device).wait_event(gate)
        buffers = cache.prefill_bank_buffers
        futures = [
            self._pool.submit(self._fetch_expert, layer_id, int(e), -1,
                              buffers, buffer_id)
            for e in disk.tolist()
        ]
        for f in futures:
            f.result()
        self._sync_fetches()
        self._dbg_fetched[layer_id] = {int(e) for e in disk.tolist()}
        return disk.numel()

    # ------------------------------------------------------------- PILOT prefetch
    def _slab_bytes(self) -> int:
        """Pinned slab size for one prefetched expert: every non-scalar bank's
        merged reads land in one slab."""
        total = 0
        scalar_banks = getattr(self._index, "scalar_banks", ())
        for bank_idx in range(len(self._banks)):
            if bank_idx in scalar_banks:
                continue
            total += self._row_bytes[bank_idx] + 2 * _ALIGN
        return total

    def _new_slab(self) -> torch.Tensor:
        try:
            return torch.zeros(self._slab_bytes(), dtype=torch.uint8,
                               pin_memory=True)
        except (RuntimeError, TypeError):
            return torch.zeros(self._slab_bytes(), dtype=torch.uint8)

    def mark_turn_src(self, layer: int, expert: int, value: int) -> None:
        """Record how one (layer, expert) row was served, for this turn's HITS bitmap.

        Called from two very different contexts: the engine's decode step, which runs
        inside its inference mode, and the graph-doorbell service thread, which runs
        *outside* it. ``_turn_src`` is created under inference mode, and an in-place
        write to an inference tensor from outside that scope raises ``RuntimeError:
        Inplace update to inference tensor outside InferenceMode is not allowed`` --
        which killed the doorbell thread and hung the engine in a prefill spin on
        2026-10-03. Re-entering the scope here makes the write legal from either side.
        """
        src = self._turn_src
        if src is None:
            return
        with torch.inference_mode():
            src[layer, expert] = value

    def end_turn(self) -> dict:
        """Close the current telemetry turn: snapshot the HITS bitmap, persist
        one JSONL record when FT_DISK_TIER_TELEMETRY is set, and reset.

        Safe to call from the doorbell service thread: every tensor op runs inside an
        explicit inference scope (see :meth:`mark_turn_src`).
        """
        import numpy as np
        src = self._turn_src
        if src is None:
            return {}
        with torch.inference_mode():
            hit = src >= 0
            record = {
                "turn": self._turn,
                "served_ram": int((src == 0).sum()),
                "served_stash": int((src == 1).sum()),
                "served_disk": int((src == 2).sum()),
                "pf_hits": self._pf_hits,
                "pf_wasted": self._pf_wasted,
                "layers": {
                    str(l): np.packbits(hit[l].numpy()).tobytes().hex()
                    for l in range(src.shape[0]) if bool(hit[l].any())
                },
            }
            self._turn_src.fill_(-1)
        self._turn += 1
        if self._telemetry_path:
            with open(self._telemetry_path, "a") as f:
                f.write(json.dumps(record) + "\n")
        return record

    def prefetch_from_routing(self, layer_id: int, expert_ids: torch.Tensor) -> None:
        """PILOT trigger: called by the decode path at layer L with L's RAW routing
        (before ensure_experts rewrites it). Identity-map prediction -- colibri
        measured 62.3% top-8 overlap between adjacent layers on DSv4 -- so L's
        routed experts are the prefetch set for L+1..L+window."""
        if torch.cuda.is_current_stream_capturing():
            # The telemetry below is a D2H, and an unpinned one is illegal inside a capture. The
            # fetch itself is not lost: with graphs on, replay-time disk reads go through the
            # doorbell bridge (offload_cache.copy_missing -> graph_stage_fetch, served by the host
            # thread). What is lost is the predictor's heat data while graphs are on -- it still
            # accumulates from every non-captured step (prefill, and any eager decode).
            return
        # Telemetry first (runs even with the prefetch window disabled).
        flat = expert_ids.reshape(-1).to(torch.int64).cpu()
        self._route_hist[layer_id].scatter_add_(
            0, flat, torch.ones_like(flat, dtype=torch.float64))
        self._route_last[layer_id, flat] = self._route_clock
        if layer_id == 0:
            self._route_clock += 1
            self._hist_since_decay += 1
            if self._hist_since_decay >= 256:  # ~256 decode tokens per decay round
                self._route_hist *= 0.97   # per-round decay: stale heat dies off
                self._hist_since_decay = 0
        if self._prefetch_window <= 0:
            return
        ids = {int(e) for e in flat.tolist()}
        if self._g0 or self._local_num != self._index.num_experts:
            # Owner-local EP: routing arrives GLOBAL; keep only this rank's owned
            # experts and renumber into the local slot-cache namespace.
            lo, hi = self._g0, self._g0 + self._local_num
            ids = {e - lo for e in ids if lo <= e < hi}
        self._prefetch_local(layer_id, ids)

    def prefetch_from_local_ids(self, layer_id: int, ids, *,
                               allow_events: bool = False) -> None:
        """PILOT trigger for callers that already hold LOCAL (bank row) ids.

        The graph-doorbell request block carries the local row ids of the disk rows the
        captured graph is about to need (``moe/graph_fetch.py``), so the host service
        thread can predict layer L+1 with no extra device->host copy.

        ``allow_events=False`` is the default HERE because ``ev.synchronize()`` from the
        doorbell thread can queue behind the replayed graph whose spin kernel is waiting
        for that very thread (the deadlock ``graph_fetch.GraphFetchBridge._bounce``
        documents); a slab with a pending H2D is skipped instead of waited on.
        """
        if self._prefetch_window <= 0:
            return
        self._prefetch_local(layer_id, {int(e) for e in ids},
                             allow_events=allow_events)

    def _prefetch_local(self, layer_id: int, ids: set[int], *,
                       allow_events: bool = True) -> None:
        """Issue the identity-map prefetch for L+1..L+window given LOCAL ids."""
        for target in range(layer_id + 1,
                            min(layer_id + 1 + self._prefetch_window,
                                self._index.num_layers)):
            with self._stash_lock:
                for e in sorted(ids):
                    if int(self._row_map_cpu[target][e]) < self._ram:
                        continue  # RAM-resident: PCIe fallback is already fast
                    if e in self._resident_disk[target]:
                        self._pf_skipped_resident += 1
                        continue
                    key = (target, e)
                    if key in self._stash:
                        continue  # reservation: already in flight / stashed
                    slab_ent = self._slab_free.pop() if self._slab_free else None
                    if slab_ent is None:
                        if self._slabs_allocated >= self._slab_count:
                            continue  # slab pool exhausted; degrade gracefully
                        slab_ent = (self._new_slab(), None)
                        self._slabs_allocated += 1
                    slab, ev = slab_ent
                    if ev is not None:
                        if not allow_events:
                            self._slab_free.append(slab_ent)  # busy slab: skip it
                            continue
                        ev.synchronize()  # pending H2D out of this slab
                    entry = {"slab": slab, "future": None, "groups": None}
                    self._stash[key] = entry
                    self._pf_issued += 1
                    entry["future"] = self._prefetch_pool.submit(
                        self._prefetch_one, target, e, entry)

    def _prefetch_one(self, layer: int, expert: int, entry: dict) -> None:
        """Pool task: read one expert's merged groups into its slab. Pure host
        work -- no CUDA, no slot cache -- so it is safe to run behind the GEMM."""
        slab = entry["slab"]
        scalar_banks = getattr(self._index, "scalar_banks", ())
        groups = []
        slab_off = 0
        for bank_idx in range(len(self._banks)):
            if bank_idx in scalar_banks:
                continue
            for shard_idx, a0, a1, members in [
                    (g[0], g[1], g[2], g[3])
                    for g in self._group_runs(bank_idx, layer, expert)]:
                slen = a1 - a0
                mv = (ctypes.c_char * slen).from_address(
                    slab.data_ptr() + slab_off)
                fd, _direct = self._fd(shard_idx)
                os.preadv(fd, [mv], a0)
                self._preadv_calls += 1
                groups.append((slab_off, a0, bank_idx, members))
                slab_off += slen
        entry["groups"] = groups

    def _stash_to_slot(self, entry: dict, layer: int, expert: int,
                       slot: int) -> None:
        """Stash hit: copy the slab's bytes into the slot rows (pinned H2D instead
        of an NVMe read), then recycle the slab."""
        with torch.inference_mode():
            slab = entry["slab"]
            scalar_banks = getattr(self._index, "scalar_banks", ())
            for bank_idx, (_host_layer, gpu_cache) in enumerate(self._banks):
                row = gpu_cache[slot]
                if bank_idx in scalar_banks:
                    self._fill_scalar_row(bank_idx, layer, expert, row)
                    continue
            for slab_off, a0, bank_idx, members in entry["groups"]:
                row = self._banks[bank_idx][1][slot]
                for d0, d1, off, nbytes in members:
                    src = slab[slab_off + (off - a0):
                               slab_off + (off - a0) + nbytes]
                    dst = row[d0:d1]
                    dst.copy_(src.view(dst.dtype).view(dst.shape),
                              non_blocking=True)
        ev = None
        if self._banks[0][1].device.type == "cuda":
            # Arm: the next prefetch into this slab waits for these async H2D
            # copies (same pinned-reuse race as the staging ring).
            ev = torch.cuda.Event()
            ev.record()
        with self._stash_lock:
            self._slab_free.append((slab, ev))

    def stash_to_staging(self, entry: dict, layer: int, expert: int,
                         dst_rows) -> None:
        """Stash hit on the graph-doorbell path: slab -> pinned staging rows.

        The pinned->pinned analogue of :meth:`_stash_to_slot`. It makes NO CUDA calls
        and recycles the slab WITHOUT arming an event, which is what lets the doorbell
        host thread call it while a graph replay spins (``moe/graph_fetch.py``).
        """
        slab = entry["slab"]
        scalar_banks = getattr(self._index, "scalar_banks", ())
        for bank_idx in range(len(self._banks)):
            if bank_idx in scalar_banks:
                self._fill_scalar_row(bank_idx, layer, expert, dst_rows[bank_idx])
        for slab_off, a0, bank_idx, members in entry["groups"]:
            row = dst_rows[bank_idx]
            for d0, d1, off, nbytes in members:
                src = slab[slab_off + (off - a0): slab_off + (off - a0) + nbytes]
                dst = row[d0:d1]
                dst.copy_(src.view(dst.dtype).view(dst.shape))
        with self._stash_lock:
            self._slab_free.append((slab, None))  # CPU-only use: no event needed

    def prefetch_from_routing_cpu(self, layer_id: int, ids) -> None:
        """Test hook: prefetch with a plain iterable of ids (no GPU tensor)."""
        if self._prefetch_window <= 0:
            return
        self.prefetch_from_routing(layer_id, torch.tensor(sorted(ids),
                                                          dtype=torch.int32))

    def route_histogram(self) -> torch.Tensor:
        """Decayed (L, E) routing-mass snapshot for placement decisions."""
        return self._route_hist.clone()

    def save_histogram(self, path: str) -> None:
        payload = {"layers": self._index.num_layers,
                   "experts": self._index.num_experts,
                   "hist": self._route_hist.tolist()}
        with open(path, "w") as f:
            json.dump(payload, f)

    def load_histogram(self, path: str) -> None:
        with open(path) as f:
            payload = json.load(f)
        hist = torch.tensor(payload["hist"], dtype=torch.float64)
        if hist.shape == self._route_hist.shape:
            self._route_hist.copy_(hist)

    def fetch_pending(self, cache, layer_id: int) -> None:
        """Fetch this layer's disk-resident misses into their slots; shrink the miss
        list to the RAM-resident remainder for the existing PCIe copy path."""
        n = int(cache.num_indices.item())
        if n == 0:
            return
        src = cache.src_indices[:n].cpu()
        slots = cache.evict_slots[:n].cpu()
        # Keep the host slot-occupancy mirror exact: this list is EVERY miss->slot
        # assignment for the layer, so retiring the old owner of each slot plus
        # recording the new one tracks residency precisely.
        # RAM/disk classification goes through the pinned-row layout (identity
        # map without a pin file, so ``rows == src`` and this is the old prefix
        # test; a learned pin set scatters it).
        rm = self._row_map_cpu[layer_id]
        for i in range(n):
            s = int(slots[i])
            old = self._slot_owner.pop((layer_id, s), None)
            if old is not None and int(rm[old]) >= self._ram:
                self._resident_disk[layer_id].discard(old)
            e = int(src[i])
            self._slot_owner[(layer_id, s)] = e
            if int(rm[e]) >= self._ram:
                self._resident_disk[layer_id].add(e)
        rows = rm[src.long()]
        ram_mask = rows < self._ram
        # RAM-classified misses are served from the host bank over PCIe.
        self._turn_src[layer_id][src[ram_mask].long()] = 0
        disk = [i for i in range(n) if int(rows[i]) >= self._ram]
        if os.environ.get("FT_DISK_TIER_VERIFY") and (
                layer_id == 0 or os.environ.get("FT_DISK_TIER_VERIFY_ALL_LAYERS")):
            print(f"[fetch-pend] layer={layer_id} n={n} ndisk={len(disk)} "
                  f"src_head={src[:4].tolist()}", flush=True)
        if not disk:
            # Nothing to fetch -- but with a learned pin set the PCIe copy below
            # resolves src_indices against HOST BANK ROWS, so the all-RAM plan
            # must still be translated from local expert ids to bank rows.
            # (Identity layout: rows == src, and this is the old no-op return.)
            if self._remapped and n:
                cache.src_indices[:n].copy_(rows.to(cache.src_indices.dtype))
            return
        futures = []
        for i in disk:
            expert, slot = int(src[i]), int(slots[i])
            entry = None
            if self._prefetch_window > 0:
                with self._stash_lock:
                    entry = self._stash.pop((layer_id, expert), None)
            if entry is not None:
                entry["future"].result()  # slab read done? (usually long done)
                futures.append(self._pool.submit(
                    self._stash_to_slot, entry, layer_id, expert, slot))
                self._pf_hits += 1
                self._turn_src[layer_id, expert] = 1
            else:
                futures.append(self._pool.submit(
                    self._fetch_expert, layer_id, expert, slot))
                self._turn_src[layer_id, expert] = 2
        for f in futures:
            f.result()
        self._sync_fetches()
        if os.environ.get("FT_DISK_TIER_VERIFY"):
            self._dbg_fetched[layer_id] = {int(src[i]) for i in disk}
        if (os.environ.get("FT_DISK_TIER_VERIFY")
                and (layer_id == 0 or os.environ.get("FT_DISK_TIER_VERIFY_ALL_LAYERS"))
                and self._decode_verify_steps.get(layer_id, 0) < 3):
            dec_step = self._decode_verify_steps.get(layer_id, 0) + 1
            self._decode_verify_steps[layer_id] = dec_step
            for i in disk[:8]:
                print(f"[verify-decode] step={dec_step} L{layer_id} "
                      f"expert={int(src[i])} slot={int(slots[i])} ndisk={len(disk)}", flush=True)
                self._verify_slot(cache, layer_id, int(src[i]), int(slots[i]),
                                  phase="decode")
        # Wasted prefetches: stashed for THIS layer but never routed. Recycle
        # their slabs once the read finishes so the pool stays available.
        if self._prefetch_window > 0:
            with self._stash_lock:
                stale = [k for k in self._stash if k[0] <= layer_id]
                for k in stale:
                    entry = self._stash.pop(k)
                    self._pf_wasted += 1
                    fut = entry["future"]

                    def _recycle(fut=fut, slab=entry["slab"]):
                        fut.result()  # prefetch read done; no H2D was enqueued
                        with self._stash_lock:
                            self._slab_free.append((slab, None))
                    self._prefetch_pool.submit(_recycle)
        disk_set = set(disk)
        ram = [i for i in range(n) if i not in disk_set]
        if ram:
            sel = torch.tensor(ram, dtype=torch.long)
            # The PCIe copy path resolves src_indices against the HOST BANK rows;
            # with a remapped pin set the bank row is not the local expert id.
            cache.src_indices[:len(ram)].copy_(rows[sel].to(cache.src_indices.dtype))
            cache.evict_slots[:len(ram)].copy_(slots[sel].to(cache.evict_slots.dtype))
        cache.num_indices.fill_(len(ram))

    def refresh(self, cache) -> None:
        """Rebind the slot-cache references after a runtime cache rebuild."""
        self._banks = list(cache.banks)
        self._cache = cache
        # Slots were reallocated: the occupancy mirror and any stashed prefetch
        # are stale. Dropping stash entries leaks their slabs back via _recycle
        # only if they were read; simplest correct move is to drain everything.
        self._slot_owner.clear()
        for s in self._resident_disk:
            s.clear()
        if self._prefetch_window > 0:
            with self._stash_lock:
                for key, entry in self._stash.items():
                    fut = entry["future"]

                    def _recycle(fut=fut, slab=entry["slab"]):
                        fut.result()
                        with self._stash_lock:
                            self._slab_free.append((slab, None))
                    self._prefetch_pool.submit(_recycle)
                self._stash.clear()

    def record_doorbell(self, rows: int, host_us: int) -> None:
        """One served doorbell request (graph_fetch hot path; GIL-serialized
        caller). ``host_us`` is the full request latency on the service thread
        (all rows staged), so it includes preadv waits and memmoves."""
        self._db_requests += 1
        self._db_rows += rows
        self._db_bytes += rows * self._db_row_bytes
        self._db_host_us += host_us

    def stats(self) -> dict:
        return {
            "experts_fetched": self._fetches,
            "bytes_fetched": self._fetch_bytes,
            "preadv_calls": self._preadv_calls,
            "scalar_preload_reads": self._scalar_preload_reads,
            "prefetch_issued": self._pf_issued,
            "prefetch_hits": self._pf_hits,
            "prefetch_wasted": self._pf_wasted,
            "prefetch_skipped_resident": self._pf_skipped_resident,
            "route_hist_total": float(self._route_hist.sum()),
            "pin_remapped": self._remapped,
            "doorbell_requests": self._db_requests,
            "doorbell_rows": self._db_rows,
            "doorbell_bytes": self._db_bytes,
            "doorbell_host_ms": self._db_host_us / 1000.0,
        }
