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
  directly and would read released pages;
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
    served from disk by :class:`DiskTier`."""

    ram_experts: int


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



def autopin_pinned_count(route_hist: "torch.Tensor", budget_bytes: int,
                         ram_row_bytes: int) -> int:
    """P1 AUTOPIN (colibri tier.h): pinned share = 0.5 x budget x
    min(1, observations/200000), then the largest contiguous expert prefix
    [0, K) whose measured routing mass fits that share.

    route_hist: (L, E) float routing-mass histogram (decayed counts).
    The prefix constraint matches the host-bank layout (RAM rows are a
    contiguous prefix); arbitrary-set pinning needs a loader row remap and
    is deliberately out of scope here.
    """
    import torch as _t
    E = route_hist.shape[-1]
    total_obs = float(route_hist.sum())
    if total_obs <= 0 or ram_row_bytes <= 0:
        return 0
    pinned_budget = 0.5 * budget_bytes * min(1.0, total_obs / 200000.0)
    k_max = min(E, int(pinned_budget // ram_row_bytes))
    if k_max <= 0:
        return 0
    # Per-expert mass averaged over layers; prefix coverage of [0, K).
    mass = route_hist.sum(dim=0) / route_hist.shape[0]
    coverage = _t.cumsum(mass, dim=0) / mass.sum().clamp(min=1e-9)
    # Pick the largest K <= k_max that still improves coverage meaningfully;
    # beyond the mass knee extra pins buy nothing.
    k = k_max
    while k > 1 and float(coverage[k - 1]) - float(coverage[k - 2]) < 1e-4:
        k -= 1
    return k


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
        # (bank_layer, expert, proj, kind) -> (tensor_name, shard)
        loc: dict[tuple[int, int, str, str], tuple[str, str]] = {}
        for name, shard in weight_map.items():
            m = spec.key_pattern.match(name)
            if m is None:
                continue
            bank_layer = spec.layer_to_bank(int(m.group("layer")), config)
            if bank_layer is None:
                continue
            loc[(bank_layer, int(m.group("expert")), m.group("proj"), m.group("kind"))] = (
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


class DiskTier:
    """Runtime fetcher: disk-resident slot-cache misses -> staging -> GPU slot."""

    def __init__(self, index: Nvfp4DiskIndex, cache, ram_experts: int, workers: int = 8,
                 ownership=None) -> None:
        self._index = index
        # Owner-local EP: the slot cache (and therefore every miss/fetch/slot
        # identifier below) lives in the LOCAL expert namespace
        # [0, ownership.local_num_experts), while the disk index spans the GLOBAL
        # checkpoint rows [0, index.num_experts). ``_g0`` is the local->global
        # offset applied at every index access; ``_ram`` is the RAM prefix
        # expressed in the LOCAL namespace (global ``ram_experts`` clamped onto
        # this rank's owned range). With ownership=None both collapse to the
        # identity (global) mapping.
        self._g0 = ownership.global_start if ownership is not None else 0
        self._local_num = (ownership.local_num_experts if ownership is not None
                           else index.num_experts)
        self._ram = min(self._local_num, max(0, ram_experts - self._g0))
        self._banks = list(cache.banks)  # [(per_layer_host, gpu_cache)] in schema order
        self._row_bytes = [
            b[0][0][0].numel() * b[0][0][0].element_size() for b in self._banks
        ]  # full expert-row bytes per bank (staging must hold the biggest one)
        # Per-bank destination row slices (gate|up split at the row midpoint).
        self._dst_slices: list[list[tuple[int, int]]] = []
        for bank_idx, (host_layer, _gpu) in enumerate(self._banks):
            row = host_layer[0][0]
            if len(_NVP4_BANK_SEGS[bank_idx]) == 2:
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
        self._staging_size = ((max_row + _ALIGN - 1) // _ALIGN + 2) * _ALIGN
        self._staging = threading.local()
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="disk-tier")
        self._fd_lock = threading.Lock()
        self._fds: dict[int, tuple[int, bool]] = {}
        self._fetches = 0
        self._fetch_bytes = 0
        self._preadv_calls = 0
        self._scalar_preload_reads = 0
        self._scalars: dict | None = None
        self._scalars_lock = threading.Lock()
        self._decode_verify_steps = 0
        self._map_verify_steps = 0
        self._cache = cache
        # ---- PILOT prefetch (P0-4 port of colibri's router-guided cross-layer
        # prefetch): at layer L the decode path hands us L's raw routing; we read
        # the predicted experts for L+1 into pinned host SLABS. A stash entry never
        # touches a GPU slot, so it cannot evict the current demand set (P0-1) and
        # its (layer, expert) key suppresses duplicate reads vs the on-demand path
        # (P0-2 reservation). Layer L+1's fetch_pending consumes stash hits with a
        # fast pinned->slot H2D instead of an NVMe round trip.
        # Default ON (user decision 2026-09-28): the e2e consistency gate proves
        # token-identical output with window=1, and stash entries can never evict
        # the demand set. Set FT_DISK_TIER_PREFETCH=0 to disable explicitly.
        self._prefetch_window = int(os.environ.get("FT_DISK_TIER_PREFETCH", "1"))
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
        """Read every expert's scalar-bank (weight_scale_2) values ONCE, merging
        exactly-adjacent file runs to keep startup syscalls low. One fp32 scalar
        per segment; blob layout per (layer, bank): expert-major, each expert's
        segments concatenated in index order."""
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
                    blob = bytearray(stride * self._index.num_experts)
                    flat = []  # (shard_idx, file_off, nbytes, blob_pos)
                    for e in range(self._index.num_experts):
                        pos = e * stride
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
        per-expert fp32 scalar blob -- no disk read, no staging."""
        expert = expert + self._g0  # local (slot-cache) id -> global checkpoint row
        segs = self._index.row_segments(bank_idx, layer, expert)
        stride = sum(nb for _, _, nb in segs)
        blob = self._scalar_blob(layer, bank_idx)
        base = expert * stride
        for k, (d0, d1) in enumerate(self._dst_slices[bank_idx]):
            val = struct.unpack_from("<f", blob, base + 4 * k)[0]
            row[d0:d1].fill_(val)

    def _group_runs(self, bank_idx: int, layer: int, expert: int):
        """Merged preadv groups for one non-scalar bank: [(shard, a0, a1,
        [(d0, d1, off, nbytes)])]. Sorts by file position and merges EXACTLY
        adjacent segments into one read (P0-3)."""
        expert = expert + self._g0  # local (slot-cache) id -> global checkpoint row
        segs = self._index.row_segments(bank_idx, layer, expert)
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

    def _fetch_expert(self, layer: int, expert: int, slot: int) -> None:
        # The server runs under inference_mode; the fetch pool threads do not,
        # so the H2D writes into the (inference) slot cache need their own scope.
        with torch.inference_mode():
            self._fetch_expert_inner(layer, expert, slot)

    def _fetch_expert_inner(self, layer: int, expert: int, slot: int) -> None:
        ring = self._staging_ring()
        ri = getattr(self._staging, "ri", 0)
        scalar_banks = getattr(self._index, "scalar_banks", ())
        for bank_idx, (_host_layer, gpu_cache) in enumerate(self._banks):
            row = gpu_cache[slot]
            if bank_idx in scalar_banks:
                self._fill_scalar_row(bank_idx, layer, expert, row)
                continue
            groups = self._group_runs(bank_idx, layer, expert)
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
                if ev is not None:
                    # Arm: the next reuse of this buffer waits for this copy.
                    ev.record()
        self._staging.ri = ri
        self._fetches += 1
        self._fetch_bytes += sum(self._row_bytes)

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
        """Reference row bytes for (bank, layer, expert) straight from the checkpoint."""
        expert = expert + self._g0  # local (slot-cache) id -> global checkpoint row
        ref = torch.zeros(row_bytes, dtype=torch.uint8)
        segs = self._index.row_segments(bank_idx, layer, expert)
        for (d0, d1), (shard_idx, off, nbytes) in zip(self._dst_slices[bank_idx], segs):
            fd, direct = self._fd(shard_idx)
            a0 = off if not direct else (off & ~(_ALIGN - 1))
            slen = nbytes if not direct else (off + nbytes - a0 + _ALIGN - 1) & ~(_ALIGN - 1)
            buf = os.pread(fd, slen, a0)
            row_off = off - a0
            seg = buf[row_off:row_off + nbytes]
            if bank_idx in (2, 5):
                import struct as _st
                import numpy as _np
                f16 = _np.float16(_st.unpack("<f", seg[:4])[0]).tobytes()
                for r in range(d0, d1):
                    ref[r * row_el:(r + 1) * row_el] = torch.frombuffer(f16, dtype=torch.uint8)
            else:
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
        expert = min(10, self._ram - 1)  # a RAM-resident expert
        print(f"[verify-ram] layer={layer} expert={expert} (RAM prefix)", flush=True)
        self._verify_slot(cache, layer, expert)
        # Check ALL RAM experts: slot vs host row (host correctness established
        # separately). Count mismatches; identify the source of the first one.
        n_bad = 0
        identified = False
        for e in range(self._ram):
            for bank_idx, (host_layer, gpu_cache) in enumerate(self._banks):
                slot_row = gpu_cache[e].contiguous()
                flat = slot_row.view(torch.uint8).reshape(-1)
                hflat = host_layer[layer][e].contiguous().view(torch.uint8).reshape(-1)
                if flat.numel() != hflat.numel() or not bool(torch.equal(flat.cpu(), hflat)):
                    n_bad += 1
                    if n_bad <= 12:
                        print(f"[verify-ram] MISMATCH e={e} bank={bank_idx} "
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

    def verify_decode_mapping(self, cache, layer_id: int, topk_ids: torch.Tensor) -> None:
        """Debug: after the LRU rewrite + fetch/copy, check that every slot the GEMM
        will read actually holds the expert the bookkeeping says it holds. Gated on
        FT_DISK_TIER_VERIFY; first 4 decode steps only."""
        if self._map_verify_steps >= 4:
            return
        self._map_verify_steps += 1
        slots = torch.unique(topk_ids.reshape(-1))
        nbad = 0
        for s in slots.tolist():
            s = int(s)
            flat_id = int(cache.id_of_slot[s].item())
            if flat_id < 0:
                print(f"[verify-map] step={self._map_verify_steps} slot={s} id_of_slot=-1",
                      flush=True)
                nbad += 1
                continue
            expert = flat_id % cache.num_experts
            for bank_idx, (_host_layer, gpu_cache) in enumerate(self._banks):
                slot_row = gpu_cache[s].contiguous()
                flat = slot_row.view(torch.uint8).reshape(-1)
                ref = self._ref_row(bank_idx, layer_id, expert, flat.numel(),
                                    slot_row.element_size(),
                                    slot_row.numel() // slot_row.shape[0]
                                    if slot_row.dim() > 1 else 1)
                if not bool(torch.equal(flat.cpu(), ref)):
                    nbad += 1
                    if nbad <= 8:
                        print(f"[verify-map] step={self._map_verify_steps} slot={s} "
                              f"expert={expert} bank={bank_idx} MISMATCH "
                              f"slot_head={flat[:8].tolist()} ref_head={ref[:8].tolist()}",
                              flush=True)
        print(f"[verify-map] step={self._map_verify_steps} slots={slots.numel()} bad={nbad}",
              flush=True)

    def materialize_layer(self, cache, layer_id: int, expert_ids: torch.Tensor) -> None:
        """Disk-tier prefill: materialize the RAM-resident prefix into identity slots
        (the normal kernel restricted to K experts; the following ``copy_missing``
        streams it over PCIe), then fetch the routed disk-resident experts into
        THEIR identity slots. The identity mapping (position == expert id) is
        preserved, so the prefill GEMM is unchanged."""
        from freetoken.moe.offload_kernels import _materialize_layer_gpu

        # Prefill identity mapping owns ALL of slots [0, E) for this layer, but
        # the kernel only scans slots < materialize_count, so the disk slots
        # [ram, E) that still hold a previous layer's experts (previous prefill
        # layer or decode LRU) would keep their slot_for_id entries -- phantom
        # decode hits that read another layer's weights. Clear them first
        # (device-side, no sync).
        seg = cache.id_of_slot[self._ram:cache.num_experts]
        valid = seg >= 0
        cache.slot_for_id.view(-1)[seg[valid].long()] = -1
        seg[valid] = -1
        cache.usage[self._ram:cache.num_experts][valid] = 0

        _materialize_layer_gpu(cache, layer_id, materialize_count=self._ram)
        routed = expert_ids.reshape(-1)
        if self._g0 or self._local_num != self._index.num_experts:
            # Owner-local EP: routing arrives GLOBAL; renumber to local rows and
            # drop the experts this rank does not own.
            routed = routed - self._g0
            routed = routed[(routed >= 0) & (routed < self._local_num)]
        disk = torch.unique(routed[routed >= self._ram])
        if os.environ.get("FT_DISK_TIER_DEBUG") and layer_id < 3:
            print(f"[disk-tier dbg] layer={layer_id} routed={routed.numel()} "
                  f"unique_disk={disk.numel()} disk={disk.tolist()[:12]}", flush=True)
        if disk.numel() == 0:
            return
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
        if os.environ.get("FT_DISK_TIER_VERIFY") and layer_id in (0, 20) and disk.numel() > 0:
            limit = disk.numel() if layer_id == 0 else 6  # layer 0: ALL experts (race hunt)
            for e in disk.tolist()[:limit]:
                self._verify_slot(cache, layer_id, int(e), phase="prefill")
        # Same bookkeeping the materialize kernel writes, per fetched expert.
        flat = layer_id * cache.num_experts + disk
        cache.slot_for_id[layer_id, disk] = disk
        cache.id_of_slot[disk] = flat
        cache.usage[disk] = cache.step

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

    def end_turn(self) -> dict:
        """Close the current telemetry turn: snapshot the HITS bitmap, persist
        one JSONL record when FT_DISK_TIER_TELEMETRY is set, and reset."""
        import numpy as np
        src = self._turn_src
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
        self._turn += 1
        self._turn_src.fill_(-1)
        if self._telemetry_path:
            with open(self._telemetry_path, "a") as f:
                f.write(json.dumps(record) + "\n")
        return record

    def prefetch_from_routing(self, layer_id: int, expert_ids: torch.Tensor) -> None:
        """PILOT trigger: called by the decode path at layer L with L's RAW routing
        (before ensure_experts rewrites it). Identity-map prediction -- colibri
        measured 62.3% top-8 overlap between adjacent layers on DSv4 -- so L's
        routed experts are the prefetch set for L+1..L+window."""
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
        for target in range(layer_id + 1,
                            min(layer_id + 1 + self._prefetch_window,
                                self._index.num_layers)):
            with self._stash_lock:
                for e in sorted(ids):
                    if e < self._ram:
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

    def prefetch_from_routing_cpu(self, layer_id: int, ids) -> None:
        """Test hook: prefetch with a plain iterable of ids (no GPU tensor)."""
        if self._prefetch_window <= 0:
            return
        self.prefetch_from_routing(layer_id, torch.tensor(sorted(ids),
                                                          dtype=torch.int32))

    def route_histogram(self) -> torch.Tensor:
        """Decayed (L, E) routing-mass snapshot for placement decisions."""
        return self._route_hist.clone()

    def autopin_advice(self, budget_bytes: int) -> int:
        """Recommended contiguous RAM prefix K from measured routing mass."""
        row = self._row_bytes[0] + 2 * _ALIGN if len(self._row_bytes) else 0
        ram_row = sum(self._row_bytes[b] for b in range(len(self._banks))
                      if b not in getattr(self._index, "scalar_banks", ()))
        return autopin_pinned_count(self._route_hist, budget_bytes,
                                    max(ram_row, row, 1))

    def pin_advice(self, layer_id: int) -> tuple[int, int, int] | None:
        """LFRU admission (colibri tier.h): which currently pinned expert should
        yield its RAM row to which hotter non-resident one. None = hysteresis
        says stay. Decision only -- actuation needs the loader row remap."""
        from freetoken.moe.tier_admission import heat_u32, pick_lfru

        return pick_lfru(heat_u32(self._route_hist[layer_id]),
                         self._route_last[layer_id], self._route_clock,
                         list(range(self._ram)))

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
        for i in range(n):
            s = int(slots[i])
            old = self._slot_owner.pop((layer_id, s), None)
            if old is not None and old >= self._ram:
                self._resident_disk[layer_id].discard(old)
            e = int(src[i])
            self._slot_owner[(layer_id, s)] = e
            if e >= self._ram:
                self._resident_disk[layer_id].add(e)
        # RAM-classified misses are served from the host bank over PCIe.
        self._turn_src[layer_id][src[src < self._ram].long()] = 0
        disk = [i for i in range(n) if int(src[i]) >= self._ram]
        if os.environ.get("FT_DISK_TIER_VERIFY") and layer_id == 0:
            print(f"[fetch-pend] layer=0 n={n} ndisk={len(disk)} "
                  f"src_head={src[:4].tolist()}", flush=True)
        if not disk:
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
        if (os.environ.get("FT_DISK_TIER_VERIFY") and layer_id == 0
                and self._decode_verify_steps < 3):
            self._decode_verify_steps += 1
            for i in disk[:8]:
                print(f"[verify-decode] step={self._decode_verify_steps} "
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
            cache.src_indices[:len(ram)].copy_(src[sel].to(cache.src_indices.dtype))
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
        }
