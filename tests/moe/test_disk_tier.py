"""CPU tests for the NVMe disk tier (moe/disk_tier.py).

A synthetic NVFP4 MoE checkpoint (2 layers, 4 experts) is written as safetensors
shards with deterministic per-tensor content; the tests verify that
:class:`Nvfp4DiskIndex` resolves the right byte ranges and that
:class:`DiskTier` places the right bytes into slot-cache rows and rewrites the
miss list. No CUDA needed -- the "GPU" banks here are CPU tensors and the
staging pin is stubbed.
"""

import json
import re
import struct
import threading
import types

import pytest
import torch

from freetoken.moe.disk_tier import DiskTier, Nvfp4DiskIndex
from freetoken.moe.host_banks import HostBank
from freetoken.models.nvfp4_banks import Nvfp4ExpertSourceSpec

H, I, E, L = 16, 32, 4, 2
SHARDS = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")

SPEC = Nvfp4ExpertSourceSpec(
    key_pattern=re.compile(
        r"^model\.language_model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
        r"(?P<proj>gate_proj|up_proj|down_proj)\."
        r"(?P<kind>weight|weight_scale|weight_scale_2)$"
    ),
    proj_to_role={"gate_proj": "gate", "up_proj": "up", "down_proj": "down"},
    layer_to_bank=lambda layer, config: layer,
    desc="disk-tier test",
)

# (proj, kind, shape, dtype) -- the native NVFP4 per-expert tensor layout.
TENSOR_SPECS = (
    ("gate_proj", "weight", (I, H // 2), torch.uint8),
    ("up_proj", "weight", (I, H // 2), torch.uint8),
    ("down_proj", "weight", (H, I // 2), torch.uint8),
    ("gate_proj", "weight_scale", (I, H // 16), torch.uint8),
    ("up_proj", "weight_scale", (I, H // 16), torch.uint8),
    ("down_proj", "weight_scale", (H, I // 16), torch.uint8),
    # weight_scale_2 is a per-expert fp32 SCALAR in the real checkpoints; the
    # bank row is its fp16 value broadcast across the row.
    ("gate_proj", "weight_scale_2", (), torch.float32),
    ("up_proj", "weight_scale_2", (), torch.float32),
    ("down_proj", "weight_scale_2", (), torch.float32),
)

BANK_SHAPES = (
    (2 * I, H // 2),  # gate_up_packed
    (2 * I, H // 16),  # gate_up_scale
    (2 * I,),  # gate_up_global
    (H, I // 2),  # down_packed
    (H, I // 16),  # down_scale
    (H,),  # down_global
)
BANK_DTYPES = (torch.uint8, torch.uint8, torch.float16, torch.uint8, torch.uint8, torch.float16)


def _name(layer, expert, proj, kind):
    return f"model.language_model.layers.{layer}.mlp.experts.{expert}.{proj}.{kind}"


def _tensor_for(layer, expert, proj, kind):
    """Deterministic content: a base offset per (layer, expert, proj, kind) so any
    misplacement is visible."""
    proj_i = ("gate_proj", "up_proj", "down_proj").index(proj)
    kind_i = ("weight", "weight_scale", "weight_scale_2").index(kind)
    base = layer * 100000 + expert * 1000 + proj_i * 100 + kind_i * 10
    for p, k, shape, dtype in TENSOR_SPECS:
        if p == proj and k == kind:
            if shape == ():  # per-expert fp32 scalar (kept in fp16 range)
                return torch.tensor(float(base % 50000), dtype=torch.float32)
            n = int(torch.tensor(shape).prod())
            if dtype == torch.uint8:
                return torch.arange(n, dtype=torch.uint8).add_(base % 251).view(shape)
            return (torch.arange(n, dtype=torch.float32) + base).to(dtype).view(shape)
    raise AssertionError((proj, kind))


@pytest.fixture()
def checkpoint(tmp_path):
    import safetensors.torch

    by_shard = {s: {} for s in SHARDS}
    weight_map = {}
    for layer in range(L):
        for expert in range(E):
            for proj, kind, _shape, _dtype in TENSOR_SPECS:
                shard = SHARDS[(layer * E + expert) % 2]
                name = _name(layer, expert, proj, kind)
                by_shard[shard][name] = _tensor_for(layer, expert, proj, kind)
                weight_map[name] = shard
    for shard, tensors in by_shard.items():
        safetensors.torch.save_file(tensors, str(tmp_path / shard), metadata={"format": "pt"})
    with open(tmp_path / "model.safetensors.index.json", "w", encoding="utf-8") as f:
        json.dump({"weight_map": weight_map, "metadata": None}, f)
    config = types.SimpleNamespace(num_experts=E, hidden_size=H, moe_intermediate_size=I,
                                   num_layers=L, first_k_dense_replace=0)
    return tmp_path, config


def _index(checkpoint):
    path, config = checkpoint
    return Nvfp4DiskIndex(str(path), config, SPEC)


def test_index_segments_match_file_bytes(checkpoint):
    import safetensors

    path, config = checkpoint
    index = _index(checkpoint)
    assert len(index.shard_paths) == 2
    # Every (bank, layer, expert) row segment must point at the exact tensor bytes.
    for bank_idx in range(6):
        for layer in range(L):
            for expert in range(E):
                segs = index.row_segments(bank_idx, layer, expert)
                expected = {
                    0: [("gate_proj", "weight"), ("up_proj", "weight")],
                    1: [("gate_proj", "weight_scale"), ("up_proj", "weight_scale")],
                    2: [("gate_proj", "weight_scale_2"), ("up_proj", "weight_scale_2")],
                    3: [("down_proj", "weight")],
                    4: [("down_proj", "weight_scale")],
                    5: [("down_proj", "weight_scale_2")],
                }[bank_idx]
                assert len(segs) == len(expected)
                for (shard_idx, off, nbytes), (proj, kind) in zip(segs, expected):
                    shard_path = index.shard_paths[shard_idx]
                    with open(shard_path, "rb") as f:
                        (hlen,) = struct.unpack("<Q", f.read(8))
                        meta = json.loads(f.read(hlen))
                        tstart, tend = meta[_name(layer, expert, proj, kind)]["data_offsets"]
                        base = 8 + hlen  # data_offsets are data-section-relative
                        assert (off, off + nbytes) == (tstart + base, tend + base), (
                            bank_idx, layer, expert, proj, kind)
                    with safetensors.safe_open(shard_path, framework="pt", device="cpu") as sf:
                        tensor = sf.get_tensor(_name(layer, expert, proj, kind))
                    assert nbytes == tensor.numel() * tensor.element_size()


def _fake_cache(num_experts=E):
    """CPU stand-in for OffloadMoeCache: banks in schema order + miss-list tensors."""
    banks = [
        (
            [torch.zeros(num_experts, *shape, dtype=dtype) for _ in range(L)],
            torch.full((8, *shape), 0xFF, dtype=dtype),
        )
        for shape, dtype in zip(BANK_SHAPES, BANK_DTYPES)
    ]
    cache = type("FakeCache", (), {})()
    cache.banks = banks
    cache.num_experts = num_experts
    cache.num_layers = L
    cache.num_indices = torch.tensor([0], dtype=torch.int64)
    cache.src_indices = torch.zeros(64, dtype=torch.int32)
    cache.evict_slots = torch.zeros(64, dtype=torch.int32)
    return cache


def _tier(checkpoint, cache, ram_experts=2, ownership=None, pin_rows=None):
    index = _index(checkpoint)
    tier = DiskTier(index, cache, ram_experts=ram_experts, workers=2,
                    ownership=ownership, pin_rows=pin_rows)
    # Staging must hold the largest bank row's O_DIRECT super-block (a size
    # regression here overflows the buffer -> EFAULT/segfault at fetch time).
    max_row = max(
        b[0][0][0].numel() * b[0][0][0].element_size() for b in cache.banks)
    assert tier._staging_size >= max_row + 2 * 4096, (tier._staging_size, max_row)
    # Stub the pinned staging (HostBank.pin needs CUDA) with the same per-thread
    # ring semantics as production (threading.local, no CUDA events on CPU).
    local = threading.local()

    def _staging_ring():
        ring = getattr(local, "ring", None)
        if ring is None:
            ring = [[HostBank((tier._staging_size,), torch.uint8), None]
                    for _ in range(tier._STAGING_RING)]
            local.ring = ring
        return ring

    tier._staging_ring = _staging_ring
    return tier


def _expected_rows(layer, expert):
    """The 6 bank rows for one expert, as flat uint8, in schema order."""
    rows = []
    for bank_idx, (shape, dtype) in enumerate(zip(BANK_SHAPES, BANK_DTYPES)):
        if bank_idx == 2:
            gate = _tensor_for(layer, expert, "gate_proj", "weight_scale_2").to(torch.float16)
            up = _tensor_for(layer, expert, "up_proj", "weight_scale_2").to(torch.float16)
            row = torch.cat([gate.expand(I), up.expand(I)]).view(shape)
        elif bank_idx == 5:
            row = (_tensor_for(layer, expert, "down_proj", "weight_scale_2")
                   .to(torch.float16).expand(H).view(shape))
        elif bank_idx < 3:
            kind = ("weight", "weight_scale")[bank_idx]
            gate = _tensor_for(layer, expert, "gate_proj", kind)
            up = _tensor_for(layer, expert, "up_proj", kind)
            row = torch.cat([gate.reshape(-1), up.reshape(-1)]).view(shape)
        else:
            row = _tensor_for(layer, expert, "down_proj", ("weight", "weight_scale")[bank_idx - 3])
        rows.append(row.contiguous().view(torch.uint8).reshape(-1))
    return rows


def test_fetch_expert_places_all_banks(checkpoint):
    cache = _fake_cache()
    tier = _tier(checkpoint, cache)
    for layer in range(L):
        for expert in range(E):
            slot = (layer * E + expert) % 8
            tier._fetch_expert(layer, expert, slot)
            expected = _expected_rows(layer, expert)
            for bank_idx, (host_layer, gpu_cache) in enumerate(cache.banks):
                got = gpu_cache[slot].contiguous().view(torch.uint8).reshape(-1)
                assert torch.equal(got, expected[bank_idx]), (bank_idx, layer, expert)
    stats = tier.stats()
    assert stats["experts_fetched"] == L * E


def test_fetch_pending_filters_and_rewrites(checkpoint):
    cache = _fake_cache()
    tier = _tier(checkpoint, cache, ram_experts=2)  # experts 0,1 RAM; 2,3 disk
    layer = 1
    # Miss list: expert 0 (RAM), 2 (disk), 3 (disk) -> slots 5, 6, 7.
    cache.src_indices[:3] = torch.tensor([0, 2, 3], dtype=torch.int32)
    cache.evict_slots[:3] = torch.tensor([5, 6, 7], dtype=torch.int32)
    cache.num_indices.fill_(3)

    tier.fetch_pending(cache, layer)

    # Disk misses fetched into their slots...
    expected = _expected_rows(layer, 2)
    for bank_idx, (_host, gpu_cache) in enumerate(cache.banks):
        assert torch.equal(
            gpu_cache[6].contiguous().view(torch.uint8).reshape(-1), expected[bank_idx])
    expected = _expected_rows(layer, 3)
    for bank_idx, (_host, gpu_cache) in enumerate(cache.banks):
        assert torch.equal(
            gpu_cache[7].contiguous().view(torch.uint8).reshape(-1), expected[bank_idx])
    # ...and the miss list shrank to the RAM-resident remainder.
    assert cache.num_indices.item() == 1
    assert cache.src_indices[0].item() == 0
    assert cache.evict_slots[0].item() == 5


def test_fetch_pending_all_ram_is_noop(checkpoint):
    cache = _fake_cache()
    tier = _tier(checkpoint, cache, ram_experts=2)
    cache.src_indices[:2] = torch.tensor([0, 1], dtype=torch.int32)
    cache.evict_slots[:2] = torch.tensor([0, 1], dtype=torch.int32)
    cache.num_indices.fill_(2)
    tier.fetch_pending(cache, 0)
    assert cache.num_indices.item() == 2
    assert tier.stats()["experts_fetched"] == 0


def test_fetch_pending_all_disk_clears_list(checkpoint):
    cache = _fake_cache()
    tier = _tier(checkpoint, cache, ram_experts=2)
    cache.src_indices[:1] = torch.tensor([3], dtype=torch.int32)
    cache.evict_slots[:1] = torch.tensor([4], dtype=torch.int32)
    cache.num_indices.fill_(1)
    tier.fetch_pending(cache, 0)
    assert cache.num_indices.item() == 0
    expected = _expected_rows(0, 3)
    for bank_idx, (_host, gpu_cache) in enumerate(cache.banks):
        assert torch.equal(
            gpu_cache[4].contiguous().view(torch.uint8).reshape(-1), expected[bank_idx])


def test_release_range_frees_pages():
    """release_range must actually drop the resident pages. The bank is a
    MAP_PRIVATE anonymous mapping, so MADV_DONTNEED frees for real; mincore
    verifies the pages are gone (a MAP_SHARED mapping would keep them)."""
    import ctypes as ct

    size = 4 * 1024 * 1024
    bank = HostBank((size,), torch.uint8)
    bank.tensor.fill_(7)  # fault every page in
    libc = ct.CDLL("libc.so.6", use_errno=True)
    libc.mincore.argtypes = [ct.c_void_p, ct.c_size_t, ct.POINTER(ct.c_ubyte)]
    libc.mincore.restype = ct.c_int

    def resident_pages(addr, nbytes):
        vec = (ct.c_ubyte * ((nbytes + 4095) // 4096))()
        assert libc.mincore(addr, nbytes, vec) == 0
        return sum(1 for b in vec if b & 1)

    pages = size // 4096
    assert resident_pages(bank.addr, size) == pages
    bank.release_range(0, size)
    assert resident_pages(bank.addr, size) == 0
    # The mapping stays valid: the refaulted pages read back as zeros.
    assert bank.tensor[0] == 0


def test_tail_unbacked_after_release():
    """Lazy-tail invariant (the disk-tier RAM math): after release_range, the
    tail rows [K, E) back NO pages -- until something writes them. mincore over
    the tail is the cheap startup check check_tail_unbacked() runs for real."""
    from freetoken.moe.disk_tier import release_bank_tails, tail_resident_bytes

    E, K = 8, 4
    bank = HostBank((E, 4096), torch.uint8)  # page-sized rows
    bank.tensor[:K].fill_(1)  # touch only the prefix
    release_bank_tails({"b": [bank]}, E, K)
    assert tail_resident_bytes(bank, E, K) == 0
    # The mapping still works: one tail write backs exactly one page (the
    # invariant is "nothing touches the tail", not "the tail refuses to back").
    bank.tensor[K].fill_(2)
    assert tail_resident_bytes(bank, E, K) == 4096


def test_release_bank_tails_unaligned_row_boundary():
    """A row boundary that is not page-aligned (the small scale banks) must not
    fail the boot: release_bank_tails warns and skips that bank instead of
    asserting in release_range. The tail rows were never written, so skipping
    loses nothing. Aligned boundaries still release."""
    import ctypes as ct

    from freetoken.moe.disk_tier import release_bank_tails

    libc = ct.CDLL("libc.so.6", use_errno=True)
    libc.mincore.argtypes = [ct.c_void_p, ct.c_size_t, ct.POINTER(ct.c_ubyte)]
    libc.mincore.restype = ct.c_int

    def resident_pages(addr, nbytes):
        vec = (ct.c_ubyte * ((nbytes + 4095) // 4096))()
        assert libc.mincore(addr, nbytes, vec) == 0
        return sum(1 for b in vec if b & 1)

    # A real Ornith gate_up_scale row size: 2048 bytes/row, NOT page-aligned.
    E, K = 256, 127
    bank = HostBank((E, 2048), torch.uint8)
    bank.tensor.fill_(7)  # fault every page in
    assert resident_pages(bank.addr, bank.nbytes) == bank.nbytes // 4096
    # K=127: offset = 127*2048 = 259072, not % 4096 -> warn+skip, no AssertionError.
    release_bank_tails({"gate_up_scale": [bank]}, E, K)
    # Skipped: the (already resident) tail pages are untouched, not freed.
    assert resident_pages(bank.addr, bank.nbytes) == bank.nbytes // 4096

    # Aligned K on the same bank shape: offset = 128*2048 = 262144 (% 4096) -> releases.
    bank2 = HostBank((E, 2048), torch.uint8)
    bank2.tensor.fill_(7)
    release_bank_tails({"gate_up_scale": [bank2]}, E, 128)
    assert resident_pages(bank2.addr + 128 * 2048, bank2.nbytes - 128 * 2048) == 0


def test_scalar_banks_preloaded_not_read_per_fetch(checkpoint):
    """weight_scale_2 banks (2/5) are 4-byte scalars per expert; they must be
    preloaded ONCE instead of costing an aligned disk read on every fetch.
    The fetched rows must stay bitwise identical to the file bytes."""
    index = _index(checkpoint)
    cache = _fake_cache()
    tier = _tier(checkpoint, cache)
    # Scalars cover every expert (RAM-resident ones included: fetch paths may
    # touch any slot in tests/materialize) and are read from the same bytes.
    blob = tier._scalar_blob(0, 2)
    assert len(blob) == E * 8  # gate + up fp32 scalars per expert
    for expert in range(E):
        gate, up = struct.unpack_from("<2f", blob, expert * 8)
        assert gate == _tensor_for(0, expert, "gate_proj", "weight_scale_2").item()
        assert up == _tensor_for(0, expert, "up_proj", "weight_scale_2").item()
    assert tier.stats()["scalar_preload_reads"] > 0

    # A full sweep of fetches: scalar rows correct, and the total preadv count
    # is strictly below the old one-syscall-per-segment behavior (9 per expert:
    # 2+2+2+1+1+1). Scalar segments (3 per expert) no longer hit disk at all.
    for layer in range(L):
        for expert in range(E):
            slot = (layer * E + expert) % 8
            tier._fetch_expert(layer, expert, slot)
            expected = _expected_rows(layer, expert)
            for bank_idx, (_host_layer, gpu_cache) in enumerate(cache.banks):
                got = gpu_cache[slot].contiguous().view(torch.uint8).reshape(-1)
                assert torch.equal(got, expected[bank_idx]), (bank_idx, layer, expert)
    st = tier.stats()
    assert st["experts_fetched"] == L * E
    assert st["preadv_calls"] <= 6 * L * E, st  # was 9 * L * E before the port


_ST_DTYPE = {torch.uint8: "U8", torch.float32: "F32", torch.float16: "F16"}


def _write_shard_manual(path, tensors):
    """Write a safetensors file with data packed in the GIVEN order.
    safetensors.torch.save_file sorts keys alphabetically, which scatters an
    expert's weight tensors -- but the REAL NVFP4 checkpoints pack them
    contiguously per expert (weights block, then scales block)."""
    header = {}
    data = bytearray()
    for name, t in tensors:
        b = t.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        header[name] = {"dtype": _ST_DTYPE[t.dtype], "shape": list(t.shape),
                        "data_offsets": [len(data), len(data) + len(b)]}
        data += b
    h = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h + bytes(data))


def _packed_checkpoint(tmp_path):
    """Checkpoint reproducing the real Qwen3.8-Flash-NVFP4 on-disk layout: per
    expert, down|gate|up weights contiguous, then the three scales contiguous,
    then the scalars (verified against model-00001-of-00010.safetensors)."""
    by_shard = {s: [] for s in SHARDS}
    weight_map = {}
    order = (("down_proj", "weight"), ("gate_proj", "weight"), ("up_proj", "weight"),
             ("down_proj", "weight_scale"), ("gate_proj", "weight_scale"),
             ("up_proj", "weight_scale"), ("down_proj", "weight_scale_2"),
             ("gate_proj", "weight_scale_2"), ("up_proj", "weight_scale_2"))
    for layer in range(L):
        for expert in range(E):
            shard = SHARDS[(layer * E + expert) % 2]
            for proj, kind in order:
                name = _name(layer, expert, proj, kind)
                by_shard[shard].append((name, _tensor_for(layer, expert, proj, kind)))
                weight_map[name] = shard
    for shard, tensors in by_shard.items():
        _write_shard_manual(tmp_path / shard, tensors)
    with open(tmp_path / "model.safetensors.index.json", "w", encoding="utf-8") as f:
        json.dump({"weight_map": weight_map, "metadata": None}, f)
    config = types.SimpleNamespace(num_experts=E, hidden_size=H, moe_intermediate_size=I,
                                   num_layers=L, first_k_dense_replace=0)
    return tmp_path, config


def test_adjacent_segments_share_one_preadv(tmp_path):
    """Real-checkpoint packing makes an expert's segments exactly adjacent --
    first within each bank (P0-3, 9 -> 4 preadv), and with R1x also ACROSS
    banks, so the fully-packed layout here collapses to a single merged
    extent: 1 call per fetch."""
    checkpoint = _packed_checkpoint(tmp_path)
    cache = _fake_cache()
    tier = _tier(checkpoint, cache)
    tier._fetch_expert(0, 0, 0)
    assert tier.stats()["preadv_calls"] == 1, tier.stats()
    # The merged reads still land the exact bytes (all 6 banks, both slices).
    expected = _expected_rows(0, 0)
    for bank_idx, (_host_layer, gpu_cache) in enumerate(cache.banks):
        got = gpu_cache[0].contiguous().view(torch.uint8).reshape(-1)
        assert torch.equal(got, expected[bank_idx]), bank_idx


# ------------------------------------------------------------- PILOT prefetch (P0-4)

def _tier_prefetch(checkpoint, cache, ram_experts=2, window=1, monkeypatch=None):
    import os
    os.environ["FT_DISK_TIER_PREFETCH"] = str(window)
    try:
        return _tier(checkpoint, cache, ram_experts=ram_experts)
    finally:
        os.environ.pop("FT_DISK_TIER_PREFETCH", None)


def _flush_stash(tier):
    """Wait for every in-flight prefetch read."""
    for entry in list(tier._stash.values()):
        entry["future"].result()


def test_prefetch_hit_skips_demand_disk_read(checkpoint):
    cache = _fake_cache()
    tier = _tier_prefetch(checkpoint, cache, ram_experts=2, window=1)
    # Layer 0 routes experts {2, 3} -> identity-prefetch them for layer 1.
    tier.prefetch_from_routing(0, torch.tensor([2, 3], dtype=torch.int32))
    _flush_stash(tier)
    reads_after_prefetch = tier._preadv_calls
    assert reads_after_prefetch > 0
    assert tier.stats()["prefetch_issued"] == 2

    # Layer 1 really does miss expert 2 (stash hit) but not 3 (wasted).
    cache.src_indices[:1] = torch.tensor([2], dtype=torch.int32)
    cache.evict_slots[:1] = torch.tensor([5], dtype=torch.int32)
    cache.num_indices.fill_(1)
    tier.fetch_pending(cache, 1)

    # The demand path did NO new disk reads: the slab covered it.
    assert tier._preadv_calls == reads_after_prefetch
    assert tier.stats()["prefetch_hits"] == 1
    assert tier.stats()["prefetch_wasted"] == 1
    # Slot 5 holds expert 2's bytes on every bank.
    expected = _expected_rows(1, 2)
    for bank_idx, (_host, gpu_cache) in enumerate(cache.banks):
        assert torch.equal(
            gpu_cache[5].contiguous().view(torch.uint8).reshape(-1),
            expected[bank_idx])
    assert cache.num_indices.item() == 0  # all misses were disk; list cleared


def test_prefetch_skips_slot_resident_expert(checkpoint):
    cache = _fake_cache()
    tier = _tier_prefetch(checkpoint, cache, ram_experts=2, window=1)
    # Make expert 2 slot-resident at layer 1 via a real demand fetch.
    cache.src_indices[:1] = torch.tensor([2], dtype=torch.int32)
    cache.evict_slots[:1] = torch.tensor([7], dtype=torch.int32)
    cache.num_indices.fill_(1)
    tier.fetch_pending(cache, 1)
    assert 2 in tier._resident_disk[1]
    assert tier._slot_owner[(1, 7)] == 2
    # PILOT at layer 0 predicts expert 2 for layer 1: already resident -> skip.
    tier.prefetch_from_routing(0, torch.tensor([2], dtype=torch.int32))
    assert tier.stats()["prefetch_skipped_resident"] == 1
    assert tier.stats()["prefetch_issued"] == 0
    assert not tier._stash


def test_prefetch_reservation_suppresses_duplicate(checkpoint):
    cache = _fake_cache()
    tier = _tier_prefetch(checkpoint, cache, ram_experts=2, window=1)
    tier.prefetch_from_routing(0, torch.tensor([3], dtype=torch.int32))
    tier.prefetch_from_routing(0, torch.tensor([3], dtype=torch.int32))
    _flush_stash(tier)
    assert tier.stats()["prefetch_issued"] == 1  # second call was a no-op


def test_prefetch_window_zero_disables(checkpoint, monkeypatch):
    monkeypatch.setenv("FT_DISK_TIER_PREFETCH", "0")  # default is 1 (on) since 2026-09-28
    cache = _fake_cache()
    tier = _tier(checkpoint, cache, ram_experts=2)
    tier.prefetch_from_routing(0, torch.tensor([2, 3], dtype=torch.int32))
    assert tier.stats()["prefetch_issued"] == 0
    assert not tier._stash


def test_slot_mirror_tracks_eviction(checkpoint):
    cache = _fake_cache()
    tier = _tier(checkpoint, cache, ram_experts=2)
    # Slot 5 first takes disk expert 2...
    cache.src_indices[:1] = torch.tensor([2], dtype=torch.int32)
    cache.evict_slots[:1] = torch.tensor([5], dtype=torch.int32)
    cache.num_indices.fill_(1)
    tier.fetch_pending(cache, 1)
    assert 2 in tier._resident_disk[1]
    # ...then layer 1 reuses slot 5 for RAM expert 1 -> mirror drops expert 2.
    cache.src_indices[:1] = torch.tensor([1], dtype=torch.int32)
    cache.evict_slots[:1] = torch.tensor([5], dtype=torch.int32)
    cache.num_indices.fill_(1)
    tier.fetch_pending(cache, 1)
    assert 2 not in tier._resident_disk[1]
    assert tier._slot_owner[(1, 5)] == 1


# ---------------------------------------------------------------- P1 telemetry
def test_route_histogram_observes_routing_and_decays(checkpoint):
    from freetoken.moe.disk_tier import DiskTier

    cache = _fake_cache()
    tier = _tier(checkpoint, cache)  # window=0: telemetry must still count
    tier.prefetch_from_routing(0, torch.tensor([0, 2, 2, 3], dtype=torch.int32))
    tier.prefetch_from_routing(1, torch.tensor([2, 3], dtype=torch.int32))
    hist = tier.route_histogram()
    assert hist[0].tolist() == [1.0, 0.0, 2.0, 1.0]
    assert hist[1].tolist() == [0.0, 0.0, 1.0, 1.0]
    # Decay fires after 256 layer-0 observations (one per decode token).
    for _ in range(256):
        tier.prefetch_from_routing(0, torch.tensor([0], dtype=torch.int32))
    hist = tier.route_histogram()
    assert hist[0, 0] < 1.0 + 256  # old mass decayed by 0.97
    assert abs(hist[0, 2] - 2.0 * 0.97) < 1e-9
    assert tier.stats()["route_hist_total"] > 0


def test_histogram_persistence_round_trip(checkpoint, tmp_path):
    """The routing-mass histogram is PILOT heat telemetry; it must round-trip."""
    cache = _fake_cache()
    tier = _tier(checkpoint, cache)
    tier.prefetch_from_routing(0, torch.tensor([0, 1, 1], dtype=torch.int32))
    p = tmp_path / "hist.json"
    tier.save_histogram(str(p))
    tier2 = _tier(checkpoint, _fake_cache())
    tier2.load_histogram(str(p))
    assert torch.equal(tier.route_histogram(), tier2.route_histogram())


# ---------------------------------------------------------------- P1 LFRU admission
def test_hits_bitmap_and_turn_persistence(checkpoint, tmp_path, monkeypatch):
    """HITS telemetry: source classification + per-turn JSONL persistence."""
    tele = tmp_path / "telemetry.jsonl"
    monkeypatch.setenv("FT_DISK_TIER_TELEMETRY", str(tele))
    cache = _fake_cache()
    tier = _tier(checkpoint, cache, ram_experts=2)
    # Layer 0 misses: expert 1 (RAM-resident) + expert 3 (disk read).
    cache.src_indices[:2] = torch.tensor([1, 3], dtype=torch.int32)
    cache.evict_slots[:2] = torch.tensor([0, 1], dtype=torch.int32)
    cache.num_indices.fill_(2)
    tier.fetch_pending(cache, 0)
    assert tier._turn_src[0, 1].item() == 0  # RAM bank over PCIe
    assert tier._turn_src[0, 3].item() == 2  # disk read
    rec = tier.end_turn()
    assert rec["served_ram"] == 1 and rec["served_disk"] == 1
    assert (tier._turn_src == -1).all()  # reset after end_turn
    line = json.loads(tele.read_text().strip())
    assert line["turn"] == 0 and "0" in line["layers"]
    assert tier.end_turn()["turn"] == 1


def test_hits_bitmap_stash_classification(checkpoint):
    """Stash hits classify as 1 in the HITS map."""
    cache = _fake_cache()
    tier = _tier_prefetch(checkpoint, cache, ram_experts=2, window=1)
    tier.prefetch_from_routing(0, torch.tensor([3], dtype=torch.int32))
    _flush_stash(tier)
    cache.src_indices[:1] = torch.tensor([3], dtype=torch.int32)
    cache.evict_slots[:1] = torch.tensor([5], dtype=torch.int32)
    cache.num_indices.fill_(1)
    tier.fetch_pending(cache, 1)
    assert tier._turn_src[1, 3].item() == 1  # stash hit


def test_mark_turn_src_survives_outside_inference_mode(checkpoint):
    """Regression (2026-10-03): the doorbell thread writes the HITS bitmap.

    ``_turn_src`` is built under the engine's inference mode, so it is an inference
    tensor; a raw in-place write from the graph-doorbell service thread -- which runs
    outside that scope -- raises ``RuntimeError: Inplace update to inference tensor
    outside InferenceMode is not allowed``. That exception killed the service thread
    and hung the engine in a prefill spin for 22 minutes. Both ``mark_turn_src`` and
    ``end_turn`` must therefore be usable from that thread.
    """
    cache = _fake_cache()
    tier = _tier_prefetch(checkpoint, cache, ram_experts=2, window=1)
    with torch.inference_mode():  # exactly how the engine builds it
        tier._turn_src = torch.full(
            tier._turn_src.shape, -1, dtype=tier._turn_src.dtype,
            device=tier._turn_src.device)
    assert torch.is_inference(tier._turn_src)
    with pytest.raises(RuntimeError):  # the pre-fix failure mode
        tier._turn_src[1, 3] = 1
    tier.mark_turn_src(1, 3, 1)
    assert tier._turn_src[1, 3].item() == 1
    tier.end_turn()  # must not raise from the doorbell thread either
    assert tier._turn_src[1, 3].item() == -1


# ------------------------------------------------------------- owner-local EP
def _ownership(rank, world=2):
    from freetoken.moe.ownership import ExpertOwnership

    return ExpertOwnership(global_num_experts=E, world_size=world, rank=rank)


def test_owner_ram_prefix_uses_the_rank_share_as_given(checkpoint):
    """``ram_experts`` here is already this rank's share of the host budget.

    The loader resolves the host-wide count once (``local_ram_experts``) and
    hands the same number to ``release_bank_tails`` and to the tier, so splitting
    it again divides the share a second time (measured on the EP=2 auto budget:
    120 -> 64/56 -> 32/24). With the old global-prefix alias, rank 1's whole
    range sat above the prefix and it pinned nothing at all.
    """
    # rank 0 owns global {0,1}: what it was handed is exactly what it pins.
    tier0 = _tier(checkpoint, _fake_cache(num_experts=2), ram_experts=2,
                  ownership=_ownership(0))
    assert (tier0._g0, tier0._local_num, tier0._ram) == (0, 2, 2)
    # rank 1 owns global {2,3} and pins its own (not the global) share.
    tier1 = _tier(checkpoint, _fake_cache(num_experts=2), ram_experts=1,
                  ownership=_ownership(1))
    assert (tier1._g0, tier1._local_num, tier1._ram) == (2, 2, 1)
    # A share wider than the local range clamps to it; zero is legal (this rank
    # is fully disk-resident) and must not resurrect a global prefix.
    tier2 = _tier(checkpoint, _fake_cache(num_experts=2), ram_experts=5,
                  ownership=_ownership(1))
    assert tier2._ram == 2
    tier3 = _tier(checkpoint, _fake_cache(num_experts=2), ram_experts=0,
                  ownership=_ownership(1))
    assert tier3._ram == 0


def test_local_ram_experts_split_math():
    """The split preserves the host-wide total on the release-alignment grid."""
    from freetoken.moe.disk_tier import local_ram_experts
    from freetoken.moe.ownership import ExpertOwnership

    def own(rank, world=2, e=384):
        return ExpertOwnership(global_num_experts=e, world_size=world, rank=rank)

    # EP=2 auto budget on this host (120 experts/ep): 64/56, total 120 kept.
    assert [local_ram_experts(120, own(r)) for r in range(2)] == [64, 56]
    # Full pin: every rank gets its whole local range.
    assert [local_ram_experts(384, own(r)) for r in range(2)] == [192, 192]
    # No ownership -> the global count is unchanged.
    assert local_ram_experts(120, None) == 120
    # A single rank keeps the whole budget.
    assert local_ram_experts(120, own(0, world=1)) == 120


def test_owner_fetch_reads_global_rows(checkpoint):
    """The slot cache names LOCAL rows; the tier must read the owning rank's
    GLOBAL checkpoint rows."""
    ownership = _ownership(rank=1)  # global {2,3} as local {0,1}
    cache = _fake_cache(num_experts=2)
    tier = _tier(checkpoint, cache, ram_experts=1, ownership=ownership)
    for layer in range(L):
        for local in range(2):
            slot = (layer * 2 + local) % 8
            tier._fetch_expert(layer, local, slot)
            expected = _expected_rows(layer, 2 + local)
            for bank_idx, (_host, gpu_cache) in enumerate(cache.banks):
                got = gpu_cache[slot].contiguous().view(torch.uint8).reshape(-1)
                assert torch.equal(got, expected[bank_idx]), (bank_idx, layer, local)


def test_owner_fetch_pending_local_namespace(checkpoint):
    """fetch_pending classifies RAM/disk against the LOCAL clamped prefix."""
    ownership = _ownership(rank=1)
    cache = _fake_cache(num_experts=2)
    tier = _tier(checkpoint, cache, ram_experts=1, ownership=ownership)  # local ram = 1
    # Miss list (LOCAL ids): local 0 (global 2, RAM), local 1 (global 3, disk).
    cache.src_indices[:2] = torch.tensor([0, 1], dtype=torch.int32)
    cache.evict_slots[:2] = torch.tensor([5, 6], dtype=torch.int32)
    cache.num_indices.fill_(2)
    tier.fetch_pending(cache, 1)
    # local 1 fetched from GLOBAL row 3...
    expected = _expected_rows(1, 3)
    for bank_idx, (_host, gpu_cache) in enumerate(cache.banks):
        assert torch.equal(
            gpu_cache[6].contiguous().view(torch.uint8).reshape(-1),
            expected[bank_idx])
    # ...and the miss list shrank to the RAM-resident local 0.
    assert cache.num_indices.item() == 1
    assert cache.src_indices[0].item() == 0
    assert cache.evict_slots[0].item() == 5


def test_owner_prefetch_filters_to_owned(checkpoint):
    """PILOT routing arrives GLOBAL: remote experts must be dropped and owned
    ones renumbered to local rows before stash/demand bookkeeping."""
    import os

    os.environ["FT_DISK_TIER_PREFETCH"] = "1"
    try:
        ownership = _ownership(rank=1)
        cache = _fake_cache(num_experts=2)
        tier = _tier(checkpoint, cache, ram_experts=1, ownership=ownership)
    finally:
        os.environ.pop("FT_DISK_TIER_PREFETCH", None)
    # Global routing {0 (remote), 2 (owned, RAM local 0), 3 (owned, disk local 1)}:
    # exactly ONE prefetch (local 1) is issued.
    tier.prefetch_from_routing(0, torch.tensor([0, 2, 3], dtype=torch.int32))
    _flush_stash(tier)
    assert tier.stats()["prefetch_issued"] == 1
    # Demand miss on local 1 hits the stash with GLOBAL row 3's bytes.
    cache.src_indices[:1] = torch.tensor([1], dtype=torch.int32)
    cache.evict_slots[:1] = torch.tensor([5], dtype=torch.int32)
    cache.num_indices.fill_(1)
    tier.fetch_pending(cache, 1)
    expected = _expected_rows(1, 3)
    for bank_idx, (_host, gpu_cache) in enumerate(cache.banks):
        assert torch.equal(
            gpu_cache[5].contiguous().view(torch.uint8).reshape(-1),
            expected[bank_idx])
    assert tier.stats()["prefetch_hits"] == 1
    assert cache.num_indices.item() == 0


# ---------------------------------------------------------------------------
# --moe-disk-tier auto: capacity-adaptive resolution (pure math)
# ---------------------------------------------------------------------------
from freetoken.moe.disk_tier import _AUTO_ALIGN, auto_ram_experts

_GIB = 1 << 30
# Representative geometry: Qwen3.8-Flash-Next-NVFP4 — 48 layers x 512 experts,
# ~30.9 MiB per expert per layer -> ~1.45 GiB per expert across depth,
# ~744 GiB for the full set (unit-test numbers, not the production model's).
_ROW = 32 * 1024 * 1024
_LAYERS = 48
_EXPERTS = 512
_RESERVE = 16 * _GIB


def test_auto_dormant_when_full_set_fits_with_headroom():
    total = _ROW * _LAYERS * _EXPERTS
    available = int(total * 1.25) + _RESERVE  # exactly at the dormancy boundary
    assert auto_ram_experts(_ROW, _LAYERS, _EXPERTS, available, _RESERVE) is None
    assert auto_ram_experts(_ROW, _LAYERS, _EXPERTS, 1024 * _GIB, _RESERVE) is None


def test_auto_active_just_below_dormancy_boundary():
    total = _ROW * _LAYERS * _EXPERTS
    available = int(total * 1.25) + _RESERVE - 1  # one byte short of dormant
    k = auto_ram_experts(_ROW, _LAYERS, _EXPERTS, available, _RESERVE)
    assert k is not None and 0 < k < _EXPERTS
    # Budget-limited, aligned down, and the aligned prefix must actually fit.
    assert k % _AUTO_ALIGN == 0
    assert k * _ROW * _LAYERS <= (available - _RESERVE) * 0.95


def test_auto_active_scales_with_budget():
    # 400 GiB available - 16 GiB reserve -> 384 * 0.95 / 1.5 GiB-per-expert ~ 243
    k = auto_ram_experts(_ROW, _LAYERS, _EXPERTS, 400 * _GIB, _RESERVE)
    layer_cost = _ROW * _LAYERS
    assert k == int((400 * _GIB - _RESERVE) * 0.95 // layer_cost) // _AUTO_ALIGN * _AUTO_ALIGN
    # Half the budget -> roughly half the prefix (still aligned).
    k_half = auto_ram_experts(_ROW, _LAYERS, _EXPERTS, 208 * _GIB, _RESERVE)
    assert 0 < k_half < k and k_half % _AUTO_ALIGN == 0


def test_auto_rejects_budget_below_one_aligned_prefix():
    with pytest.raises(ValueError, match="fewer than"):
        auto_ram_experts(_ROW, _LAYERS, _EXPERTS, _RESERVE + _GIB, _RESERVE)
    with pytest.raises(ValueError, match="no RAM budget"):
        auto_ram_experts(_ROW, _LAYERS, _EXPERTS, _RESERVE, _RESERVE)


# ------------------------------------------------------------- learned pin set
def test_pin_rows_to_row_map_packs_pinned_prefix():
    from freetoken.moe.disk_tier import pin_rows_to_row_map

    # pin {1,3} of 4: rows 0/1 hold experts 1/3, the rest fill ascending.
    rm = pin_rows_to_row_map([[1, 3], [0, 2]], 4)
    assert rm[0] == [2, 0, 3, 1]  # e0->2, e1->0, e2->3, e3->1
    assert rm[1] == [0, 2, 1, 3]  # e0->0, e1->2, e2->1, e3->3
    # The contiguous prefix is the identity (the no-pin-file layout).
    assert pin_rows_to_row_map([[0, 1], [0, 1]], 4) == [[0, 1, 2, 3]] * 2
    # Empty pin set: identity map, every expert disk-resident (row >= 0... the
    # caller compares against ram_rows == 0).
    assert pin_rows_to_row_map([[], []], 4) == [[0, 1, 2, 3]] * 2
    with pytest.raises(ValueError, match="unique sorted"):
        pin_rows_to_row_map([[3, 1], [0, 1]], 4)
    with pytest.raises(ValueError, match="unique sorted"):
        pin_rows_to_row_map([[1, 1], [0, 1]], 4)
    with pytest.raises(ValueError, match="unique sorted"):
        pin_rows_to_row_map([[1, 4], [0, 1]], 4)


def _pin_doc(tmp_path, budgets=(2, 2), e=E, layers=L, ep=2):
    def rows(n):
        return [list(range(n)) for _ in range(layers)]

    doc = {
        "format": "freetoken.ram_pin_set.v1",
        "num_layers": layers,
        "num_experts": e,
        "ep": ep,
        "budgets": list(budgets),
        "sources": ["test"],
        "ranks": {str(r): rows(budgets[r]) for r in range(ep)},
    }
    path = tmp_path / "pin.json"
    path.write_text(json.dumps(doc))
    return str(path), doc


def test_load_ram_pin_doc_schema(tmp_path):
    from freetoken.moe.disk_tier import load_ram_pin_doc

    path, doc = _pin_doc(tmp_path)
    assert load_ram_pin_doc(path)["budgets"] == [2, 2]
    # A non-prefix pin set is accepted: unique sorted local ids in range
    # (the local window is [0, E/ep) -- 4 here, so {1, 3} is a legal pin set).
    _, doc = _pin_doc(tmp_path, budgets=(2, 2), e=8, layers=L, ep=2)
    doc["ranks"]["0"] = [[1, 3]] * L
    path = tmp_path / "pin2.json"
    path.write_text(json.dumps(doc))
    assert load_ram_pin_doc(path)["ranks"]["0"][0] == [1, 3]
    for bad in (
        {"format": "nope"},
        {"budgets": [2]},                 # one budget for ep=2
        {"ranks": {"0": [[0, 1]] * L}},   # rank 1 missing
        {"ranks": {"0": [[0]] * L, "1": [[0, 1]] * L}},  # budget mismatch
        {"ranks": {"0": [[1, 0]] * L, "1": [[0, 1]] * L}},  # unsorted
        {"ranks": {"0": [[0, 4]] * L, "1": [[0, 1]] * L}},  # out of window
    ):
        _, doc2 = _pin_doc(tmp_path)
        doc2.update(bad)
        p = tmp_path / "bad.json"
        p.write_text(json.dumps(doc2))
        with pytest.raises(ValueError):
            load_ram_pin_doc(str(p))


def test_validate_ram_pin_doc_against_model_and_budget(tmp_path):
    from freetoken.moe.disk_tier import validate_ram_pin_doc

    _, doc = _pin_doc(tmp_path, budgets=(2, 2), e=4, layers=2, ep=2)
    assert validate_ram_pin_doc(doc, 4, 2, 2, 4) == []
    # A stale file fails the boot: budget 2 splits 1/1, not the file's 2/2.
    assert validate_ram_pin_doc(doc, 4, 2, 2, 2)
    assert validate_ram_pin_doc(doc, 8, 2, 2, 4)  # num_experts mismatch
    assert validate_ram_pin_doc(doc, 4, 40, 2, 4)  # num_layers mismatch
    assert validate_ram_pin_doc(doc, 4, 2, 4, 4)  # ep mismatch
    # The production split (120 over EP=2) is 64/56, mirroring local_ram_experts.
    _, doc2 = _pin_doc(tmp_path, budgets=(64, 56), e=384, layers=40, ep=2)
    assert validate_ram_pin_doc(doc2, 384, 40, 2, 120) == []
    assert validate_ram_pin_doc(doc2, 384, 40, 2, 128) != []  # splits 64/64


def test_disk_tier_pin_rows_builds_row_map(checkpoint):
    rows = [[1, 3], [0, 2]]
    tier = _tier(checkpoint, _fake_cache(), ram_experts=2, pin_rows=rows)
    assert tier._remapped
    assert tier._row_map_cpu[0].tolist() == [2, 0, 3, 1]
    assert tier._row_map_cpu[1].tolist() == [0, 2, 1, 3]
    assert tier._pin_ids_cpu[0].tolist() == [1, 3]
    assert tier._nonpin_dev[0].tolist() == [0, 2]
    assert tier._nonpin_dev[1].tolist() == [1, 3]
    assert tier.stats()["pin_remapped"] is True
    # A prefix pin set is the identity: the no-pin-file layout, bit for bit.
    tier2 = _tier(checkpoint, _fake_cache(), ram_experts=2,
                  pin_rows=[[0, 1], [0, 1]])
    assert not tier2._remapped
    assert tier2._row_map_cpu[0].tolist() == [0, 1, 2, 3]
    assert tier2.stats()["pin_remapped"] is False


def test_routed_disk_uses_row_map(checkpoint):
    tier = _tier(checkpoint, _fake_cache(), ram_experts=2,
                 pin_rows=[[1, 3], [0, 2]])
    # Layer 0 pins {1,3}: only 0 and 2 are disk-resident (unique, sorted).
    disk0 = tier._routed_disk(0, torch.tensor([0, 1, 2, 3, 1]))
    assert disk0.tolist() == [0, 2]
    # Layer 1 pins {0,2}: only 1 and 3 are disk-resident.
    disk1 = tier._routed_disk(1, torch.tensor([0, 1, 2, 3, 0]))
    assert disk1.tolist() == [1, 3]


def test_fetch_pending_remap_writes_bank_rows(checkpoint):
    """Learned pin set {1,3} on layer 0 (rows [2,0,3,1]): expert 0/2 are
    disk-resident, 1/3 pinned. The compacted RAM remainder must carry BANK
    ROWS (0/1), not expert ids -- that is what the PCIe copy indexes."""
    cache = _fake_cache()
    tier = _tier(checkpoint, cache, ram_experts=2, pin_rows=[[1, 3], [0, 2]])
    cache.src_indices[:3] = torch.tensor([0, 1, 3], dtype=torch.int32)
    cache.evict_slots[:3] = torch.tensor([5, 6, 7], dtype=torch.int32)
    cache.num_indices.fill_(3)

    tier.fetch_pending(cache, 0)

    # Expert 0 (row 2 >= ram) was disk-fetched into its slot.
    expected = _expected_rows(0, 0)
    for bank_idx, (_host, gpu_cache) in enumerate(cache.banks):
        assert torch.equal(
            gpu_cache[5].contiguous().view(torch.uint8).reshape(-1), expected[bank_idx])
    # The RAM remainder compacts to BANK ROWS [0, 1] (experts 1 and 3).
    assert cache.num_indices.item() == 2
    assert cache.src_indices[:2].tolist() == [0, 1]
    assert cache.evict_slots[:2].tolist() == [6, 7]
    # Layer 1's pin set {0,2} is independent: expert 1 is disk-resident there.
    cache.src_indices[:1] = torch.tensor([1], dtype=torch.int32)
    cache.evict_slots[:1] = torch.tensor([4], dtype=torch.int32)
    cache.num_indices.fill_(1)
    tier.fetch_pending(cache, 1)
    assert cache.num_indices.item() == 0
    expected = _expected_rows(1, 1)
    for bank_idx, (_host, gpu_cache) in enumerate(cache.banks):
        assert torch.equal(
            gpu_cache[4].contiguous().view(torch.uint8).reshape(-1), expected[bank_idx])


def test_fetch_pending_remap_all_ram_still_translates(checkpoint):
    """Regression for the W13 eager crash: when EVERY miss is pinned
    (``disk == []``) the early return must still translate src_indices from
    local expert ids to bank rows. The identity layout made the old early
    return a harmless no-op; with a learned pin set it left expert ids (e.g.
    118/155) in the PCIe copy plan, which then read the released,
    non-registered host tail and faulted (async IMA, surfaced later during
    CUDA graph capture at bs=4)."""
    cache = _fake_cache()
    tier = _tier(checkpoint, cache, ram_experts=2, pin_rows=[[1, 3], [0, 2]])
    # Both misses are pinned experts on layer 0 (1 -> bank row 0, 3 -> row 1).
    cache.src_indices[:2] = torch.tensor([1, 3], dtype=torch.int32)
    cache.evict_slots[:2] = torch.tensor([5, 6], dtype=torch.int32)
    cache.num_indices.fill_(2)

    tier.fetch_pending(cache, 0)

    # Nothing was disk-fetched, but the plan now carries BANK ROWS.
    assert cache.num_indices.item() == 2
    assert cache.src_indices[:2].tolist() == [0, 1]
    assert cache.evict_slots[:2].tolist() == [5, 6]


def _fill_pin_rows(cache, layout):
    """Simulate the checkpoint loader: bank row r of each layer holds the
    pinned expert layout[layer][r]'s content (all 6 banks)."""
    for bank_idx, (host_layers, _gpu) in enumerate(cache.banks):
        for layer, rows in enumerate(layout):
            for r, expert in enumerate(rows):
                row = _expected_rows(layer, expert)[bank_idx]
                host_layers[layer][r].contiguous().view(torch.uint8).reshape(-1).copy_(row)


def test_apply_pin_swaps_hot_swaps_row_content_and_maps(checkpoint):
    """Admission apply (FREETOKEN_ADMISSION_APPLY): the challenger's disk row
    replaces the incumbent's RAM slot and the two trade places in the pin
    maps; every other row stays put. Layer 0 pins {1,3} (map [2,0,3,1]):
    swap ((0,1),(0,2)) moves expert 2 into bank row 0, demotes expert 1."""
    cache = _fake_cache()
    tier = _tier(checkpoint, cache, ram_experts=2, pin_rows=[[1, 3], [0, 2]])
    _fill_pin_rows(cache, [[1, 3], [0, 2]])

    report = tier.apply_pin_swaps([((0, 1), (0, 2))])

    assert report["applied"] == [((0, 1), (0, 2))]
    assert report["failed"] == 0
    assert report["io_bytes"] == tier._db_row_bytes
    # Bank row 0 of layer 0 now holds expert 2's checkpoint bytes (6 banks).
    expected = _expected_rows(0, 2)
    for bank_idx, (host_layers, _gpu) in enumerate(cache.banks):
        assert torch.equal(
            host_layers[0][0].contiguous().view(torch.uint8).reshape(-1),
            expected[bank_idx])
    # The untouched pin row (expert 3 in row 1) is byte-identical.
    expected3 = _expected_rows(0, 3)
    for bank_idx, (host_layers, _gpu) in enumerate(cache.banks):
        assert torch.equal(
            host_layers[0][1].contiguous().view(torch.uint8).reshape(-1),
            expected3[bank_idx])
    # Maps flipped; layer 1 untouched; bijection preserved.
    assert tier._row_map_cpu[0].tolist() == [2, 3, 0, 1]
    assert tier._row_map_cpu[1].tolist() == [0, 2, 1, 3]
    assert sorted(tier._row_map_cpu[0].tolist()) == [0, 1, 2, 3]
    assert tier._pin_ids_cpu[0].tolist() == [2, 3]
    assert tier._nonpin_dev[0].tolist() == [0, 1]
    # Device tensors updated IN PLACE (captured graphs baked their pointers).
    assert tier._row_map_dev[0].tolist() == [2, 3, 0, 1]
    assert tier._pin_ids_dev[0].tolist() == [2, 3]
    assert tier._pin_ids_dev_i64[0].tolist() == [2, 3]
    # Residency classification follows the new layout: experts 0/1 disk now.
    assert tier._routed_disk(0, torch.tensor([0, 1, 2, 3])).tolist() == [0, 1]
    assert tier._remapped
    # Bookkeeping: apply does its own accounting, NOT fetch demand.
    s = tier.stats()
    assert s["apply_swaps"] == 1
    assert s["apply_failures"] == 0
    assert s["apply_io_bytes"] == tier._db_row_bytes
    assert s["apply_host_ms"] >= 0.0
    assert tier._fetches == 0
    # The demoted expert still serves from disk through the normal fetch path.
    tier._fetch_expert_inner(0, 1, 5)
    for bank_idx, (_host, gpu_cache) in enumerate(cache.banks):
        assert torch.equal(
            gpu_cache[5].contiguous().view(torch.uint8).reshape(-1),
            _expected_rows(0, 1)[bank_idx])
    assert tier._fetches == 1


def test_apply_pin_swaps_disk_read_failure_keeps_old_row(checkpoint, monkeypatch):
    """Prepare-phase preadv failure: the pair is dropped with the layout
    untouched (old row kept, maps unflipped), counted as an apply failure."""
    cache = _fake_cache()
    tier = _tier(checkpoint, cache, ram_experts=2, pin_rows=[[1, 3], [0, 2]])
    _fill_pin_rows(cache, [[1, 3], [0, 2]])

    def _boom(fd, bufs, off):
        raise OSError(5, "injected EIO")

    monkeypatch.setattr("os.preadv", _boom)
    report = tier.apply_pin_swaps([((0, 1), (0, 2))])

    assert report["applied"] == []
    assert report["failed"] == 1
    expected = _expected_rows(0, 1)
    for bank_idx, (host_layers, _gpu) in enumerate(cache.banks):
        assert torch.equal(
            host_layers[0][0].contiguous().view(torch.uint8).reshape(-1),
            expected[bank_idx])
    assert tier._row_map_cpu[0].tolist() == [2, 0, 3, 1]
    assert tier._pin_ids_cpu[0].tolist() == [1, 3]
    s = tier.stats()
    assert s["apply_swaps"] == 0
    assert s["apply_failures"] == 1
    assert s["apply_io_bytes"] == 0


def test_apply_pin_swaps_updates_bulk_plan_snapshot(checkpoint):
    """F1 regression (2b x 3): _row_map_list is the construction-time tolist
    mirror feeding build_prefetch_plan. A hot swap must flip its two rows in
    the mirror too, or the plan keeps excluding the demoted expert (lost
    prefetch) and prefetching the newly-pinned one (wasted disk read)."""
    cache = _fake_cache()
    tier = _tier(checkpoint, cache, ram_experts=2, pin_rows=[[1, 3], [0, 2]])
    _fill_pin_rows(cache, [[1, 3], [0, 2]])
    assert tier._row_map_list[0] == [2, 0, 3, 1]

    report = tier.apply_pin_swaps([((0, 1), (0, 2))])

    assert report["applied"] == [((0, 1), (0, 2))]
    # The mirror follows the flip: expert 2 pinned (row 0), expert 1 demoted.
    assert tier._row_map_list[0] == [2, 3, 0, 1]
    assert tier._row_map_list[1] == [0, 2, 1, 3]
    # The plan's pin exclusion now follows the NEW layout: layer 0's plan
    # covers the disk-resident {0,1} and not the newly-pinned expert 2.
    from freetoken.moe.prefetch_plan import build_prefetch_plan
    plan = build_prefetch_plan(
        [{0: 5.0, 1: 1.0, 2: 4.0, 3: 2.0},
         {0: 5.0, 1: 1.0, 2: 4.0, 3: 2.0}],
        tier._row_map_list, tier._ram, start_layer=-1, depth=2, top_k=8)
    assert plan[0] == [0, 1]  # pre-fix stale mirror gave [0, 2]
    assert plan[1] == [3, 1]  # layer 1 untouched: pins {0,2} excluded


class _SegIndex:
    """Handcrafted segment layouts for _row_groups merge-math tests."""
    scalar_banks = (2, 5)

    def __init__(self, segs):
        self._segs = segs

    def row_segments(self, bank_idx, layer, expert):
        return self._segs[(bank_idx, layer, expert)]


def _row_groups_tier(segs, n_banks=6):
    t = object.__new__(DiskTier)
    t._index = _SegIndex(segs)
    t._convert = False
    t._g0 = 0
    t._banks = [None] * n_banks
    t._dst_slices = [[(0, 100)]] * n_banks
    t._fd = lambda shard: (None, False)  # direct=False: exact (unaligned) extents
    return t


def _per_bank_members(t, layer, expert):
    out = []
    for b in range(len(t._banks)):
        if b in t._index.scalar_banks:
            continue
        for _s, _a0, _a1, members, _e in t._group_runs(b, layer, expert):
            for (d0, d1, off, nbytes) in members:
                out.append((b, d0, d1, off, nbytes))
    return sorted(out)


def test_row_groups_merges_cross_bank_adjacency():
    # v41-style: banks 0/1/3 (weights) contiguous 1000..4000; bank 4 (scale) far away.
    segs = {(0, 0, 0): [(0, 1000, 1000)], (1, 0, 0): [(0, 2000, 1000)],
            (3, 0, 0): [(0, 3000, 1000)], (4, 0, 0): [(0, 9000, 500)]}
    t = _row_groups_tier(segs)
    groups = t._row_groups(0, 0)
    assert len(groups) == 2
    shard, a0, a1, members, end = groups[0]
    assert (shard, a0, a1, end) == (0, 1000, 4000, 4000)
    assert sorted(m[0] for m in members) == [0, 1, 3]
    # Byte conservation: identical member multiset as the per-bank plan.
    merged = sorted((m[0], m[1], m[2], m[3], m[4])
                    for g in groups for m in g[3])
    assert merged == _per_bank_members(t, 0, 0)


def test_row_groups_no_merge_across_gap_or_shard():
    # 1-byte gap between bank 0 and 1; bank 3 on another shard: no cross-bank merge.
    segs = {(0, 0, 0): [(0, 1000, 1000)], (1, 0, 0): [(0, 2001, 1000)],
            (3, 0, 0): [(1, 500, 1000)], (4, 0, 0): [(0, 3001, 500)]}
    t = _row_groups_tier(segs)
    groups = t._row_groups(0, 0)
    # bank1+bank4 are adjacent (2001+1000==3001) -> one merged group; bank0 and
    # bank3 stand alone: 3 groups, vs 4 per-bank groups.
    assert len(groups) == 3
    assert len(_per_bank_members(t, 0, 0)) == 4
    merged = sorted((m[0], m[1], m[2], m[3], m[4])
                    for g in groups for m in g[3])
    assert merged == _per_bank_members(t, 0, 0)


def test_row_groups_excludes_scalar_banks():
    segs = {(0, 0, 0): [(0, 1000, 1000)], (1, 0, 0): [(0, 2000, 1000)],
            (3, 0, 0): [(0, 3000, 1000)], (4, 0, 0): [(0, 9000, 500)]}
    t = _row_groups_tier(segs)
    members = [m for g in t._row_groups(0, 0) for m in g[3]]
    assert all(m[0] not in (2, 5) for m in members)


# ---------------------------------------------------------------------------
# M1: stats() surfaces the doorbell ledger (host counters + bridge spin mirror).


class _StatsBridge:
    def spin_stats(self):
        return {
            "doorbell_spins": 3,
            "doorbell_wait_ms": 50.0,
            "doorbell_wait_ms_peak": 100.0,
        }

    def raise_if_unhealthy(self):
        return None


def _bare_tier() -> DiskTier:
    t = object.__new__(DiskTier)
    t._fetches = 0
    t._fetch_bytes = 0
    t._preadv_calls = 0
    t._scalar_preload_reads = 0
    t._pf_issued = 0
    t._pf_hits = 0
    t._pf_wasted = 0
    t._pf_skipped_resident = 0
    t._route_hist = torch.zeros(8)
    t._remapped = 0
    t._db_requests = 2
    t._db_rows = 5
    t._db_bytes = 40960
    t._db_host_us = 1500
    t._db_timeouts = 0
    t._db_row_bytes = 8192
    # ②a prefill data-source split fields, defaults == OFF mode.
    t._prefill_pin_source = False
    t._pf_pin_rows = 0
    t._pf_pin_bytes = 0
    t._pf_disk_rows = 0
    t._pf_disk_bytes = 0
    # 2b bulk prefetch bookkeeping fields, defaults == OFF mode.
    t._prefill_bulk = False
    t._bk_issued = 0
    t._bk_hits = 0
    t._bk_wasted = 0
    t._bk_bytes = 0
    # Admission apply ledger, defaults == zero.
    t._apply_swaps = 0
    t._apply_failures = 0
    t._apply_io_bytes = 0
    t._apply_host_ns = 0
    return t


def test_stats_carries_doorbell_ledger_without_bridge():
    t = _bare_tier()
    s = t.stats()
    assert s["doorbell_requests"] == 2
    assert s["doorbell_rows"] == 5
    assert s["doorbell_bytes"] == 40960
    assert s["doorbell_host_ms"] == 1.5
    assert s["doorbell_timeouts"] == 0
    # No bridge yet (capture not done): no spin fields at all.
    assert "doorbell_wait_ms" not in s
    assert "doorbell_spins" not in s


def test_stats_merges_bridge_spin_mirror_and_timeouts():
    t = _bare_tier()
    t._graph_bridge = _StatsBridge()
    t.record_doorbell_timeout()
    s = t.stats()
    assert s["doorbell_timeouts"] == 1
    assert s["doorbell_spins"] == 3
    assert s["doorbell_wait_ms"] == 50.0
    assert s["doorbell_wait_ms_peak"] == 100.0
    # raise_if_unhealthy delegates to the bridge when present.
    t.raise_if_unhealthy()  # stub bridge has no _err: returns cleanly
# --------------------------------------------- ②a prefill data-source split
def _fill_pinned_host_rows(cache, pin_rows):
    """Back the host banks' pinned rows [0, ram) with the pinned experts'
    content (what the loader does in production): bank row r of layer L holds
    expert pin_rows[L][r]."""
    for layer, rows in enumerate(pin_rows):
        for row, expert in enumerate(rows):
            expected = _expected_rows(layer, expert)
            for bank_idx, (per_layer, _gpu) in enumerate(cache.banks):
                dst = per_layer[layer][row].contiguous().view(torch.uint8).reshape(-1)
                dst.copy_(expected[bank_idx])


def _prefill_buffers(cache, depth=2, fill=0xFF):
    """Borrowed-ring stand-in: [depth, E, *shape] per bank, sentinel-filled
    BYTE-wise (an fp16 tensor filled with the VALUE 0xFF would be 255.0)."""
    bufs = [
        torch.empty((depth, cache.num_experts, *shape), dtype=dtype)
        for shape, dtype in zip(BANK_SHAPES, BANK_DTYPES)
    ]
    for buf in bufs:
        buf.view(torch.uint8).fill_(fill)
    cache.prefill_bank_buffers = bufs
    return bufs


def _buffer_rows(buffers, buffer_id, expert):
    return [buf[buffer_id][expert].contiguous().view(torch.uint8).reshape(-1)
            for buf in buffers]


def test_routed_source_split_identity_layout(checkpoint):
    """Default prefix pin (no pin file): experts < ram are RAM-pinned."""
    tier = _tier(checkpoint, _fake_cache(), ram_experts=2)
    pin, disk = tier._routed_split(1, torch.tensor([0, 1, 2, 3, 1]))
    assert pin.tolist() == [0, 1]  # unique, sorted
    assert disk.tolist() == [2, 3]
    # Unrouted experts are excluded from both sides.
    pin, disk = tier._routed_split(1, torch.tensor([1, 3]))
    assert pin.tolist() == [1]
    assert disk.tolist() == [3]
    # Empty routing is legal (a layer with no routed tokens).
    pin, disk = tier._routed_split(1, torch.tensor([], dtype=torch.int64))
    assert pin.numel() == 0 and disk.numel() == 0


def test_routed_source_split_uses_row_map(checkpoint):
    """Learned pin set: the split follows row_map, not the expert id."""
    tier = _tier(checkpoint, _fake_cache(), ram_experts=2,
                 pin_rows=[[1, 3], [0, 2]])
    # Layer 0 pins {1,3}; layer 1 pins {0,2}.
    pin, disk = tier._routed_split(0, torch.tensor([0, 1, 2, 3]))
    assert pin.tolist() == [1, 3]
    assert disk.tolist() == [0, 2]
    pin, disk = tier._routed_split(1, torch.tensor([0, 1, 2, 3]))
    assert pin.tolist() == [0, 2]
    assert disk.tolist() == [1, 3]


def test_routed_source_split_owner_ep(checkpoint):
    """Owner-local EP: GLOBAL routing is renumbered to local rows and remote
    experts dropped before the row_map lookup."""
    ownership = _ownership(rank=1)  # global {2,3} as local {0,1}, ram=1
    tier = _tier(checkpoint, _fake_cache(num_experts=2), ram_experts=1,
                 ownership=ownership)
    # Global routing {0,1 (remote), 2 (-> local 0, pinned), 3 (-> local 1, disk)}.
    pin, disk = tier._routed_split(0, torch.tensor([0, 1, 2, 3]))
    assert pin.tolist() == [0]
    assert disk.tolist() == [1]
    # A remote-only routing splits to nothing.
    pin, disk = tier._routed_split(0, torch.tensor([0, 1]))
    assert pin.numel() == 0 and disk.numel() == 0


def test_fetch_routed_into_off_fetches_disk_only_and_counts_split(
        checkpoint, monkeypatch):
    """Flag OFF (default == current behavior): the fetch plan touches only the
    routed DISK rows; the routed pin rows are the ring copy's job, so their
    buffer rows stay untouched. The ②a counters still record the split."""
    monkeypatch.delenv("FREETOKEN_PREFILL_PIN_SOURCE", raising=False)
    cache = _fake_cache()
    tier = _tier(checkpoint, cache, ram_experts=2, pin_rows=[[1, 3], [0, 2]])
    assert tier._prefill_pin_source is False
    assert tier.stats()["prefill_pin_source"] is False
    _fill_pinned_host_rows(cache, [[1, 3], [0, 2]])
    buffers = _prefill_buffers(cache)

    fetched = tier.fetch_routed_into(cache, 0, torch.tensor([0, 1, 2, 3]), 0)

    assert fetched == 2  # disk-resident {0, 2}
    for expert in (0, 2):
        expected = _expected_rows(0, expert)
        got = _buffer_rows(buffers, 0, expert)
        for bank_idx in range(len(buffers)):
            assert torch.equal(got[bank_idx], expected[bank_idx]), bank_idx
    # Pin rows {1, 3}: OFF -> not the fetch plan's job, sentinel preserved.
    for expert in (1, 3):
        got = _buffer_rows(buffers, 0, expert)
        for bank_idx in range(len(buffers)):
            assert int(got[bank_idx][0]) == 0xFF and int(got[bank_idx][-1]) == 0xFF
    stats = tier.stats()
    assert stats["prefill_pin_rows"] == 2
    assert stats["prefill_disk_rows"] == 2
    assert stats["prefill_pin_bytes"] == 2 * sum(tier._row_bytes)
    assert stats["prefill_disk_bytes"] == 2 * tier._db_row_bytes


def test_fetch_routed_into_pin_source_on_serves_pin_rows(
        checkpoint, monkeypatch):
    """Flag ON: the routed PINNED rows are served from the HostBank pinned
    rows (row_map translation), the routed disk rows from preadv -- the buffer
    ends up complete without any ring prefix copy."""
    monkeypatch.setenv("FREETOKEN_PREFILL_PIN_SOURCE", "1")
    cache = _fake_cache()
    tier = _tier(checkpoint, cache, ram_experts=2, pin_rows=[[1, 3], [0, 2]])
    assert tier._prefill_pin_source is True
    _fill_pinned_host_rows(cache, [[1, 3], [0, 2]])
    buffers = _prefill_buffers(cache)

    fetched = tier.fetch_routed_into(cache, 0, torch.tensor([0, 1, 2, 3]), 0)

    assert fetched == 2  # return value stays the disk-read count
    for expert in (0, 1, 2, 3):
        expected = _expected_rows(0, expert)
        got = _buffer_rows(buffers, 0, expert)
        for bank_idx in range(len(buffers)):
            assert torch.equal(got[bank_idx], expected[bank_idx]), (bank_idx, expert)
    stats = tier.stats()
    assert stats["prefill_pin_source"] is True
    assert stats["prefill_pin_rows"] == 2
    assert stats["prefill_disk_rows"] == 2

    # Partial routing: only routed rows are served, the rest stay stale
    # (never gathered by the grouped GEMM -- same argument as the unrouted
    # disk tail in OFF mode).
    tier.fetch_routed_into(cache, 1, torch.tensor([1]), 1)
    expected = _expected_rows(1, 1)  # expert 1 is disk-resident on layer 1
    got = _buffer_rows(buffers, 1, 1)
    for bank_idx in range(len(buffers)):
        assert torch.equal(got[bank_idx], expected[bank_idx]), bank_idx
    for expert in (0, 2, 3):
        got = _buffer_rows(buffers, 1, expert)
        for bank_idx in range(len(buffers)):
            assert int(got[bank_idx][0]) == 0xFF and int(got[bank_idx][-1]) == 0xFF
    stats = tier.stats()
    assert stats["prefill_pin_rows"] == 2  # expert 1 is disk on layer 1
    assert stats["prefill_disk_rows"] == 3


def test_fetch_routed_into_pin_source_identity_layout(
        checkpoint, monkeypatch):
    """Flag ON with the default prefix pin (identity row_map): pinned experts
    0/1 come from host bank rows 0/1, disk experts 2/3 from preadv."""
    monkeypatch.setenv("FREETOKEN_PREFILL_PIN_SOURCE", "1")
    cache = _fake_cache()
    tier = _tier(checkpoint, cache, ram_experts=2)  # prefix pin {0,1}
    assert not tier._remapped
    _fill_pinned_host_rows(cache, [[0, 1], [0, 1]])
    buffers = _prefill_buffers(cache)

    tier.fetch_routed_into(cache, 1, torch.tensor([0, 1, 2, 3]), 1)

    for expert in (0, 1, 2, 3):
        expected = _expected_rows(1, expert)
        got = _buffer_rows(buffers, 1, expert)
        for bank_idx in range(len(buffers)):
            assert torch.equal(got[bank_idx], expected[bank_idx]), (bank_idx, expert)


# ---------------------------------------------------------------------------
# 2b: block-level bulk prefetch (FREETOKEN_PREFILL_BULK_*; default OFF).

def _bulk_prediction(tmp_path, hist):
    p = tmp_path / "bulk_pred.json"
    p.write_text(json.dumps({"layers": len(hist), "experts": len(hist[0]),
                             "hist": hist}))
    return str(p)


def _flush_bulk(tier):
    """Wait for every in-flight bulk block read (one future per block slab)."""
    recs = {id(entry["rec"]): entry["rec"]
            for entry in tier._bulk_stash.values()}
    for rec in recs.values():
        rec["future"].result()


def test_bulk_prefetch_off_by_default(checkpoint, monkeypatch):
    """No env: the 2b machinery is inert (off == byte-identical current
    behavior) and stats() still surfaces the bookkeeping keys."""
    for var in ("FREETOKEN_PREFILL_BULK_PREFETCH", "FREETOKEN_PREFILL_BULK_PREDICT"):
        monkeypatch.delenv(var, raising=False)
    tier = _tier(checkpoint, _fake_cache(), ram_experts=2)
    assert tier._prefill_bulk is False
    assert tier._bulk_pool is None
    stats = tier.stats()
    assert stats["prefill_bulk"] is False
    assert stats["prefill_bulk_issued"] == 0
    assert stats["prefill_bulk_hits"] == 0
    assert stats["prefill_bulk_wasted"] == 0
    assert stats["prefill_bulk_bytes"] == 0


def test_bulk_prefetch_requires_prediction_doc(checkpoint, monkeypatch, capsys):
    """Flag on but no learned distribution: disable with a log line, never
    speculate from nothing (the PILOT identity predictor measured 1.01%)."""
    monkeypatch.setenv("FREETOKEN_PREFILL_BULK_PREFETCH", "1")
    monkeypatch.delenv("FREETOKEN_PREFILL_BULK_PREDICT", raising=False)
    tier = _tier(checkpoint, _fake_cache(), ram_experts=2)
    assert tier._prefill_bulk is False
    assert "FREETOKEN_PREFILL_BULK_PREDICT" in capsys.readouterr().out


def test_bulk_prefetch_rejects_mismatched_doc(checkpoint, monkeypatch, tmp_path):
    """A prediction doc whose L x E does not match the checkpoint fails the
    boot -- speculating from the wrong distribution is worse than none."""
    pred = _bulk_prediction(tmp_path, [[1.0] * 4] * 3)  # 3 layers vs checkpoint 2
    monkeypatch.setenv("FREETOKEN_PREFILL_BULK_PREFETCH", "1")
    monkeypatch.setenv("FREETOKEN_PREFILL_BULK_PREDICT", pred)
    with pytest.raises(ValueError, match="does not match"):
        _tier(checkpoint, _fake_cache(), ram_experts=2)


def test_bulk_prefetch_hit_serves_rows_and_skips_demand_read(
        checkpoint, monkeypatch, tmp_path):
    """Rolling plan: layer 0's demand fetch issues the learned plan for
    layer 1; the block read lands in a shared slab; layer 1's routed disk rows
    are then served from the slab (zero preadv in the demand window) with
    bytes identical to the demand path. The pinned expert with the highest
    score (0: 5.0) must NOT be prefetched -- it is a host-bank row."""
    pred = _bulk_prediction(tmp_path, [[0.0] * 4, [5.0, 0.0, 1.0, 0.9]])
    monkeypatch.setenv("FREETOKEN_PREFILL_BULK_PREFETCH", "1")
    monkeypatch.setenv("FREETOKEN_PREFILL_BULK_PREDICT", pred)
    monkeypatch.setenv("FREETOKEN_PREFILL_BULK_DEPTH", "1")
    monkeypatch.setenv("FREETOKEN_PREFILL_BULK_TOPK", "8")
    monkeypatch.setenv("FREETOKEN_PREFILL_BULK_BLOCK", "2")
    cache = _fake_cache()
    tier = _tier(checkpoint, cache, ram_experts=2)  # prefix pin {0,1}
    assert tier._prefill_bulk is True
    _fill_pinned_host_rows(cache, [[0, 1], [0, 1]])
    buffers = _prefill_buffers(cache)

    # Layer 0: demand-served {2,3}; _bulk_advance(0) then plans layer 1.
    assert tier.fetch_routed_into(cache, 0, torch.tensor([0, 1, 2, 3]), 0) == 2
    assert tier._bk_issued == 2  # disk-resident {2,3}, pinned 0 excluded
    _flush_bulk(tier)
    preadv_after_bulk = tier._preadv_calls

    # Layer 1: both routed disk rows are stash hits.
    fetched = tier.fetch_routed_into(cache, 1, torch.tensor([0, 1, 2, 3]), 1)

    assert fetched == 2
    assert tier._preadv_calls == preadv_after_bulk  # no demand-window NVMe
    for expert in (2, 3):
        expected = _expected_rows(1, expert)
        got = _buffer_rows(buffers, 1, expert)
        for bank_idx in range(len(buffers)):
            assert torch.equal(got[bank_idx], expected[bank_idx]), bank_idx
    for expert in (0, 1):  # pin rows: the ring's job (2a OFF), sentinel kept
        got = _buffer_rows(buffers, 1, expert)
        for bank_idx in range(len(buffers)):
            assert int(got[bank_idx][0]) == 0xFF and int(got[bank_idx][-1]) == 0xFF
    assert tier._bk_hits == 2
    assert tier._bk_wasted == 0
    assert len(tier._bulk_slab_free) == 1  # block slab recycled after the hits
    stats = tier.stats()
    assert stats["prefill_bulk"] is True
    assert stats["prefill_bulk_issued"] == 2
    assert stats["prefill_bulk_hits"] == 2
    assert stats["prefill_bulk_wasted"] == 0
    assert stats["prefill_bulk_bytes"] > 0


def test_bulk_prefetch_miss_falls_back_to_demand(
        checkpoint, monkeypatch, tmp_path):
    """Empty learned distribution (zero scores): nothing is issued, and the
    next layer's fetch is the unchanged demand path -- correct rows, preadv
    reads, no hits and nothing wasted."""
    pred = _bulk_prediction(tmp_path, [[0.0] * 4, [0.0] * 4])
    monkeypatch.setenv("FREETOKEN_PREFILL_BULK_PREFETCH", "1")
    monkeypatch.setenv("FREETOKEN_PREFILL_BULK_PREDICT", pred)
    monkeypatch.setenv("FREETOKEN_PREFILL_BULK_DEPTH", "1")
    cache = _fake_cache()
    tier = _tier(checkpoint, cache, ram_experts=2)
    _fill_pinned_host_rows(cache, [[0, 1], [0, 1]])
    buffers = _prefill_buffers(cache)

    assert tier.fetch_routed_into(cache, 0, torch.tensor([0, 1, 2, 3]), 0) == 2
    assert tier._bk_issued == 0
    preadv_before = tier._preadv_calls
    assert tier.fetch_routed_into(cache, 1, torch.tensor([0, 1, 2, 3]), 1) == 2
    assert tier._preadv_calls > preadv_before  # demand reads happened
    for expert in (2, 3):
        expected = _expected_rows(1, expert)
        got = _buffer_rows(buffers, 1, expert)
        for bank_idx in range(len(buffers)):
            assert torch.equal(got[bank_idx], expected[bank_idx]), bank_idx
    assert tier._bk_hits == 0
    assert tier._bk_wasted == 0


def test_bulk_prefetch_wasted_recycles_the_block_slab(
        checkpoint, monkeypatch, tmp_path):
    """Prefetched but never routed: the entries go stale at the next advance,
    count as wasted, and the shared block slab returns to the free pool."""
    pred = _bulk_prediction(tmp_path, [[0.0] * 4, [0.0, 0.0, 1.0, 0.9]])
    monkeypatch.setenv("FREETOKEN_PREFILL_BULK_PREFETCH", "1")
    monkeypatch.setenv("FREETOKEN_PREFILL_BULK_PREDICT", pred)
    monkeypatch.setenv("FREETOKEN_PREFILL_BULK_DEPTH", "1")
    monkeypatch.setenv("FREETOKEN_PREFILL_BULK_BLOCK", "2")
    cache = _fake_cache()
    tier = _tier(checkpoint, cache, ram_experts=2)
    _fill_pinned_host_rows(cache, [[0, 1], [0, 1]])
    _prefill_buffers(cache)

    tier.fetch_routed_into(cache, 0, torch.tensor([0, 1, 2, 3]), 0)
    assert tier._bk_issued == 2
    _flush_bulk(tier)
    # Layer 1 routes only the pinned experts: the prefetched {2,3} go stale.
    assert tier.fetch_routed_into(cache, 1, torch.tensor([0, 1]), 1) == 0
    assert tier._bk_hits == 0
    assert tier._bk_wasted == 2
    assert len(tier._bulk_stash) == 0
    assert len(tier._bulk_slab_free) == 1


def test_bulk_prefetch_failed_block_read_still_returns_the_slab(
        checkpoint, monkeypatch, tmp_path):
    """F5 regression (2b): a block read that raises in the pool thread (here
    an injected EIO) must not strand its slab -- the refcounted recycle frees
    it back to the pool even though the read error itself stays loud."""
    pred = _bulk_prediction(tmp_path, [[0.0] * 4, [0.0, 0.0, 1.0, 0.9]])
    monkeypatch.setenv("FREETOKEN_PREFILL_BULK_PREFETCH", "1")
    monkeypatch.setenv("FREETOKEN_PREFILL_BULK_PREDICT", pred)
    monkeypatch.setenv("FREETOKEN_PREFILL_BULK_DEPTH", "1")
    monkeypatch.setenv("FREETOKEN_PREFILL_BULK_BLOCK", "2")
    cache = _fake_cache()
    tier = _tier(checkpoint, cache, ram_experts=2)
    _fill_pinned_host_rows(cache, [[0, 1], [0, 1]])
    _prefill_buffers(cache)

    def _boom(fd, bufs, off):
        raise OSError(5, "injected EIO")

    monkeypatch.setattr("os.preadv", _boom)
    # Layer 0 routes only the pinned experts: no demand read (which would
    # also hit preadv), and _bulk_advance still issues the layer-1 block.
    assert tier.fetch_routed_into(cache, 0, torch.tensor([0, 1]), 0) == 0
    assert tier._bk_issued == 2
    for rec in {id(e["rec"]): e["rec"] for e in tier._bulk_stash.values()}.values():
        with pytest.raises(OSError):
            rec["future"].result()  # wait for the failed block read
    # Layer 1 again routes only pins: the stale recycle of the failed block
    # raises (loud) but the slab still returns to the free pool.
    with pytest.raises(OSError):
        tier.fetch_routed_into(cache, 1, torch.tensor([0, 1]), 1)
    assert tier._bk_wasted == 2
    assert len(tier._bulk_stash) == 0
    assert len(tier._bulk_slab_free) == 1


def test_bulk_prefetch_failed_block_hit_path_recycles_popped_entries(
        checkpoint, monkeypatch, tmp_path):
    """F5 regression, hit path: _stash_to_buffer re-raises the block read
    error; the entries already popped from the stash must still be drained
    and recycled so their slab is not lost before the error propagates."""
    pred = _bulk_prediction(tmp_path, [[0.0] * 4, [0.0, 0.0, 1.0, 0.9]])
    monkeypatch.setenv("FREETOKEN_PREFILL_BULK_PREFETCH", "1")
    monkeypatch.setenv("FREETOKEN_PREFILL_BULK_PREDICT", pred)
    monkeypatch.setenv("FREETOKEN_PREFILL_BULK_DEPTH", "1")
    monkeypatch.setenv("FREETOKEN_PREFILL_BULK_BLOCK", "2")
    cache = _fake_cache()
    tier = _tier(checkpoint, cache, ram_experts=2)
    _fill_pinned_host_rows(cache, [[0, 1], [0, 1]])
    _prefill_buffers(cache)

    def _boom(fd, bufs, off):
        raise OSError(5, "injected EIO")

    monkeypatch.setattr("os.preadv", _boom)
    # Layer 0 routes only the pinned experts (no demand read); the layer-1
    # block is issued behind it and fails in the pool thread.
    assert tier.fetch_routed_into(cache, 0, torch.tensor([0, 1]), 0) == 0
    assert tier._bk_issued == 2
    for rec in {id(e["rec"]): e["rec"] for e in tier._bulk_stash.values()}.values():
        with pytest.raises(OSError):
            rec["future"].result()  # wait for the failed block read
    # Layer 1 routes the prefetched experts: the hit path pops them, the
    # failed read raises out of _stash_to_buffer, and the recycle still runs.
    with pytest.raises(OSError):
        tier.fetch_routed_into(cache, 1, torch.tensor([0, 1, 2, 3]), 1)
    assert len(tier._bulk_stash) == 0
    assert len(tier._bulk_slab_free) == 1


def test_row_groups_multi_merges_across_experts():
    """The block read plan: two experts whose extents chain adjacently on one
    shard merge into a single group; members stay expert-tagged and the member
    multiset is byte-identical to the union of the per-expert plans."""
    segs = {
        # Expert 0: banks 0/1/3/4 contiguous 1000..4500 (the R1x rule).
        (0, 0, 0): [(0, 1000, 1000)], (1, 0, 0): [(0, 2000, 1000)],
        (3, 0, 0): [(0, 3000, 1000)], (4, 0, 0): [(0, 4000, 500)],
        # Expert 1: the chain continues 4500..8000 -- cross-EXPERT merge.
        (0, 0, 1): [(0, 4500, 1000)], (1, 0, 1): [(0, 5500, 1000)],
        (3, 0, 1): [(0, 6500, 1000)], (4, 0, 1): [(0, 7500, 500)],
    }
    t = _row_groups_tier(segs)
    groups = t._row_groups_multi(0, [0, 1])
    assert len(groups) == 1
    assert len(groups) < len(t._row_groups(0, 0)) + len(t._row_groups(0, 1))
    shard, a0, a1, members, end = groups[0]
    assert (shard, a0, a1, end) == (0, 1000, 8000, 8000)
    got = sorted(tuple(m) for m in members)
    want = sorted(
        (e, b, d0, d1, off, nb)
        for e in (0, 1)
        for (b, d0, d1, off, nb) in _per_bank_members(t, 0, e))
    assert got == want


def test_row_groups_multi_no_merge_across_gap_or_shard():
    # Expert 0 -> 1 has a 1-byte gap (2000 -> 2001); expert 2 lives on shard 1.
    # Nothing merges: the multi plan is exactly the per-expert plans' union.
    segs = {}
    for e, (shard, base) in enumerate(((0, 1000), (0, 2001), (1, 1000))):
        for b in (0, 1, 3, 4):
            segs[(b, 0, e)] = [(shard, base + b * 100000, 1000)]
    t = _row_groups_tier(segs)
    groups = t._row_groups_multi(0, [0, 1, 2])
    assert len(groups) == sum(len(t._row_groups(0, e)) for e in (0, 1, 2))
    got = sorted((m[0], m[1], m[2], m[3], m[4], m[5])
                 for g in groups for m in g[3])
    want = sorted(
        (e, b, d0, d1, off, nb)
        for e in (0, 1, 2)
        for (b, d0, d1, off, nb) in _per_bank_members(t, 0, e))
    assert got == want
