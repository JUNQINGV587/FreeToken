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


def _tier(checkpoint, cache, ram_experts=2, ownership=None):
    index = _index(checkpoint)
    tier = DiskTier(index, cache, ram_experts=ram_experts, workers=2,
                    ownership=ownership)
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
    """Real-checkpoint packing makes an expert's gate|up weights (and scales)
    exactly adjacent, so they merge into one read per bank: banks 0/1/3/4 cost
    1 preadv each, scalar banks cost 0 -> 4 calls per fetch (was 9)."""
    checkpoint = _packed_checkpoint(tmp_path)
    cache = _fake_cache()
    tier = _tier(checkpoint, cache)
    tier._fetch_expert(0, 0, 0)
    assert tier.stats()["preadv_calls"] == 4, tier.stats()
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


# ---------------------------------------------------------------- P1 telemetry / AUTOPIN
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


def test_autopin_pinned_count_respects_budget_and_mass():
    from freetoken.moe.disk_tier import autopin_pinned_count

    # E=8, all routing mass on experts 0..1 -> tiny K even with a huge budget.
    hist = torch.zeros((2, 8), dtype=torch.float64)
    hist[:, 0] = 1000.0
    hist[:, 1] = 1000.0
    k = autopin_pinned_count(hist, budget_bytes=1 << 30, ram_row_bytes=1 << 20)
    assert k == 2  # coverage knee: pinning beyond expert 1 buys nothing

    # Uniform mass: K limited by the 0.5*budget*min(1, obs/200000) share.
    hist = torch.ones((2, 8), dtype=torch.float64) * 1000.0  # 16000 obs
    k = autopin_pinned_count(hist, budget_bytes=1 << 30, ram_row_bytes=1 << 20)
    # share = 0.5 * 1GiB * 16000/200000 = ~40 MiB -> 40 rows, capped by E=8.
    assert k == 8
    hist = torch.ones((2, 8), dtype=torch.float64) * 10.0  # 160 obs -> cold
    k = autopin_pinned_count(hist, budget_bytes=1 << 30, ram_row_bytes=1 << 20)
    assert k == 0  # min(1, 160/200000) shrinks the share below one row

    # No observations -> no pins.
    assert autopin_pinned_count(torch.zeros((2, 8)), 1 << 30, 1 << 20) == 0


def test_autopin_advice_and_histogram_persistence(checkpoint, tmp_path):
    cache = _fake_cache()
    tier = _tier(checkpoint, cache)
    tier.prefetch_from_routing(0, torch.tensor([0, 1, 1], dtype=torch.int32))
    assert 0 <= tier.autopin_advice(budget_bytes=1 << 30) <= 4
    p = tmp_path / "hist.json"
    tier.save_histogram(str(p))
    tier2 = _tier(checkpoint, _fake_cache())
    tier2.load_histogram(str(p))
    assert torch.equal(tier.route_histogram(), tier2.route_histogram())


# ---------------------------------------------------------------- P1 LFRU admission
def test_tier_admission_semantics_match_colibri_contract():
    from freetoken.moe import tier_admission as ta

    # Hysteresis: 25% + 4 margin. cold=100 -> promote needs hot > 129.
    assert not ta.should_promote(129, 100)
    assert ta.should_promote(130, 100)
    assert ta.should_promote(5, 0)          # fixed margin handles tiny samples
    assert not ta.should_promote(4, 0)
    assert ta.decay_value(7) == 3

    # Sticky saturation: huge float mass clamps to uint32 max, never wraps.
    h = ta.heat_u32(torch.tensor([1e30, 5.0]))
    assert int(h[0]) == (1 << 32) - 1 and int(h[1]) == 5

    # Recency is a tiebreak only: 1 frequency point (256) beats max recency (255).
    fresh_cold = ta.lfru_score(heat=1, last=100, clock=100)   # 256|255
    stale_hot = ta.lfru_score(heat=2, last=0, clock=100)      # 512|0
    assert stale_hot > fresh_cold

    # pick_lfru: pinned {0,1}, expert 2 much hotter -> swap coldest pinned out.
    heat = torch.tensor([10, 50, 200], dtype=torch.int64)
    last = torch.tensor([0, 0, 0], dtype=torch.int64)
    swap = ta.pick_lfru(heat, last, clock=0, pinned=[0, 1])
    assert swap is not None and swap[0] == 0 and swap[1] == 2 and swap[2] > 0
    # Below the hysteresis margin -> no swap.
    heat2 = torch.tensor([10, 12, 13], dtype=torch.int64)
    assert ta.pick_lfru(heat2, last, clock=0, pinned=[0, 1]) is None
    # pick_swap (frequency-only variant) same contract.
    assert ta.pick_swap(heat, pinned=[0, 1]) == (0, 2, 190)
    assert ta.pick_swap(torch.tensor([1, 1, 1]), pinned=[0, 1]) is None


def test_disk_tier_pin_advice_uses_telemetry(checkpoint):
    cache = _fake_cache()
    tier = _tier(checkpoint, cache)  # ram_experts=2 -> pinned prefix [0, 1]
    assert tier.pin_advice(0) is None  # no observations yet
    for _ in range(300):
        tier.prefetch_from_routing(0, torch.tensor([2, 2, 3], dtype=torch.int32))
    advice = tier.pin_advice(0)
    # Expert 2 dominates routing mass -> it should be advised into a RAM row,
    # evicting the colder of the pinned {0, 1}.
    assert advice is not None and advice[1] == 2 and advice[0] in (0, 1)


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


# ------------------------------------------------------------- owner-local EP
def _ownership(rank, world=2):
    from freetoken.moe.ownership import ExpertOwnership

    return ExpertOwnership(global_num_experts=E, world_size=world, rank=rank)


def test_owner_ram_prefix_clamps(checkpoint):
    """ram_experts is a GLOBAL count; each rank clamps it to its owned prefix."""
    # rank 0 owns global {0,1}: global ram=3 -> local ram = 2 (everything RAM).
    tier0 = _tier(checkpoint, _fake_cache(num_experts=2), ram_experts=3,
                  ownership=_ownership(0))
    assert (tier0._g0, tier0._local_num, tier0._ram) == (0, 2, 2)
    # rank 1 owns global {2,3}: local ram = 3 - 2 = 1.
    tier1 = _tier(checkpoint, _fake_cache(num_experts=2), ram_experts=3,
                  ownership=_ownership(1))
    assert (tier1._g0, tier1._local_num, tier1._ram) == (2, 2, 1)
    # global ram=1 stops before rank 1's range -> rank 1 has NO RAM experts.
    tier2 = _tier(checkpoint, _fake_cache(num_experts=2), ram_experts=1,
                  ownership=_ownership(1))
    assert tier2._ram == 0


def test_owner_fetch_reads_global_rows(checkpoint):
    """The slot cache names LOCAL rows; the tier must read the owning rank's
    GLOBAL checkpoint rows."""
    ownership = _ownership(rank=1)  # global {2,3} as local {0,1}
    cache = _fake_cache(num_experts=2)
    tier = _tier(checkpoint, cache, ram_experts=3, ownership=ownership)
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
    tier = _tier(checkpoint, cache, ram_experts=3, ownership=ownership)  # local ram = 1
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
        tier = _tier(checkpoint, cache, ram_experts=3, ownership=ownership)
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
