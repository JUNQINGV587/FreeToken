"""GPU tests for classed LRU admission (freetoken.moe.classed_ensure).

Geometry: 2 layers x 16 experts, 64 slots. The disk tier pins experts 0..7
(``ram_local=8``, identity row map), so experts 8..15 are the disk class. The
wrapper sizes the pin region as ``min(FREETOKEN_PIN_CACHE_SLOTS=64, 64//4) = 16``
slots, i.e. ``pin_start=48``; the per-call guard admits ``numel <= min(16, 48)``.
"""
from __future__ import annotations

import pytest
import torch

from freetoken.moe.offload_kernels import ensure_experts

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

NUM_LAYERS = 2
NUM_EXPERTS = 16
CACHE_SIZE = 64
RAM_LOCAL = 8  # experts 0..7 are RAM-pin rows (identity row map)
PIN_REGION = 16  # min(64, CACHE_SIZE // 4)
PIN_START = CACHE_SIZE - PIN_REGION  # 48


class _FakeTier:
    def __init__(self):
        self._ram = RAM_LOCAL
        self._row_map_dev = torch.arange(
            NUM_EXPERTS, dtype=torch.int32, device="cuda"
        ).repeat(NUM_LAYERS, 1).contiguous()


class _FakeCache:
    """Minimal stand-in for OffloadExpertCache, mirroring its real dtypes."""

    def __init__(self, with_tier: bool = True):
        self.num_layers = NUM_LAYERS
        self.num_experts = NUM_EXPERTS
        self.cache_size = CACHE_SIZE
        self.slot_for_id = torch.full(
            (NUM_LAYERS, NUM_EXPERTS), -1, dtype=torch.int32, device="cuda")
        self.id_of_slot = torch.full(
            (CACHE_SIZE,), -1, dtype=torch.int32, device="cuda")
        self.usage = torch.zeros((CACHE_SIZE,), dtype=torch.int64, device="cuda")
        self.step = torch.zeros((), dtype=torch.int64, device="cuda")
        self.evict_slots = torch.empty((CACHE_SIZE,), dtype=torch.int32, device="cuda")
        self.src_indices = torch.empty((CACHE_SIZE,), dtype=torch.int32, device="cuda")
        self.num_indices = torch.zeros((1,), dtype=torch.int64, device="cuda")
        self.collect_stats = False
        self.lru_stats = torch.zeros((NUM_LAYERS, 3), dtype=torch.int64, device="cuda")
        self._disk_tier = _FakeTier() if with_tier else None


def _run(cache, calls):
    """calls: list of (layer_id, [local ids]); returns rewritten slot ids per call."""
    outs = []
    for layer_id, local in calls:
        ids = torch.tensor(local, dtype=torch.int32, device="cuda")
        ensure_experts(cache, layer_id, ids)
        outs.append(ids)
    torch.cuda.synchronize()
    return outs


def _assert_maps_consistent(cache):
    slot_for_id = cache.slot_for_id.view(-1).cpu()
    id_of_slot = cache.id_of_slot.cpu()
    for slot, owner in enumerate(id_of_slot.tolist()):
        if owner >= 0:
            assert slot_for_id[owner].item() == slot, f"map skew at slot {slot}"
    for flat, slot in enumerate(slot_for_id.tolist()):
        if slot >= 0:
            assert id_of_slot[slot].item() == flat, f"map skew for id {flat}"


def test_misses_land_in_their_class_regions():
    cache = _FakeCache()
    disk_out, pin_out = _run(cache, [
        (0, [8, 9, 10, 11]),   # disk class
        (0, [0, 1, 2, 3]),     # pin class
    ])
    assert (disk_out < PIN_START).all(), f"disk rows in pin region: {disk_out}"
    assert (pin_out >= PIN_START).all(), f"pin rows in main region: {pin_out}"
    _assert_maps_consistent(cache)


def test_pin_churn_never_evicts_disk_residents():
    cache = _FakeCache()
    # Fill with every disk expert of both layers (16 distinct ids, main region 48).
    _run(cache, [(layer, list(range(8, 16))) for layer in range(NUM_LAYERS)])
    # Churn pins hard: 8 distinct pin ids/layer through a 16-slot region, 12 rounds.
    for r in range(12):
        lo, hi = r % 8, (r + 4) % 8
        subset = sorted({(lo + i) % 8 for i in range(4)} | {(hi + i) % 8 for i in range(4)})
        _run(cache, [(layer, subset) for layer in range(NUM_LAYERS)])
    slot_for_id = cache.slot_for_id.cpu()
    for layer in range(NUM_LAYERS):
        resident = slot_for_id[layer, 8:16]
        assert (resident >= 0).all(), (
            f"layer {layer} disk experts evicted by pin churn: {resident}")
        assert (resident < PIN_START).all()
    _assert_maps_consistent(cache)


def test_disk_misses_never_occupy_pin_region():
    cache = _FakeCache()
    for r in range(10):  # mixed traffic
        _run(cache, [
            (0, [8 + (r % 8), 8 + ((r + 1) % 8)]),
            (1, [r % 8, (r + 3) % 8]),
        ])
    id_of_slot = cache.id_of_slot.cpu()
    for slot in range(PIN_START, CACHE_SIZE):
        owner = id_of_slot[slot].item()
        if owner >= 0:
            assert owner % NUM_EXPERTS < RAM_LOCAL, (
                f"disk-class id {owner} in pin-region slot {slot}")
    _assert_maps_consistent(cache)


def test_duplicates_collapse_to_one_copy():
    cache = _FakeCache()
    (out,) = _run(cache, [(0, [9, 9, 9, 2, 2])])
    assert int(cache.num_indices.item()) == 2
    assert out[0].item() == out[1].item() == out[2].item()
    assert out[3].item() == out[4].item()
    assert out[0].item() != out[3].item()
    _assert_maps_consistent(cache)


def test_stats_accumulate_like_vanilla():
    cache = _FakeCache()
    cache.collect_stats = True
    _run(cache, [(0, [8, 9, 0]), (0, [8, 10, 1])])  # 6 active, 3+2 misses, 2 calls
    active, miss, calls = cache.lru_stats[0].cpu().tolist()
    assert (active, miss, calls) == (6, 5, 2)


def test_wrapper_falls_back_to_vanilla_for_wide_queries():
    # numel = 10 > pin_region(16)? No: 10 <= 16 -> classed. Use numel = 20 > 16.
    cache = _FakeCache()
    wide = torch.tensor([0, 1, 2, 3] * 5, dtype=torch.int32, device="cuda")  # 20, pin class
    ensure_experts(cache, 0, wide)
    torch.cuda.synchronize()
    # Vanilla cold-cache victims are slots 0,1,2,3 (all usage 0, argmin -> lowest):
    # a pin class id in the main region proves the fallback ran.
    assert (wide < PIN_START).all(), f"expected vanilla placement, got {wide}"
    _assert_maps_consistent(cache)


def test_cuda_graph_capture_and_replay():
    cache = _FakeCache()
    twin = _FakeCache()  # eager reference, stepped in lockstep (same history)
    query = torch.tensor([8, 9, 0, 1], dtype=torch.int32, device="cuda")
    # Eager warm-up on both (also JIT-compiles the kernel outside capture).
    for _ in range(2):
        ensure_experts(cache, 0, torch.tensor([8, 9, 0, 1], dtype=torch.int32,
                                              device="cuda"))
        ensure_experts(twin, 0, torch.tensor([8, 9, 0, 1], dtype=torch.int32,
                                             device="cuda"))
    torch.cuda.synchronize()

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        ensure_experts(cache, 0, query)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ensure_experts(cache, 0, query)

    for values in ([10, 11, 2, 3], [8, 0, 12, 4], [9, 9, 5, 13]):
        v = torch.tensor(values, dtype=torch.int32, device="cuda")
        query.copy_(v)          # v still holds local ids here
        graph.replay()          # rewrites query -> slots
        ensure_experts(twin, 0, v)  # rewrites v -> slots
        torch.cuda.synchronize()
        assert torch.equal(cache.slot_for_id, twin.slot_for_id)
        assert torch.equal(cache.id_of_slot, twin.id_of_slot)
        assert torch.equal(query, v), f"graph/eager divergence on {values}"
