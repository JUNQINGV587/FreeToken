"""Prefill hit-D2D split: resident experts must be gathered device-side, misses
H2D'd via cudaMemcpyBatchAsync, and the buffer must end up byte-identical to the
full-layer copy for every mix -- including experts resident in the volatile
buffer slots (< 2 * num_experts), which must be re-fetched over PCIe."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch

from freetoken.moe import offload_cache
from freetoken.moe.offload_cache import OffloadMoeCache

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
JIT = pytest.mark.skipif(
    os.getenv("FREETOKEN_DISABLE_JIT", "").strip().lower() in {"1", "true", "yes", "on"},
    reason="batch_memcpy has no AOT prebuild; needs runtime JIT",
)


def _cuda_at_least(major: int, minor: int) -> bool:
    cuda = torch.version.cuda
    if cuda is None:
        return False
    return tuple(int(x) for x in cuda.split(".")[:2]) >= (major, minor)

BATCH_API = pytest.mark.skipif(
    not _cuda_at_least(13, 0), reason="the cudaMemcpyBatchAsync binding needs CUDA >= 13.0"
)

NUM_LAYERS, E, CACHE_SIZE = 3, 8, 24  # hit region = slots [16, 24)


def _make_cache() -> tuple[OffloadMoeCache, dict[str, list[torch.Tensor]]]:
    dev = torch.device("cuda")
    sources = {
        "gate_up": [torch.randn(E, 32, 8, dtype=torch.bfloat16).pin_memory() for _ in range(NUM_LAYERS)],
        "down": [torch.randn(E, 8, 16, dtype=torch.bfloat16).pin_memory() for _ in range(NUM_LAYERS)],
    }
    cache = OffloadMoeCache(
        num_layers=NUM_LAYERS,
        num_experts=E,
        cache_size=CACHE_SIZE,
        device=dev,
        prefill_overlap=True,
        prefill_hit_d2d=True,
    )
    cache.set_bank_sources(sources)
    return cache, sources


def _seed_resident(cache, sources, layer_id: int, expert_id: int, slot: int) -> None:
    cache.slot_for_id[layer_id, expert_id] = slot
    cache.id_of_slot[slot] = layer_id * E + expert_id
    for name, per_layer in sources.items():
        cache.bank_caches[name][slot].copy_(per_layer[layer_id][expert_id])


@CUDA
@JIT
@BATCH_API
def test_batch_memcpy_roundtrip():
    from freetoken.kernel.batch_memcpy import batch_memcpy_jit

    rows, feat = 16, 1024
    src = torch.randint(0, 256, (rows, feat), dtype=torch.uint8).pin_memory()
    dst = torch.zeros(rows, feat, dtype=torch.uint8, device="cuda")
    perm = torch.randperm(rows)
    dst_ptrs = torch.tensor([dst[i].data_ptr() for i in range(rows)], dtype=torch.int64)
    src_ptrs = torch.tensor([src[p].data_ptr() for p in perm.tolist()], dtype=torch.int64)
    sizes = torch.full((rows,), feat, dtype=torch.int64)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        batch_memcpy_jit(dst_ptrs, src_ptrs, sizes, stream.cuda_stream)
    stream.synchronize()
    assert torch.equal(dst.cpu(), src[perm])


@CUDA
@JIT
@BATCH_API
def test_prefill_hit_d2d_matches_sources():
    cache, sources = _make_cache()
    # layer 1 residents: two real hits in the hit region, one expert stuck in a
    # buffer slot (< 2E) whose cache bytes are poisoned -- it must come back via
    # H2D, so poison must never surface in the buffer.
    _seed_resident(cache, sources, layer_id=1, expert_id=1, slot=17)
    _seed_resident(cache, sources, layer_id=1, expert_id=3, slot=20)
    _seed_resident(cache, sources, layer_id=1, expert_id=0, slot=2)
    for name in sources:
        cache.bank_caches[name][2].fill_(float("nan"))

    cache.begin_prefill()
    assert cache._prefill_hit_d2d_active
    cache.prefetch_prefill_layer(0)
    cache.prefetch_prefill_layer(1)
    for layer_id in (0, 1):
        views = cache.wait_prefill_layer(layer_id)
        torch.cuda.synchronize()
        for view, (name, per_layer) in zip(views, sources.items()):
            assert torch.equal(view.cpu(), per_layer[layer_id]), (layer_id, name)
        cache.release_prefill_layer(layer_id)
    # prefetch(0) had no hits; prefetch(1) served experts 1 and 3 from the cache.
    assert cache.prefill_hit_rows == 2
    assert cache.prefill_total_rows == 2 * E
    # the buffer-slot resident was invalidated (its slot belongs to buffer 0)...
    assert int(cache.slot_for_id[1, 0].item()) == -1
    # ...while true hits keep their cache residency.
    assert int(cache.slot_for_id[1, 1].item()) == 17
    assert int(cache.slot_for_id[1, 3].item()) == 20


@CUDA
@JIT
@BATCH_API
@pytest.mark.parametrize("nhit", [0, E])
def test_prefill_hit_d2d_pure_extremes(nhit):
    cache, sources = _make_cache()
    for e in range(nhit):
        _seed_resident(cache, sources, layer_id=2, expert_id=e, slot=2 * E + e)
    cache.begin_prefill()
    cache.prefetch_prefill_layer(2)
    views = cache.wait_prefill_layer(2)
    torch.cuda.synchronize()
    for view, (name, per_layer) in zip(views, sources.items()):
        assert torch.equal(view.cpu(), per_layer[2]), name
    assert cache.prefill_hit_rows == nhit


@CUDA
def test_prefill_hit_d2d_noop_without_spare_slots():
    dev = torch.device("cuda")
    sources = {
        "gate_up": [torch.randn(E, 32, 8, dtype=torch.bfloat16).pin_memory() for _ in range(NUM_LAYERS)],
        "down": [torch.randn(E, 8, 16, dtype=torch.bfloat16).pin_memory() for _ in range(NUM_LAYERS)],
    }
    cache = OffloadMoeCache(
        num_layers=NUM_LAYERS,
        num_experts=E,
        cache_size=2 * E,  # the cpu backend's pinned geometry: no hit region
        device=dev,
        prefill_overlap=True,
        prefill_hit_d2d=True,
    )
    cache.set_bank_sources(sources)
    cache.begin_prefill()
    assert not cache._prefill_hit_d2d_active
    cache.prefetch_prefill_layer(0)
    views = cache.wait_prefill_layer(0)
    torch.cuda.synchronize()
    for view, (name, per_layer) in zip(views, sources.items()):
        assert torch.equal(view.cpu(), per_layer[0]), name


def test_small_bank_gather_is_off_by_default():
    # The prototype changes which rows cross PCIe, so it must never be on by accident.
    assert offload_cache._SMALL_BANK_GATHER is False


@CUDA
@JIT
@BATCH_API
def test_small_bank_gather_fills_the_layer_buffer(monkeypatch):
    # With the gather covering the small banks too, a layer must still end up complete:
    # hit rows come from the gather, miss rows from the copy run. Any row the plan drops
    # shows up here as a mismatch against the sources -- the red line for the prototype.
    monkeypatch.setattr(offload_cache, "_SMALL_BANK_GATHER", True)
    cache, sources = _make_cache()
    assert len(cache._gather_bank_ids) == len(sources), "every bank must be gatherable"
    _seed_resident(cache, sources, layer_id=1, expert_id=1, slot=17)
    _seed_resident(cache, sources, layer_id=1, expert_id=3, slot=20)
    _seed_resident(cache, sources, layer_id=1, expert_id=0, slot=2)  # buffer slot -> miss
    for name in sources:
        cache.bank_caches[name][2].fill_(float("nan"))

    cache.begin_prefill()
    cache.prefetch_prefill_layer(0)
    cache.prefetch_prefill_layer(1)
    for layer_id in (0, 1):
        views = cache.wait_prefill_layer(layer_id)
        torch.cuda.synchronize()
        for view, (name, per_layer) in zip(views, sources.items()):
            assert torch.equal(view.cpu(), per_layer[layer_id]), (layer_id, name)
        cache.release_prefill_layer(layer_id)
    assert cache.prefill_hit_rows == 2


def _latch_stub(chunk_open: bool = True) -> OffloadMoeCache:
    """An OffloadMoeCache shell carrying only the overlap-chunk latch state."""
    cache = OffloadMoeCache.__new__(OffloadMoeCache)
    cache.prefill_overlap = True
    cache._prefill_depth = 2
    cache._prefill_chunk_open = chunk_open
    cache._prefill_buffer_layer = [3, None] if chunk_open else [None, None]
    cache._prefill_buffer_released = [False, True] if chunk_open else [True, True]
    return cache


def test_abort_prefill_chunk_clears_stuck_latch():
    # A mid-chunk exception leaves the chunk half-open; abort must reset the latch so the
    # next begin_prefill re-fences the copy stream instead of no-oping behind a stale map.
    cache = _latch_stub()
    cache.abort_prefill_chunk()
    assert cache._prefill_chunk_open is False
    assert cache._prefill_buffer_layer == [None, None]
    assert cache._prefill_buffer_released == [True, True]


def test_abort_prefill_chunk_noop_when_closed_or_overlap_off():
    cache = _latch_stub(chunk_open=False)
    cache.abort_prefill_chunk()
    assert cache._prefill_buffer_layer == [None, None]
    cache.prefill_overlap = False
    cache._prefill_chunk_open = True
    cache.abort_prefill_chunk()
    assert cache._prefill_chunk_open is True


def _layer_stub(cache, gemm):
    return SimpleNamespace(
        owner_cache=None,
        offload_cache=cache,
        layer_id=0,
        num_experts=8,
        _wait_prefill_overlap=lambda c: (),
        _expert_gemm=gemm,
    )


def test_mid_chunk_exception_aborts_the_chunk():
    from freetoken.layers.moe import OffloadMoELayer

    calls: list[str] = []
    cache = SimpleNamespace(
        prefill_overlap=True,
        alphas_for_layer=lambda layer: None,
        release_prefill_layer=lambda layer: calls.append("release"),
        abort_prefill_chunk=lambda: calls.append("abort"),
    )

    def boom(*args, **kwargs):
        raise RuntimeError("simulated GEMM failure")

    with pytest.raises(RuntimeError, match="simulated GEMM failure"):
        OffloadMoELayer._prefill_routed(_layer_stub(cache, boom), None, None, None)
    assert calls == ["abort"], "the chunk must be aborted, never released, on exception"


def test_clean_layer_releases_without_abort():
    from freetoken.layers.moe import OffloadMoELayer

    calls: list[str] = []
    cache = SimpleNamespace(
        prefill_overlap=True,
        alphas_for_layer=lambda layer: None,
        release_prefill_layer=lambda layer: calls.append("release"),
        abort_prefill_chunk=lambda: calls.append("abort"),
    )
    out = OffloadMoELayer._prefill_routed(_layer_stub(cache, lambda *a, **k: "out"), None, None, None)
    assert out == "out"
    assert calls == ["release"]
