"""Host KV tier: bit-exact page round trips, LRU capacity, and fail-fast geometry checks.

CPU only -- the tier is pure host bookkeeping plus device<->host copies, so its correctness is
testable without a GPU. The pool layouts used here mirror ``MHAKVCache._kv_buffer``
(``(2, layers, pages, page_size, kv_heads, head_dim)``) so a strided ``[:, page]`` view is
exercised, not just a contiguous one.
"""
from __future__ import annotations

import pytest
import torch

from freetoken.kvcache.base import HostTierKeyCollision
from freetoken.kvcache.host_tier import HostKVTier, HostTierUnpinned, TierGeometry

LAYERS, PAGE, HEADS, DIM = 3, 4, 2, 8


def geom(**kw) -> TierGeometry:
    base = dict(num_layers=LAYERS, page_size=PAGE, num_kv_heads=HEADS, head_dim=DIM,
                dtype=torch.float16)
    return TierGeometry(**(base | kw))


def pool_buffer(num_pages: int, g: TierGeometry) -> torch.Tensor:
    """The pool's K/V layout: ``(2, layers, pages, page_size, kv_heads, head_dim)``."""
    return torch.empty(2, g.num_layers, num_pages, g.page_size, g.num_kv_heads, g.head_dim,
                       dtype=g.dtype)


def page_views(buf: torch.Tensor, page: int) -> tuple[torch.Tensor, torch.Tensor]:
    return buf[0][:, page], buf[1][:, page]


def test_round_trip_is_bit_exact():
    g = geom()
    tier = HostKVTier(g, 2)
    buf = pool_buffer(2, g)
    k, v = page_views(buf, 0)
    # NaN/Inf payloads: an equality test on floats would pass on the wrong bit pattern
    k.fill_(float("nan"))
    v.fill_(float("inf"))
    tier.spill(0, k, v)

    k.zero_()
    v.zero_()
    assert tier.restore(0, k, v)

    assert torch.equal(k.view(torch.int16), torch.full_like(k, float("nan")).view(torch.int16))
    assert torch.equal(v.view(torch.int16), torch.full_like(v, float("inf")).view(torch.int16))


def test_strided_page_view_round_trips():
    g = geom()
    tier = HostKVTier(g, 2)
    src, dst = pool_buffer(3, g), pool_buffer(3, g)
    torch.manual_seed(0)
    src.normal_()

    k_src, v_src = page_views(src, 1)  # strided: the page is one slice of the page axis
    assert not k_src.is_contiguous()
    tier.spill(1, k_src, v_src)

    k_dst, v_dst = page_views(dst, 2)  # a different page id: the tier addresses by its own key
    assert tier.restore(1, k_dst, v_dst)
    assert torch.equal(k_dst, k_src)
    assert torch.equal(v_dst, v_src)


def test_lru_drops_the_oldest_page_and_reports_it():
    g = geom()
    dropped: list[int] = []
    tier = HostKVTier(g, 2, on_drop=dropped.append)
    buf = pool_buffer(3, g)

    tier.spill(0, *page_views(buf, 0))
    tier.spill(1, *page_views(buf, 1))
    assert tier.restore(0, *page_views(buf, 0))  # page 0 becomes the most recent

    tier.spill(2, *page_views(buf, 2))  # capacity 2 -> the LRU victim is page 1

    assert dropped == [1]
    assert 1 not in tier
    assert tier.resident_pages == 2
    assert tier.stats.dropped == 1
    assert tier.stats.dropped_bytes == tier.bytes_per_page


def test_restore_reports_a_miss_for_an_unknown_page():
    g = geom()
    tier = HostKVTier(g, 1)
    buf = pool_buffer(1, g)
    k, v = page_views(buf, 0)
    k.fill_(3.0)

    assert tier.restore(7, k, v) is False
    assert tier.stats.misses == 1
    assert torch.all(k == 3.0)  # a miss leaves the caller's buffer untouched


def test_drop_frees_the_slot_without_counting_a_drop():
    g = geom()
    tier = HostKVTier(g, 1)
    buf = pool_buffer(2, g)

    tier.spill(0, *page_views(buf, 0))
    assert tier.drop(0) is True
    assert tier.drop(0) is False
    tier.spill(1, *page_views(buf, 1))

    assert tier.resident_pages == 1
    assert tier.stats.dropped == 0
    assert 1 in tier


@pytest.mark.parametrize(
    "bad",
    [
        torch.empty(LAYERS, PAGE + 1, HEADS, DIM, dtype=torch.float16),
        torch.empty(LAYERS, PAGE, HEADS + 1, DIM, dtype=torch.float16),
        torch.empty(LAYERS, PAGE, HEADS, DIM, dtype=torch.bfloat16),
    ],
)
def test_geometry_mismatch_fails_fast(bad: torch.Tensor):
    g = geom()
    tier = HostKVTier(g, 1)
    ok = torch.empty(LAYERS, PAGE, HEADS, DIM, dtype=torch.float16)
    with pytest.raises(AssertionError):
        tier.spill(0, bad, ok)


def test_qsa_index_shadow_and_rope_positions_round_trip():
    g = geom(index_layers=2, index_head_dim=6, index_ratio=2, rope_pos=True)
    tier = HostKVTier(g, 1)
    kv = pool_buffer(1, g)
    index = torch.arange(g.index_page_elems, dtype=g.dtype).view(
        g.index_layers, g.index_rows_per_page, g.index_head_dim
    )
    rope = torch.arange(g.page_size * 3, dtype=torch.int32).view(g.page_size, 3)

    tier.spill(0, *page_views(kv, 0), index_slab=index, rope_slab=rope)
    index_out = torch.zeros_like(index)
    rope_out = torch.zeros_like(rope)
    assert tier.restore(0, *page_views(kv, 0), index_slab=index_out, rope_slab=rope_out)

    assert torch.equal(index_out, index)
    assert torch.equal(rope_out, rope)


def test_bytes_per_page_matches_the_pool_slab_sizes():
    g = geom(index_layers=2, index_head_dim=6, index_ratio=2, rope_pos=True)
    tier = HostKVTier(g, 4)

    assert tier.bytes_per_page == (
        2 * LAYERS * PAGE * HEADS * DIM * 2  # K + V, float16
        + 2 * (PAGE // 2) * 6 * 2  # QSA shadow rows x index layers x 2 bytes
        + PAGE * 3 * 4  # mrope positions, int32
    )
    assert tier.capacity_bytes == 4 * tier.bytes_per_page
    assert tier.resident_bytes == 0


def test_spill_refuses_a_key_that_is_already_resident():
    """A pool page id is not a safe key: the pool recycles it, so a second spill under the
    same key must fail rather than silently hand the first owner the second owner's bytes."""
    g = geom()
    tier = HostKVTier(g, 2)
    buf = pool_buffer(2, g)
    k, v = page_views(buf, 0)
    k.fill_(1.0)
    tier.spill(7, k, v)

    with pytest.raises(HostTierKeyCollision):
        tier.spill(7, *page_views(buf, 1))

    assert torch.equal(k, torch.ones_like(k)), "the resident copy is untouched"
    assert tier.resident_pages == 1
    assert tier.stats.spills == 1


def test_non_blocking_without_pinning_is_refused():
    """pageable + non_blocking silently degrades to a synchronous copy in torch, so the flag
    would be a performance lie rather than an error."""
    g = geom()
    tier = HostKVTier(g, 2)
    buf = pool_buffer(2, g)

    with pytest.raises(HostTierUnpinned):
        tier.spill(0, *page_views(buf, 0), non_blocking=True)
    with pytest.raises(HostTierUnpinned):
        tier.restore(0, *page_views(buf, 0), non_blocking=True)

    assert tier.resident_pages == 0, "the refused spill must not take a slot"
    assert tier.stats.spills == 0


def test_clear_reports_every_resident_key_to_on_drop():
    """clear() is what a pool rebuild calls; a silent clear would leave host-resident nodes
    pointing at a tier that no longer holds their bytes."""
    g = geom()
    dropped: list[int] = []
    tier = HostKVTier(g, 3, on_drop=dropped.append)
    buf = pool_buffer(3, g)
    tier.spill(1, *page_views(buf, 0))
    tier.spill(2, *page_views(buf, 1))

    tier.clear()

    assert sorted(dropped) == [1, 2]
    assert tier.resident_pages == 0
    assert 1 not in tier
    assert not tier.restore(1, *page_views(buf, 0)), "cleared copies are gone, not stale"
