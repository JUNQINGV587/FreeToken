"""Pool <-> host tier bridge: slot-id addressing, the QSA shadow, and bit-exact round trips.

CPU only. The bridge is pure bookkeeping over the pool's own buffers, so its correctness is
testable without a GPU; the fake pool mirrors ``QSAKVCache``'s layout -- a strided ``_kv_buffer``
plus the compressed index slab and the per-token rope positions that a restore has to remap.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.kvcache.host_tier import HostKVTier, TierGeometry, TierGeometryMismatch
from freetoken.kvcache.host_tier_bridge import (
    HOST_TIER_PAGES_ENV,
    PoolHostBridge,
    host_tier_pages_from_env,
    maybe_build_bridge,
)

LAYERS, PAGE, HEADS, DIM = 2, 4, 1, 6
INDEX_DIM, RATIO, PAGES = 5, 2, 8
ROWS_PER_PAGE = PAGE // RATIO


class FakeQSAPool:
    """``QSAKVCache``'s slab layouts, with the same accessors the bridge uses."""

    def __init__(self, *, index_layers: int = LAYERS, mrope: bool = True) -> None:
        self._kv_buffer = torch.zeros(2, LAYERS, PAGES, PAGE, HEADS, DIM, dtype=torch.float16)
        self._k_buffer = self._kv_buffer[0]
        self._v_buffer = self._kv_buffer[1]
        self._num_index_layers = index_layers
        self._index_head_dim = INDEX_DIM
        self._index_ratio = RATIO
        self._mrope = mrope
        rows = PAGES * ROWS_PER_PAGE
        self._cmp_k_buffer = torch.zeros(index_layers, rows + 2, INDEX_DIM, dtype=torch.float16)
        self._pending_ring = torch.zeros(2, index_layers, RATIO, INDEX_DIM, dtype=torch.float16)
        self._rope_positions = torch.zeros(PAGES * PAGE, 3, dtype=torch.int32)

    @property
    def num_storage_layers(self) -> int:
        return int(self._k_buffer.shape[0])

    @property
    def rope_positions(self) -> torch.Tensor:
        return self._rope_positions

    @property
    def cmp_scratch_base(self) -> int:
        return PAGES * ROWS_PER_PAGE

    def cmp_k_cache(self, slot: int) -> torch.Tensor:
        return self._cmp_k_buffer[slot]

    def pending_ring(self, slot: int) -> torch.Tensor:
        return self._pending_ring[:, slot]

    def page_kv_view(self, page_index: int) -> torch.Tensor:
        return self._kv_buffer[:, :, page_index]


def geom(**kw) -> TierGeometry:
    base = dict(num_layers=LAYERS, page_size=PAGE, num_kv_heads=HEADS, head_dim=DIM,
                dtype=torch.float16, index_layers=LAYERS, index_head_dim=INDEX_DIM,
                index_ratio=RATIO, rope_pos=True)
    return TierGeometry(**(base | kw))


class Node:
    """A radix node's span: ``value`` is one pool slot id per token, page-aligned."""

    def __init__(self, uuid: str, length: int, first_page: int) -> None:
        self.uuid = uuid
        self.length = length
        self.value = (torch.arange(length, dtype=torch.int32) // PAGE + first_page) * PAGE


class Alloc:
    """Hands out page indices, in order, and records what was asked for."""

    def __init__(self, *pages: int) -> None:
        self.pages = list(pages)
        self.calls: list[int] = []

    def __call__(self, n: int) -> torch.Tensor:
        self.calls.append(n)
        if len(self.pages) < n:
            return torch.empty(0, dtype=torch.int32)
        out, self.pages = self.pages[:n], self.pages[n:]
        return torch.tensor(out, dtype=torch.int32)


def fill_page(pool: FakeQSAPool, page: int, seed: int) -> None:
    """Distinct, exactly representable payload per page, in all three slabs."""
    kv = pool.page_kv_view(page)
    kv[0].fill_(seed)
    kv[1].fill_(-seed)
    for layer in range(pool._num_index_layers):
        r0 = page * ROWS_PER_PAGE
        pool.cmp_k_cache(layer)[r0:r0 + ROWS_PER_PAGE].fill_(seed * 10 + layer)
    pool.rope_positions[page * PAGE:(page + 1) * PAGE] = torch.tensor(
        [[seed, page, layer] for layer in range(PAGE)], dtype=torch.int32
    )


def snapshot(pool: FakeQSAPool, page: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    kv = pool.page_kv_view(page).clone()
    idx = torch.stack([pool.cmp_k_cache(l)[page * ROWS_PER_PAGE:(page + 1) * ROWS_PER_PAGE]
                       for l in range(pool._num_index_layers)]).clone()
    rope = pool.rope_positions[page * PAGE:(page + 1) * PAGE].clone()
    return kv, idx, rope


def bridge(pool: FakeQSAPool, *, capacity: int = 4, alloc: Alloc | None = None):
    tier = HostKVTier(geom(), capacity)
    return PoolHostBridge(pool, tier, PAGE, alloc_pages=alloc or Alloc()), tier


def test_round_trip_into_fresh_pages_is_bit_exact_in_all_three_slabs():
    pool = FakeQSAPool()
    for page in range(PAGES):
        fill_page(pool, page, seed=page + 1)
    original = {page: snapshot(pool, page) for page in range(PAGES)}
    node = Node("n1", 3 * PAGE, first_page=1)          # pages 1, 2, 3
    br, tier = bridge(pool, alloc=Alloc(5, 6, 7))

    key = br.spill(node)

    assert key is not None and tier.entry_slots(key) is not None
    assert len(tier.entry_slots(key)) == 3
    for page in (1, 2, 3):                              # the caller frees them afterwards
        pool.page_kv_view(page).zero_()
        for layer in range(LAYERS):
            pool.cmp_k_cache(layer)[page * ROWS_PER_PAGE:(page + 1) * ROWS_PER_PAGE].zero_()
        pool.rope_positions[page * PAGE:(page + 1) * PAGE].zero_()

    value = br.materialize(node, key)

    assert value is not None
    assert [int(v) for v in value] == [5 * PAGE] * PAGE + [6 * PAGE] * PAGE + [7 * PAGE] * PAGE
    assert tier.entry_slots(key) is None, "the entry is released once the KV is back on the device"
    for fresh, src in zip((5, 6, 7), (1, 2, 3)):
        got = snapshot(pool, fresh)
        for a, b in zip(got, original[src]):
            assert torch.equal(a, b), "NaN/Inf-free equality would hide a bit-pattern slip"


def test_a_slot_id_is_never_read_as_a_page_number():
    """Page 3's first slot is 12; read as a page number that is off the end of an 8-page pool."""
    pool = FakeQSAPool()
    for page in range(PAGES):
        fill_page(pool, page, seed=page + 7)
    node = Node("n2", PAGE, first_page=3)
    br, tier = bridge(pool)

    key = br.spill(node)

    assert key is not None
    tier_k = tier.k_page(tier.entry_slots(key)[0])
    assert torch.equal(tier_k, pool.page_kv_view(0)[0].new_full(tier_k.shape, 10)), (
        "spilled page 3 (seed 10); page 12 does not exist and would have raised"
    )


def test_the_index_shadow_and_rope_land_on_the_fresh_page_rows():
    pool = FakeQSAPool()
    fill_page(pool, 0, seed=4)
    node = Node("n3", PAGE, first_page=0)
    br, _ = bridge(pool, alloc=Alloc(6))

    key = br.spill(node)
    br.materialize(node, key)

    assert torch.equal(pool.cmp_k_cache(0)[6 * ROWS_PER_PAGE:(6 + 1) * ROWS_PER_PAGE],
                       pool.cmp_k_cache(0)[0:ROWS_PER_PAGE])
    assert torch.equal(pool.rope_positions[6 * PAGE:(6 + 1) * PAGE],
                       pool.rope_positions[0:PAGE])
    # An unrelated page's rows must stay as they were: only the fresh page is written.
    assert pool.cmp_k_cache(0)[1 * ROWS_PER_PAGE:2 * ROWS_PER_PAGE].count_nonzero() == 0
    assert pool.rope_positions[1 * PAGE:2 * PAGE].count_nonzero() == 0


def test_spill_refuses_a_span_that_is_not_whole_pages():
    pool = FakeQSAPool()
    br, tier = bridge(pool)
    node = Node("n4", PAGE + 1, first_page=0)
    node.value = torch.arange(PAGE + 1, dtype=torch.int32)   # last token in a second page

    assert br.spill(node) is None
    assert tier.resident_entries == 0


def test_a_second_spill_under_the_same_key_is_refused():
    pool = FakeQSAPool()
    br, _ = bridge(pool)
    node = Node("n5", PAGE, first_page=0)

    assert br.spill(node) is not None
    assert br.spill(node) is None, "the pool recycles page ids; an overwrite would alias bytes"


def test_materialize_gives_up_when_the_tier_has_dropped_the_entry():
    pool = FakeQSAPool()
    fill_page(pool, 0, seed=3)
    before = snapshot(pool, 0)
    br, tier = bridge(pool, alloc=Alloc(7))
    node = Node("n6", PAGE, first_page=0)
    key = br.spill(node)
    tier.drop(key)

    assert br.materialize(node, key) is None
    for a, b in zip(snapshot(pool, 0), before):
        assert torch.equal(a, b), "a cold prefix leaves the pool untouched"


def test_materialize_refuses_a_short_allocation():
    pool = FakeQSAPool()
    br, tier = bridge(pool, alloc=Alloc())      # no pages left
    node = Node("n7", 2 * PAGE, first_page=0)
    key = br.spill(node)

    assert br.materialize(node, key) is None
    assert tier.entry_slots(key) is None, "the entry is dropped, not left half-restorable"


def test_a_pool_without_the_rope_bank_is_refused_at_construction():
    pool = FakeQSAPool(mrope=False)

    with pytest.raises(TierGeometryMismatch):
        PoolHostBridge(pool, HostKVTier(geom(), 2), PAGE, alloc_pages=Alloc())


def test_an_index_layers_mismatch_is_refused_at_construction():
    pool = FakeQSAPool(index_layers=LAYERS - 1)
    pool._cmp_k_buffer = pool._cmp_k_buffer[:LAYERS - 1]

    with pytest.raises(TierGeometryMismatch):
        PoolHostBridge(pool, HostKVTier(geom(), 2), PAGE, alloc_pages=Alloc())


def test_forget_releases_the_entry():
    pool = FakeQSAPool()
    br, tier = bridge(pool)
    node = Node("n9", PAGE, first_page=0)
    key = br.spill(node)

    assert br.forget(key) is True
    assert br.forget(key) is False
    assert tier.resident_entries == 0


# ---------------------------------------------------------------- opt-in wiring

def test_the_tier_is_off_unless_the_deployment_asks_for_it():
    pool = FakeQSAPool()

    assert host_tier_pages_from_env({}) == 0
    assert host_tier_pages_from_env({HOST_TIER_PAGES_ENV: ""}) == 0
    assert host_tier_pages_from_env({HOST_TIER_PAGES_ENV: "0"}) == 0
    assert host_tier_pages_from_env({HOST_TIER_PAGES_ENV: "-4"}) == 0
    assert host_tier_pages_from_env({HOST_TIER_PAGES_ENV: "eight"}) == 0
    assert host_tier_pages_from_env({HOST_TIER_PAGES_ENV: " 12 "}) == 12
    assert maybe_build_bridge(pool, PAGE, alloc_pages=Alloc(), env={}) is None
    assert maybe_build_bridge(pool, PAGE, alloc_pages=Alloc(),
                              env={HOST_TIER_PAGES_ENV: "0"}) is None


def test_asking_for_it_builds_a_tier_sized_by_the_variable():
    pool = FakeQSAPool()

    built = maybe_build_bridge(pool, PAGE, alloc_pages=Alloc(),
                               env={HOST_TIER_PAGES_ENV: "6"})

    assert built is not None
    tier, br = built
    assert tier.num_pages == 6
    assert tier.geometry.index_layers == LAYERS and tier.geometry.rope_pos is True
    assert br.num_pages == PAGES
    node = Node("w1", PAGE, first_page=2)
    assert br.spill(node) is not None


def test_a_pool_the_geometry_cannot_describe_disables_the_tier_instead_of_raising():
    pool = FakeQSAPool(mrope=False)      # geometry_from_pool would say rope_pos=False...
    pool._rope_positions = None          # ...and its rope slab is gone anyway

    built = maybe_build_bridge(pool, PAGE, alloc_pages=Alloc(),
                               env={HOST_TIER_PAGES_ENV: "4"})

    assert built is not None, "a pool with no rope bank is still carriable"
    assert built[0].geometry.rope_pos is (pool._mrope is True)
    assert maybe_build_bridge(object(), PAGE, alloc_pages=Alloc(),
                              env={HOST_TIER_PAGES_ENV: "4"}) is None


def test_the_manager_allocator_hands_out_page_indices_and_never_evicts():
    from freetoken.scheduler.cache import CacheManager

    fake = SimpleNamespace(
        free_slots=torch.tensor([0, PAGE, 2 * PAGE, 3 * PAGE], dtype=torch.int32),
        page_size=PAGE, device=torch.device("cpu"),
    )

    got = CacheManager._host_alloc_pages(fake, 2)

    assert [int(v) for v in got] == [0, 1], "page indices, not slot ids"
    assert [int(v) for v in fake.free_slots] == [2 * PAGE, 3 * PAGE]
    short = CacheManager._host_alloc_pages(fake, 3)
    assert short.numel() == 0, "a short free list is a refusal, not an eviction"
    assert [int(v) for v in fake.free_slots] == [2 * PAGE, 3 * PAGE], "nothing consumed"
