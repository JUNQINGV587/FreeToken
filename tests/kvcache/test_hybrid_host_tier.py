"""Host-resident nodes: spilling frees the pages but keeps the prefix matchable.

Driven straight against ``HybridRadixCache``, which is pool-agnostic (it only stores page
indices), so the "host tier" here is two fake callbacks and no GPU or pool is involved.

The accounting identity under test is the one ``CacheManager.check_integrity`` enforces on a
live engine -- every page is either free or counted in ``full_evictable + full_protected`` --
plus the three-state rules: a host-resident node keeps its token span and its snapshot, its
ancestors stay pinned, and nothing may hand its (now empty) ``value`` back to a pool.
"""
from __future__ import annotations

import pytest
import torch

from freetoken.kvcache.base import HostTierKeyCollision
from freetoken.kvcache.hybrid_radix_cache import HybridRadixCache
from freetoken.kvcache.radix_cache import RadixTreeNode

PAGE = 2


def tokens(n: int) -> torch.Tensor:
    return torch.arange(n, dtype=torch.int32)


def pages(n: int) -> torch.Tensor:
    """One page index per token, page-aligned: what ``node.value`` really holds."""
    return (torch.arange(n, dtype=torch.int32) // PAGE) * PAGE


class FakeHostTier:
    """Records what was spilled, and hands out fresh page indices on materialize."""

    def __init__(self) -> None:
        self.store: dict[str, torch.Tensor] = {}
        self.spilled: list[str] = []
        self.materialized: list[str] = []
        self._next_page = 1000

    def spill(self, node) -> str:
        key = f"k{node.uuid}"
        self.store[key] = node.value.clone()
        self.spilled.append(key)
        return key

    def materialize(self, node, key) -> torch.Tensor:
        self.materialized.append(key)
        n_pages = node.length // PAGE
        fresh = torch.arange(self._next_page, self._next_page + n_pages, dtype=torch.int32)
        self._next_page += n_pages
        return fresh.repeat_interleave(PAGE)


def make_cache(with_tier: bool = True):
    tier = FakeHostTier() if with_tier else None
    cache = HybridRadixCache(
        torch.device("cpu"), PAGE,
        host_spill=tier.spill if tier else None,
        host_materialize=tier.materialize if tier else None,
    )
    return cache, tier


def accounts(cache) -> int:
    return cache.full_evictable + cache.full_protected + cache.host_resident_size


def test_spill_hands_the_pages_back_but_keeps_the_prefix_matchable():
    cache, tier = make_cache()
    ids, idx = tokens(8), pages(8)
    cache.insert(ids, idx, mamba_value=3)
    assert cache.full_evictable == 8

    er = cache.evict_full(8)

    assert torch.equal(er.kv_indices, idx), "the caller needs these pages to free them"
    assert er.mamba_slots == [], "the snapshot stays: it is what makes the host copy resumable"
    assert cache.full_evictable == 0 and cache.host_resident_size == 8
    assert len(tier.spilled) == 1
    cache.check_integrity()

    m = cache.match_prefix(ids)

    assert m.cached_len == 8 and m.mamba_value == 3
    assert tier.materialized == tier.spilled
    assert cache.host_resident_size == 0 and cache.full_evictable == 8
    cache.check_integrity()


def test_a_host_resident_child_keeps_its_prefix_resident():
    cache, _ = make_cache()
    cache.insert(tokens(8), pages(8), mamba_value=1)      # parent
    cache.insert(tokens(16), pages(16), mamba_value=2)    # child, the LRU leaf
    parent = next(iter(cache.root.children.values()))
    assert accounts(cache) == 16

    cache.evict_full(8)                                   # spills the child only

    assert cache.host_resident_size == 8 and cache.full_evictable == 8
    assert not parent.is_leaf(), "the spilled child keeps its slot, so the prefix is no leaf"
    assert cache.full_protected == 0, "no ref-count pinning: the tree shape is the protection"
    # an eviction that would take the parent cannot even see it: only leaves are candidates
    assert len(cache.evict_full(8).kv_indices) == 0
    assert torch.equal(parent.value, pages(8)), "the prefix KV is untouched"
    cache.check_integrity()


def test_materialize_restores_the_accounting():
    cache, _ = make_cache()
    cache.insert(tokens(8), pages(8), mamba_value=1)
    cache.insert(tokens(16), pages(16), mamba_value=2)
    cache.evict_full(8)

    m = cache.match_prefix(tokens(16))

    assert m.cached_len == 16 and m.mamba_value == 2
    assert cache.host_resident_size == 0
    assert cache.full_evictable == 16 and cache.full_protected == 0
    assert accounts(cache) == 16
    cache.check_integrity()


def test_host_drop_unlinks_the_node_and_frees_its_prefix_to_evict():
    cache, tier = make_cache()
    cache.insert(tokens(8), pages(8), mamba_value=1)
    cache.insert(tokens(16), pages(16), mamba_value=2)
    cache.evict_full(8)
    key = tier.spilled[-1]
    parent = next(iter(cache.root.children.values()))

    assert cache.host_drop(key) is True
    assert cache.host_drop(key) is False, "a dropped key must not be handled twice"

    assert cache.host_resident_size == 0
    assert cache.full_evictable == 8
    assert parent.is_leaf(), "with the child gone, the prefix is a leaf and evictable again"
    assert len(cache.evict_full(8).kv_indices) == 8
    cache.check_integrity()


def test_a_snapshotless_leaf_is_evicted_not_spilled():
    cache, tier = make_cache()
    ids, idx = tokens(8), pages(8)
    cache.insert(ids, idx, mamba_value=1)
    node = next(iter(cache.root.children.values()))
    node.mamba_value = None          # a tombstoned leaf: nothing can resume from it
    cache.mamba_evictable -= 1

    er = cache.evict_full(8)

    assert tier.spilled == [], "spilling it would write KV that no match can ever read"
    assert torch.equal(er.kv_indices, idx)
    assert cache.host_resident_size == 0 and cache.full_evictable == 0
    assert cache.match_prefix(ids).cached_len == 0


def test_evict_mamba_leaves_a_host_resident_snapshot_alone():
    cache, _ = make_cache()
    ids = tokens(8)
    cache.insert(ids, pages(8), mamba_value=1)
    cache.evict_full(8)

    er = cache.evict_mamba(1)

    assert er.mamba_slots == [], "freeing it would strand the host-resident KV"
    assert cache.host_resident_size == 8
    assert cache.match_prefix(ids).mamba_value == 1


def test_without_a_host_tier_the_cache_behaves_as_before():
    cache, _ = make_cache(with_tier=False)
    ids, idx = tokens(8), pages(8)
    cache.insert(ids, idx, mamba_value=1)

    er = cache.evict_full(8)

    assert torch.equal(er.kv_indices, idx) and er.mamba_slots == [1]
    assert cache.host_resident_size == 0 and cache.full_evictable == 0
    assert cache.match_prefix(ids).cached_len == 0, "no host tier -> the prefix is really gone"
    cache.check_integrity()


@pytest.mark.parametrize("evict", [4, 8, 16])
def test_tokens_are_conserved_through_spill_and_materialize(evict: int):
    cache, _ = make_cache()
    ids = tokens(16)
    cache.insert(tokens(8), pages(8), mamba_value=1)
    cache.insert(ids, pages(16), mamba_value=2)

    cache.evict_full(evict)
    cache.check_integrity()
    assert accounts(cache) == 16

    cache.match_prefix(ids)
    cache.check_integrity()
    assert accounts(cache) == 16


def test_spill_refuses_a_node_that_is_not_page_aligned():
    """Unreachable through insert(), which aligns every node boundary to page_size; the guard is
    here so a future path that breaks that invariant fails loudly instead of writing a partial
    page into the host tier."""
    cache, _ = make_cache()
    node = RadixTreeNode(cache.key_fn)
    node.set_key_value(tokens(3), pages(3))
    node.mamba_value = 1

    with pytest.raises(ValueError, match="not a multiple of page_size"):
        cache._spill(node, [])

    assert cache.host_resident_size == 0


def test_spill_refuses_a_tier_key_that_already_names_another_node():
    """Two nodes under one tier key would make whichever materializes second read the other
    prefix's bytes, so the second spill must fail before the pages leave."""
    cache, _ = make_cache()
    cache.host_spill = lambda node: "shared"      # a key space that is not per node
    cache.insert(tokens(8), pages(8), mamba_value=1)
    cache.insert(tokens(8) + 100, pages(8) + 100, mamba_value=2)  # a sibling leaf

    cache.evict_full(8)                           # eviction takes whole leaves, so this is one
    assert cache.host_resident_size == 8
    assert accounts(cache) == 16

    with pytest.raises(HostTierKeyCollision):
        cache.evict_full(8)                       # the other sibling would reuse the key

    assert cache.host_resident_size == 8, "the refused spill must move no accounting"
    assert accounts(cache) == 16
    cache.check_integrity()


def test_node_value_holds_pool_slot_ids_not_page_numbers():
    """``node.value`` carries one pool SLOT id per token, so a page index is ``slot // PAGE``.

    A slot id is ``page * page_size + offset`` (CacheManager.free_slots is built that way), which
    makes ``value[::page_size]`` the page's FIRST SLOT, not its number. Reading a slot id as a
    page number silently addresses another page -- or runs off the end of the pool -- and the pool
    recycles ids, so the mistake stays quiet until the wrong KV is read.
    """
    cache, _ = make_cache(with_tier=False)
    cache.insert(tokens(8), pages(8) + 2 * PAGE, mamba_value=3)
    node = cache.root
    while node.children:
        node = next(iter(node.children.values()))
    assert node.length == 8
    # Stored as slot ids: page 2's first slot is 4, page 5's is 10.
    assert [int(v) for v in node.value[::PAGE]] == [4, 6, 8, 10]
    assert int(node.value[0]) != 2, "value[0] is a slot id, never a page number"
    # The conversion every consumer needs (the read side does the same: fa.py, qsa_sparse.py).
    assert [int(v) // PAGE for v in node.value[::PAGE]] == [2, 3, 4, 5]
