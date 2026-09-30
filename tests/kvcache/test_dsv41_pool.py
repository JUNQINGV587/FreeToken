"""DSV4.1 paged KV pool + cost model (CPU; no checkpoint, no GPU).

Sections:
  * band ownership: one compressed / indexer / state tier per source layer, shared
    between the owner and every consumer of its band (by identity);
  * band-routed writes: ``store_compressed`` / ``store_indexer`` / ``set_state`` reached
    through a consumer land on the owner's row;
  * ``cmp_rows`` arithmetic for ratio 1 and 2, including a decode-shaped partial group;
  * the window ring: frontend slots, wrap at P, and the derived ``state_loc`` ring map;
  * the swa_pool plug-in surface the generic CacheManager drives;
  * sizing: per-tier slots, exact bytes, the classmethod cost/solve surface;
  * ``resolve_pool_class`` dispatch (DSV4.1 vs DSV4).
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from freetoken.attention import AttnType
from freetoken.kvcache import create_kv_pool, resolve_pool_class
from freetoken.kvcache.base import CacheRebuildRejected
from freetoken.kvcache.dsv4_paged_pool import DSV4PagedKVCache
from freetoken.kvcache.dsv41_cost_model import (
    _dsv41_pool_sizes,
    dsv41_auto_cost_model,
    dsv41_pool_bytes,
    dsv41_pool_sizes,
    dsv41_solve_num_pages,
)
from freetoken.kvcache.dsv41_paged_pool import DSV41PagedKVCache
from freetoken.models.config import KVCacheGroupSpec
from freetoken.models.deepseek_v41.args import DeepseekV41Args

DEVICE = torch.device("cpu")
P = 128
N_LAYERS = 40
KV_SOURCES = (2, 8, 14, 20)
# 24/28/32/36 are index sources but not KV sources: each has an indexer but no `wk`, so it
# owns no key tier (it reads the band KV source's cache) while still publishing the topk
# list its consumers read.
IDX_SOURCES = (2, 8, 14, 20, 24, 28, 32, 36)
# 40 backbone ratios; the checkpoint's 3 MTP entries sit past [:n_layers] (out of scope).
RATIOS = (0, 0) + (2,) * 18 + (1,) * 20


def _args(**over):
    base = dict(
        n_layers=N_LAYERS,
        compress_ratios=RATIOS,
        kv_source_layers=KV_SOURCES,
        index_source_layers=IDX_SOURCES,
        max_seq_len=1024,
        head_dim=512,
        index_head_dim=128,
        window_size=128,
    )
    base.update(over)
    return DeepseekV41Args(**base)


def _pool(num_pages=8, swa_ratio=0.5, n_scratch=1, **over):
    args = _args(**over)
    sizes = dsv41_pool_sizes(num_pages=num_pages, args=args, swa_ratio=swa_ratio, P=P)
    pool = DSV41PagedKVCache(
        sizes=sizes, args=args, device=DEVICE, dtype=torch.bfloat16, P=P, n_scratch=n_scratch
    )
    return pool, sizes, args


def _paged_pool(num_pages=8, swa_ratio=0.5, **over):
    pool, sizes, args = _pool(num_pages=num_pages, swa_ratio=swa_ratio, **over)
    pool._init_paged_state(max_running_req=2, radix=True)
    return pool, sizes, args


def _config(args=None, **over):
    cfg = dict(
        page_size=P,
        max_running_req=2,
        max_seq_len=1024,
        cache_type="swa_radix",
        swa_full_tokens_ratio=0.5,
        num_page_override=None,
        swa_num_pages_override=None,
        memory_ratio=0.9,
    )
    cfg.update(over)
    model_config = SimpleNamespace(dsv41_args=args or _args())
    return SimpleNamespace(model_config=model_config, **cfg)


def _expand(bases):
    return (torch.tensor(bases).view(-1, 1) + torch.arange(P)).flatten()


def _vec(x):
    """A bf16-exact constant row (powers of two survive the bf16 round-trip)."""
    return torch.full((1, 512), float(x))


# ----- band ownership ---------------------------------------------------------


def test_tiers_are_per_source_and_shared_by_identity():
    pool, sizes, args = _pool(n_scratch=2)
    assert pool.num_layers == N_LAYERS
    assert pool.compress_ratios == RATIOS

    # Window: one private ring per layer (all 40).
    ptrs = {pool.window_pool[L].data_ptr() for L in range(N_LAYERS)}
    assert len(ptrs) == N_LAYERS
    for L in range(N_LAYERS):
        assert pool.window_pool[L].shape == (sizes.n_win_slots, args.head_dim)
        assert pool.window_pool[L].dtype == torch.bfloat16

    # Compressed / state: one allocation per KV source, shared across its band.
    for src in KV_SOURCES:
        assert pool.cmp_pool[src].shape == (sizes.cmp_blocks[src] + 2, args.head_dim)
        assert pool.cmp_scratch_base_of(src) == sizes.cmp_blocks[src]
        assert pool.ring_size(src) == sizes.ring_sizes[src]
    for L in range(N_LAYERS):
        src = pool.cmp_source_of(L)
        if src is None:
            continue
        assert pool.cmp_pool[L] is pool.cmp_pool[src]
        assert pool.state_ring[L] is pool.state_ring[src]
        assert pool.cmp_scratch_base_of(L) == sizes.cmp_blocks[src]
        assert pool.cmp_ratio_of(L) == args.layer_ratio(src)

    # The bands: 3-7 <- 2, 9-13 <- 8, 15-19 <- 14, 21-39 <- 20.
    assert [pool.cmp_source_of(L) for L in range(2, N_LAYERS)] == (
        [2] * 6 + [8] * 6 + [14] * 6 + [20] * 20
    )
    assert pool.cmp_source_of(0) is None and pool.cmp_source_of(1) is None

    # Index bands: 3-7 <- 2, 9-13 <- 8, 15-19 <- 14, 21-23 <- 20, 25-27 <- 24,
    # 29-31 <- 28, 33-35 <- 32, 37-39 <- 36.
    assert [pool.idx_source_of(L) for L in range(2, N_LAYERS)] == (
        [2] * 6 + [8] * 6 + [14] * 6 + [20] * 4 + [24] * 4 + [28] * 4 + [32] * 4 + [36] * 4
    )
    assert pool.idx_source_of(0) is None and pool.idx_source_of(1) is None
    assert pool.idx_source_of(24) == 24 and pool.idx_source_of(25) == 24
    assert pool.idx_source_of(23) == 20 and pool.idx_source_of(36) == 36


def test_ratio_zero_layers_own_nothing():
    pool, _, _ = _pool()
    for L in (0, 1):
        assert pool.cmp_pool[L] is None and pool.idx_pool[L] is None
        assert pool.state_ring[L] is None and pool.indexer_state_ring[L] is None
        assert pool.cmp_scratch_base_of(L) is None and pool.idx_scratch_base_of(L) is None
        assert pool.cmp_ratio_of(L) == 0
        assert pool.cmp_source_of(L) is None and pool.idx_source_of(L) is None
        with pytest.raises(AssertionError):
            pool.store_compressed(torch.zeros(1, 512), L, torch.tensor([0]))
        with pytest.raises(AssertionError):
            pool.store_indexer(torch.zeros(1, 128), L, torch.tensor([0]))
        with pytest.raises(AssertionError):
            pool.ring_size(L)


def test_index_only_sources_share_the_kv_sources_indexer_tier():
    """24/28/32/36 have an indexer but no ``wk``, so they own no key tier: they read the KV
    source's cache (vLLM: "Non-owning index sources share the kv source's paged K cache").
    They stay index *sources* on the publication axis -- each still publishes the topk list
    its consumers read, which is what ``pool.idx_source_of`` reports."""
    pool, sizes, args = _pool(n_scratch=2)
    for i in (24, 28, 32, 36):
        src = pool.cmp_source_of(i)
        # No tier of their own -- not compressed, not state, not index keys.
        assert sizes.cmp_blocks[i] is None and sizes.state_slots[i] is None
        assert sizes.idx_blocks[i] is None
        # Their indexer reads the band KV source's key cache, at the band's compressed rows.
        assert pool.idx_pool[i] is pool.idx_pool[src]
        assert pool.idx_scratch_base_of(i) == pool.idx_scratch_base_of(src)
        assert pool.idx_scratch_base_of(src) == sizes.cmp_blocks[src]
        assert pool.cmp_ratio_of(i) == 1
        assert pool.cmp_pool[i] is pool.cmp_pool[src]
    # The publication axis is untouched: 24/28/32/36 still publish for the layers below them.
    assert pool.idx_source_of(23) == 20 and pool.idx_source_of(25) == 24
    assert pool.idx_source_of(29) == 28 and pool.idx_source_of(39) == 36
    # ... while the key cache is one per band, so 21-39 has exactly one.
    assert pool.idx_pool[21] is pool.idx_pool[20] is pool.idx_pool[39]
    assert pool.idx_pool[2] is not pool.idx_pool[20]
    # V4.1's Indexer has no compressor of its own -> no indexer state ring anywhere.
    assert all(r is None for r in pool.indexer_state_ring)
    for L in range(N_LAYERS):
        assert pool.indexer_state_ring_of(L) is None


def test_state_ring_geometry_is_ratio_slots_per_page_without_overlap():
    pool, sizes, args = _pool()
    assert pool.state_ring[2].ring_size == 2 and pool.state_ring[20].ring_size == 1
    # coff == 1: one head_dim of kv and one of score, no overlap half.
    assert pool.state_ring[2].item_size == args.head_dim
    assert pool.state_ring[2].buffer.shape == (sizes.state_slots[2] + 1, 2 * args.head_dim)
    assert pool.state_ring[20].buffer.shape == (sizes.state_slots[20] + 1, 2 * args.head_dim)
    # Consumers report their band's ring.
    assert pool.ring_size(3) == 2 and pool.ring_size(25) == 1 and pool.ring_size(39) == 1


# ----- band-routed writes -----------------------------------------------------


def test_store_compressed_routes_a_consumer_to_the_source_row():
    pool, _, _ = _pool()
    kv = torch.randn(1, 512)
    pool.store_compressed(kv, 25, torch.tensor([7]))  # 25 reads band 20
    assert torch.equal(pool.cmp_pool[25][7:8], kv.to(torch.bfloat16))
    assert torch.equal(pool.cmp_pool[20][7:8], kv.to(torch.bfloat16))
    assert pool.cmp_pool[25] is pool.cmp_pool[20]

    kv2 = torch.randn(1, 512)
    pool.store_compressed(kv2, 20, torch.tensor([8]))  # a source write, read by the band
    assert torch.equal(pool.cmp_pool[39][8:9], kv2.to(torch.bfloat16))
    assert pool.cmp_scratch_base_of(25) == pool.cmp_scratch_base_of(20)

    # Bands are disjoint allocations.
    pool.store_compressed(torch.ones(1, 512), 3, torch.tensor([0]))
    assert torch.equal(pool.cmp_pool[2][0:1], torch.ones(1, 512, dtype=torch.bfloat16))
    assert pool.cmp_pool[2] is not pool.cmp_pool[20]
    assert float(pool.cmp_pool[20][0].abs().sum()) == 0.0


def test_store_indexer_routes_a_consumer_to_the_band_kv_source_row():
    pool, sizes, _ = _pool()
    k = torch.randn(1, 128)
    pool.store_indexer(k, 25, torch.tensor([5]))  # 25 belongs to band 20
    assert torch.equal(pool.idx_pool[20][5:6], k.to(torch.bfloat16))
    assert pool.idx_pool[25] is pool.idx_pool[20]

    pool.store_indexer(k, 39, torch.tensor([6]))  # the whole 21-39 band shares 20's cache
    assert torch.equal(pool.idx_pool[20][6:7], k.to(torch.bfloat16))
    # ... including the index-only sources in the band, which read rather than write it.
    assert torch.equal(pool.idx_pool[36][6:7], k.to(torch.bfloat16))
    assert pool.idx_scratch_base_of(36) == pool.idx_scratch_base_of(20)

    # A KV source's indexer tier is keyed on the same ratio as its compressed rows.
    assert sizes.idx_blocks[2] == sizes.cmp_blocks[2]
    # Bands are disjoint allocations.
    pool.store_indexer(torch.ones(1, 128), 3, torch.tensor([0]))
    assert float(pool.idx_pool[20][0].abs().sum()) == 0.0


def test_set_state_through_a_consumer_reaches_the_source_ring():
    pool, _, _ = _pool()
    loc = pool.state_loc(torch.tensor([0, 3 * P + 5, -1]), pool.ring_size(25), P)
    state = torch.randn(loc.numel(), 1024)
    pool.set_state(25, loc, state)
    # Rows 0/1 are real ring slots; the -1 row is the scratch slot, cleared by every set().
    assert torch.equal(pool.get_state(20, loc[:2]), state[:2])
    assert torch.equal(pool.get_state(39, loc[:2]), state[:2])
    assert pool.state_ring[25] is pool.state_ring[20]

    # The scratch row is re-cleared by every set() and is not a real slot.
    scratch = pool.state_ring[20].buffer[-1]
    assert float(scratch[:512].abs().sum()) == 0.0
    assert bool(torch.isinf(scratch[512:]).all()) and bool((scratch[512:] < 0).all())
    assert torch.equal(pool.get_state(20, torch.tensor([-1]))[0], scratch)


# ----- cmp_rows arithmetic ----------------------------------------------------


def test_cmp_rows_is_floor_div_by_the_band_ratio():
    full = torch.tensor([0, 1, 2, 3, 127, 128, 255, 256])
    assert DSV41PagedKVCache.cmp_rows(full, 1).tolist() == [0, 1, 2, 3, 127, 128, 255, 256]
    assert DSV41PagedKVCache.cmp_rows(full, 2).tolist() == [0, 0, 1, 1, 63, 64, 127, 128]
    # Pure arithmetic: a -1 sentinel stays negative (gather-only, never a scatter target).
    assert int(DSV41PagedKVCache.cmp_rows(torch.tensor([-1]), 2)) == -1


def test_ratio_divides_the_window_page_so_no_group_straddles_pages():
    page = torch.arange(0, P)
    for ratio in (1, 2):
        assert P % ratio == 0
        rows = DSV41PagedKVCache.cmp_rows(page, ratio)
        assert torch.equal(rows, torch.div(page, ratio, rounding_mode="floor"))
        # The row only changes at in-page group boundaries, so a page-boundary gather
        # never has to stitch two compressed rows.
        assert rows[0].item() == 0 and rows[-1].item() == (P - 1) // ratio


def test_decode_partial_group_rows_are_masked_to_a_scratch_row():
    # One scratch row per decode row (the engine passes max_running_req + 1).
    pool, sizes, _ = _pool(n_scratch=4)
    ratio = pool.cmp_ratio_of(3)  # band of source 2
    assert ratio == 2
    base = pool.cmp_scratch_base_of(3)
    assert base == sizes.cmp_blocks[2]

    # A decode step writes a compressed row only when its group completes
    # (``(start_pos + 1) % ratio == 0``); every other row is routed to its own scratch slot
    # (a discarded write, so the masked scatter stays graph-safe).
    pos = torch.tensor([299, 300, 301, 302])
    rows = DSV41PagedKVCache.cmp_rows(pos, ratio)
    completed = (pos + 1) % ratio == 0
    assert completed.tolist() == [True, False, True, False]
    routed = torch.where(completed, rows, base + torch.arange(pos.numel()))
    assert routed.tolist() == [149, base + 1, 150, base + 3]
    assert int(routed.min()) >= 0
    assert int(routed.max()) < base + pool.n_scratch


# ----- window ring ------------------------------------------------------------


def test_window_store_writes_the_frontend_slot_and_wraps_at_p():
    pool, _, _ = _pool()
    L = 5
    for pos in (0, 1, 127, 128, 129, 255, 256):
        pool.store_window(_vec(2 ** (pos % 7)), L, torch.tensor([pos % P]))
    # Wrap: 128 -> slot 0 and 256 -> slot 0 again, 255 -> slot 127; the newest window wins.
    assert float(pool.window_pool[L][0, 0]) == 16.0  # pos 256 (2**4)
    assert float(pool.window_pool[L][1, 0]) == 8.0  # pos 129 (2**3)
    assert float(pool.window_pool[L][127, 0]) == 8.0  # pos 255 (2**3) overwrote pos 127
    assert float(pool.window_pool[L][2:126].abs().sum()) == 0.0
    assert float(pool.window_pool[L][P:].abs().sum()) == 0.0
    assert float(pool.window_pool[4].abs().sum()) == 0.0  # a layer's ring is private


def test_state_loc_is_a_page_relative_ring_index():
    ws = torch.tensor([0, 1, P - 1, P, P + 1, 3 * P + 5, -1])
    assert DSV41PagedKVCache.state_loc(ws, 1, P).tolist() == [0, 0, 0, 1, 1, 3, -1]
    loc = DSV41PagedKVCache.state_loc(torch.tensor([0, 1, 2, 3, P, P + 1]), 2, P)
    # ring_size | P: distinct pages land in disjoint ring blocks, and an unwritten slot
    # (-1) addresses the permanent scratch row instead of a real page.
    assert loc.tolist() == [0, 1, 0, 1, 2, 3]
    assert set(loc[:4].tolist()).isdisjoint(set(loc[4:].tolist()))


# ----- swa plug-in (the CacheManager surface) ---------------------------------


def test_swa_iface_alloc_binds_pages_and_preserves_in_page_offsets():
    pool, sizes, _ = _paged_pool()
    cap = sizes.n_win_slots - P
    assert pool.swa_paged is True
    assert pool.needs_rebind_on_rebuild is True
    assert pool.sliding_window_size == P
    assert pool.swa_available_size() == cap
    assert pool.swa_num_tokens - 1 == cap
    assert isinstance(pool.prefill_chunk_budget, int) and pool.prefill_chunk_budget >= P

    full = _expand([0, 2 * P]).to(torch.int32)  # the manager speaks int32 page tables
    pool.alloc_swa(full)
    assert pool.swa_available_size() == cap - 2 * P
    ws = pool.translate_loc_from_full_to_swa(full)
    assert int(ws.min()) >= 0 and int(ws.max()) < cap
    assert int(ws[0]) % P == 0
    assert torch.equal(ws - ws[0], torch.arange(2 * P))


def test_swa_iface_free_is_page_atomic_and_idempotent():
    pool, sizes, _ = _paged_pool()
    cap = sizes.n_win_slots - P
    full = _expand([0])
    pool.alloc_swa(full)
    assert pool.swa_available_size() == cap - P
    pool.free_swa(full)
    assert pool.swa_available_size() == cap
    assert int(pool.translate_loc_from_full_to_swa(full).max()) == -1
    pool.free_swa(full)  # already unbound -> a no-op, not a double free
    assert pool.swa_available_size() == cap
    with pytest.raises(AssertionError):
        pool.free_swa(torch.arange(0, P // 2))


def test_swa_iface_exhaustion_raises_and_the_dummy_stays_bound():
    pool, sizes, _ = _paged_pool()
    dummy = torch.arange(sizes.full_token - P, sizes.full_token)
    assert torch.equal(
        pool.full_to_window[dummy],
        torch.arange(sizes.n_win_slots - P, sizes.n_win_slots),
    )
    n_pages = sizes.n_win_slots // P - 1
    pool.alloc_swa(_expand([i * P for i in range(n_pages)]))
    assert pool.swa_available_size() == 0
    with pytest.raises(RuntimeError):
        pool.alloc_swa(_expand([sizes.full_token - P]))


# ----- sizing -----------------------------------------------------------------


def test_sizes_are_non_none_only_at_the_owning_source():
    _, sizes, _ = _pool()
    # All four tiers belong to the KV source: the index-K cache is filled by the kv-source
    # indexer (the only one with `wk`), even where another index source publishes the topk.
    assert [L for L in range(N_LAYERS) if sizes.cmp_blocks[L] is not None] == list(KV_SOURCES)
    assert [L for L in range(N_LAYERS) if sizes.idx_blocks[L] is not None] == list(KV_SOURCES)
    assert [L for L in range(N_LAYERS) if sizes.state_slots[L] is not None] == list(KV_SOURCES)
    assert [L for L in range(N_LAYERS) if sizes.ring_sizes[L] is not None] == list(KV_SOURCES)
    # One band is one allocation: 3 ratio-2 bands + 1 ratio-1 band.
    assert sizes.cmp_blocks[2] == sizes.cmp_blocks[8] == sizes.cmp_blocks[14]
    assert sizes.cmp_blocks[2] == sizes.full_token // 2
    assert sizes.cmp_blocks[20] == sizes.full_token
    assert sizes.state_slots[20] == sizes.n_win_pages * 1
    assert sizes.state_slots[2] == sizes.n_win_pages * 2


def test_pool_bytes_is_monotone_in_num_pages_and_matches_total_bytes():
    args = _args()
    prev = -1
    for num in range(2, 12):
        sizes = dsv41_pool_sizes(num_pages=num, args=args, swa_ratio=0.5, P=P)
        n = dsv41_pool_bytes(sizes, args, n_scratch=3)
        assert n > prev
        prev = n

    pool, sizes, args = _pool(num_pages=8, n_scratch=3)
    assert pool.total_bytes() == dsv41_pool_bytes(sizes, args, n_scratch=3)


def test_solve_num_pages_respects_the_byte_budget():
    args = _args()
    budget = 1 << 30
    sizes = dsv41_solve_num_pages(budget, args, 0.5, floor_win_pages=21, P=P, n_scratch=3)
    assert sizes.full_token > 21 * P  # grew past the window working set
    assert dsv41_pool_bytes(sizes, args, n_scratch=3) <= budget
    with pytest.raises(ValueError):
        dsv41_solve_num_pages(1, args, 0.5, floor_win_pages=21, P=P, n_scratch=3)


def test_classmethod_cost_surface_delegates_to_the_cost_model():
    args = _args()
    cfg = _config(args=args)
    per_page, fixed, page_tokens, min_reserve = DSV41PagedKVCache.kv_cost(cfg)
    assert page_tokens == cfg.page_size == P
    assert per_page > 0 and fixed >= 0 and min_reserve > 0
    assert (per_page, fixed, min_reserve) == dsv41_auto_cost_model(
        args, 0.5, 21, P=P, n_scratch=cfg.max_running_req + 1
    )
    # 8-page prefill reach (capped) + the concurrent working set reservation.
    assert DSV41PagedKVCache.min_kv_tokens(cfg) == 21 * P

    pool, _, _ = _pool()
    kv_b, win_b = pool.unit_bytes()
    assert kv_b > 0 and win_b > 0

    small = DSV41PagedKVCache.solve_num_pages(cfg, 4 << 30)
    large = DSV41PagedKVCache.solve_num_pages(cfg, 8 << 30)
    assert small > 1 and large >= small


def test_force_small_pool_env_hook(monkeypatch):
    monkeypatch.setenv("DSV41_FORCE_SMALL_POOL", "3")
    sizes = _dsv41_pool_sizes(_config(), 8)
    assert sizes.full_token == 3 * P


def test_rebuild_resizes_in_place_and_keeps_band_sharing():
    pool, _, args = _paged_pool()
    before = pool.cmp_pool[20]
    new_sizes = dsv41_pool_sizes(num_pages=12, args=args, swa_ratio=0.5, P=P)
    pool.rebuild(new_sizes)
    assert pool.sizes is new_sizes
    assert pool.cmp_pool[20] is not before
    assert pool.cmp_pool[25] is pool.cmp_pool[20]
    assert pool.state_ring[25] is pool.state_ring[20]
    assert pool.total_bytes() == dsv41_pool_bytes(new_sizes, args, n_scratch=pool.n_scratch)
    assert pool.swa_available_size() > 0  # _init_paged_state re-ran on the new sizes


def test_validate_rebuild_fit_and_reject_paths():
    pool, _, _ = _paged_pool()
    cfg = _config()
    common = dict(
        target_moe=0, per_expert_bytes=0, baseline_free=100 << 30, weights_bytes=0,
        current_num_pages=8,
    )
    pool.validate_rebuild(cfg, num_pages=None, **common)
    with pytest.raises(CacheRebuildRejected):  # MoE-only rebuild that cannot fit
        pool.validate_rebuild(cfg, num_pages=None, **{**common, "baseline_free": 1 << 20})
    with pytest.raises(CacheRebuildRejected):  # below the window working-set floor
        pool.validate_rebuild(cfg, num_pages=2, **common)


# ----- dispatch ---------------------------------------------------------------


def _spec_stub(attn_type):
    spec = KVCacheGroupSpec(
        name="dsv41",
        layer_ids=tuple(range(N_LAYERS)),
        num_kv_heads=1,
        head_dim=512,
        sliding_window=P,
        attn_type=attn_type,
    )
    return SimpleNamespace(kv_cache_group_specs=lambda: (spec,))


def test_resolve_pool_class_dispatches_dsv41_not_dsv4_or_mha():
    assert resolve_pool_class(_spec_stub(AttnType.DSV41)) is DSV41PagedKVCache
    assert resolve_pool_class(_spec_stub(AttnType.DSV4)) is DSV4PagedKVCache
    assert resolve_pool_class(_spec_stub(AttnType.SWA)) is not DSV41PagedKVCache
    # getattr fallback for duck-typed configs that don't implement the spec walk.
    assert resolve_pool_class(SimpleNamespace(dsv41_args=_args())) is DSV41PagedKVCache
    assert resolve_pool_class(SimpleNamespace(dsv4_args=object())) is DSV4PagedKVCache


def test_create_kv_pool_builds_a_usable_dsv41_pool():
    pool = create_kv_pool(_config(), num_pages=8, device=DEVICE, dtype=torch.bfloat16)
    assert isinstance(pool, DSV41PagedKVCache)
    assert pool.num_layers == N_LAYERS
    assert pool.sizes.full_token == (8 + 1) * P  # + the dummy page
    assert pool.swa_available_size() > 0
    assert pool.cmp_pool[25] is pool.cmp_pool[20]


def test_create_kv_pool_wires_the_radix_flag_from_cache_type():
    # The pool is told whether a prefix cache exists at all: 'naive' keeps the no-reuse window
    # floor, anything else (the engine resolves DSV4.1 'radix' -> 'swa_radix', see
    # _adjust_dsv41_config) reserves the generic radix live-tail floor as well. Guards the flag
    # falling out of the create_kv_pool wiring, which is what kept V4.1 on the no-reuse path.
    reuse = create_kv_pool(_config(cache_type="swa_radix"), num_pages=8, device=DEVICE,
                           dtype=torch.bfloat16)
    naive = create_kv_pool(_config(cache_type="naive"), num_pages=8, device=DEVICE,
                           dtype=torch.bfloat16)
    assert reuse._paged_params == (2, True)
    assert naive._paged_params == (2, False)
    # The radix reservation is a superset of the naive one, so the pool's default prefill chunk
    # (half of what is left) can only shrink.
    assert reuse.prefill_chunk_budget <= naive.prefill_chunk_budget
