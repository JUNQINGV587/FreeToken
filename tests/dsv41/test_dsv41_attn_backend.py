"""DSV4.1 attention backend: the band-routed addressing contract it takes over from the pool.

V4.1's compressed/indexer/state tiers are owned by a band's KV source and shared with every
consumer layer (by identity), so the backend must reach them through the pool's routing
accessors rather than by indexing a per-layer list. These are CPU-only shape, addressing and
staging checks -- the kernels themselves are covered in tests/kernels/.

The ring sections cover V4.1's geometry: ``ring_size == ratio`` (1 or 2), item = ``2 *
head_dim`` fp32, and ``P % ratio == 0`` so a token group never straddles a window page and no
boundary carry exists.
"""

from __future__ import annotations

import pytest
import torch

from freetoken.core import Batch, Context, Req, SamplingParams, get_global_ctx, set_global_ctx
from freetoken.kvcache.dsv41_cost_model import dsv41_pool_sizes
from freetoken.kvcache.dsv41_paged_pool import DSV41PagedKVCache
from freetoken.models.deepseek_v41.args import DeepseekV41Args

P, MRR, DEVICE = 128, 4, torch.device("cpu")
# Layer 0 is window-only, layer 1 owns the band (and is both a KV and an index source), 2 is
# its consumer, 3 is the ratio-1 tail (it owns a band only when a test asks for one).
RATIOS = (0, 2, 2, 1)
KV_SOURCES = (1,)
IDX_SOURCES = (1,)

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _ctx(pool):
    try:
        ctx = get_global_ctx()
    except AssertionError:
        ctx = Context(page_size=P)
        set_global_ctx(ctx)
    ctx.kv_cache = pool
    return ctx


def _stack(num_pages=32, max_seq_len=8192, page=P, device=DEVICE, **over):
    fields = dict(
        compress_ratios=RATIOS,
        kv_source_layers=KV_SOURCES,
        index_source_layers=IDX_SOURCES,
        candidate_source_layer=1,
    )
    fields.update(over)
    args = DeepseekV41Args(
        max_batch_size=MRR + 1, n_layers=len(fields["compress_ratios"]), max_seq_len=max_seq_len,
        head_dim=512, index_head_dim=128, window_size=P, **fields,
    )
    sizes = dsv41_pool_sizes(num_pages=num_pages, args=args, swa_ratio=1.0, P=page)
    pool = DSV41PagedKVCache(
        sizes=sizes, args=args, device=device, P=page, n_scratch=MRR + 1
    )
    pool._init_paged_state(MRR, True)
    pt = torch.zeros(MRR + 1, max_seq_len, dtype=torch.int32, device=device)
    # The pool's reserved dummy region is its LAST full page, permanently bound by
    # _init_paged_state, so the dummy row points there (the generic fill_(num_tokens) convention
    # with num_tokens = the allocatable token count).
    pt[MRR].fill_(sizes.full_token - page)
    pt[2, :300] = torch.arange(300, dtype=torch.int32)
    for page_id in range(3):              # bind row 2's window pages (positions 0..383)
        pool.bind_window_pages(page_id * page, page_id * page)
    pool.full_loc_map = pt
    _ctx(pool)

    from types import SimpleNamespace

    from freetoken.attention.dsv41_sparse import DSV41SparseAttnBackend

    # the backend reads only dsv41_args off the model config
    backend = DSV41SparseAttnBackend(SimpleNamespace(dsv41_args=args))
    return backend, pool, pt, sizes


def _decode_batch(rows, positions):
    reqs = [
        Req(input_ids=torch.zeros(1, dtype=torch.int32), table_idx=int(t), cached_len=0,
            output_len=1, uid=i, sampling_params=SamplingParams(), cache_handle=None)
        for i, t in enumerate(rows)
    ]
    batch = Batch(reqs=reqs, phase="decode")
    batch.padded_reqs = reqs
    batch.active_table_idx = torch.tensor(rows, dtype=torch.int64)
    batch.positions = torch.tensor(positions, dtype=torch.int64)
    return batch


# ----- band routing (accessors, not list indexing) ---------------------------------------------
def test_tier_accessors_route_a_consumer_to_its_band_owner():
    """One compressor per band: the consumer layer's pool IS the owner's tensor, so the
    accessor is the only correct answer (a list index would be right by accident)."""
    backend, pool, _, _ = _stack()
    assert backend.compress_pool(2, "attn") is backend.compress_pool(1, "attn")
    assert backend.compress_pool(2, "idx") is backend.compress_pool(1, "idx")
    assert backend.compress_state_ring(2, "attn") is backend.compress_state_ring(1, "attn")
    # V4.1's Indexer has no compressor of its own: no ring exists under that tier anywhere.
    assert backend.compress_state_ring(1, "idx") is None
    assert backend.compress_state_ring(2, "idx") is None
    # Layer 0 sits below the first KV source: window-only, no band at all.
    assert pool.cmp_pool_of(0) is None and pool.cmp_source_of(0) is None


def test_scratch_rows_sit_past_each_band_capacity():
    """A decode step that completes no block writes to its own scratch row, which is allocated
    beyond the block rows the allocator hands out -- so it can never collide with a real one."""
    backend, _, _, sizes = _stack()
    assert backend.compress_scratch_base(2, "attn") == sizes.cmp_blocks[1]
    assert backend.compress_scratch_base(2, "idx") == sizes.idx_blocks[1]
    assert backend.compress_pool(1, "attn").shape[0] == sizes.cmp_blocks[1] + MRR + 1


# ----- addressing -------------------------------------------------------------------------------
def test_blocks_to_global_maps_a_block_to_its_band_rows():
    backend, _, pt, _ = _stack()
    blocks = torch.tensor([0, 3, 7, -1])
    got = backend.blocks_to_global(blocks, 2, ti=2)
    # row 2's live locs are the identity, so block b floors to row b at ratio 2.
    assert got.tolist() == [0, 3, 7, -1]
    # the -1 sentinel is a gather-only convention: it must survive the round trip.
    assert int(got[-1]) == -1
    # decode form reads the snapshot, so it is one step removed from the live table.
    pt[2, :300] = torch.arange(300, dtype=torch.int32)
    batch = _decode_batch([2], [9])
    backend.prepare_metadata(batch)
    with get_global_ctx().forward_batch(batch):
        dec = backend.blocks_to_global(torch.tensor([[0, 3, -1]]), 2, rows=torch.tensor([0]))
    assert dec.shape == (1, 1, 3) and dec.flatten().tolist() == [0, 3, -1]


def test_win_cols_to_global_preserves_the_sentinel():
    backend, _, _, _ = _stack()
    lut = torch.tensor([100, 101, 102, 103], dtype=torch.int64)
    cols = torch.tensor([[0, 2, -1]])
    got = backend.win_cols_to_global(cols, lut)
    assert got.tolist() == [[100, 102, -1]]


def test_window_slots_of_reads_the_live_table():
    backend, pool, pt, _ = _stack()
    slots = backend.window_slots_of(2, 0, 4)
    assert slots.tolist() == [0, 1, 2, 3]
    pt[2, 400:402] = -1                                # outside the bound pages
    assert backend.window_slots_of(2, 400, 402).tolist() == [-1, -1]


# ----- compress-state ring ----------------------------------------------------------------------
def test_state_loc_is_page_local_and_preserves_negatives():
    backend, pool, _, _ = _stack()
    slots = torch.tensor([0, 1, P - 1, P, P + 1, 3 * P + 5, -1])
    # ring_size 1: one state cell per window page.
    assert backend.state_loc(slots, 1).tolist() == [0, 0, 0, 1, 1, 3, -1]
    # ring_size 2: the in-page phase picks the cell, which is what a partial group is keyed by.
    # Slots 1 and P-1 share a cell on purpose: same page, same phase, so they are two states of
    # one page that are never live together.
    assert backend.state_loc(slots, 2).tolist() == [0, 1, 1, 2, 3, 7, -1]
    assert pool.state_loc(slots, 2, P).tolist() == backend.state_loc(slots, 2).tolist()


def test_state_round_trip_for_ring_size_1_and_2():
    """The model persists exactly one trailing partial group per (row, window page); a page's
    block is complete on its own because a group cannot straddle it, so one slot per page is
    the whole live set."""
    backend, pool, _, _ = _stack(kv_source_layers=(1, 3))
    assert pool.ring_size(1) == 2 and pool.ring_size(3) == 1
    for layer_id, ring_size, slots in (
        (1, 2, [0, P, 2 * P, 3 * P + 1]),
        (3, 1, [0, P + 5, 2 * P + 7]),
    ):
        ws = torch.tensor(slots)
        blocks = torch.randn(len(slots), ring_size, 2 * pool.head_dim)
        backend.write_state(layer_id, "attn", ws, ring_size, blocks)
        got = backend.read_state(layer_id, "attn", ws, ring_size)
        assert got.shape == (len(slots), ring_size, 2 * pool.head_dim)
        assert torch.equal(got, blocks)


def test_state_is_keyed_by_page_not_by_slot():
    """A ring block belongs to a PAGE, and the phase picks a cell inside it. Two slots of one
    page therefore read the same block - which is what lets the model resume a partial group
    whose phase moved on within the page."""
    backend, pool, _, _ = _stack()
    block = torch.randn(1, 2, 2 * pool.head_dim)
    backend.write_state(1, "attn", torch.tensor([P + 1]), 2, block)
    assert torch.equal(backend.read_state(1, "attn", torch.tensor([P]), 2), block)

    assert backend.state_loc(torch.tensor([P]), 2).tolist() == [2]
    assert backend.state_loc(torch.tensor([P + 1]), 2).tolist() == [3]


def test_state_round_trip_at_a_non_default_page_size():
    """The ring is page-local, so its geometry must follow the pool's P, not a module constant."""
    backend, pool, _, _ = _stack(page=64, max_seq_len=4096)
    assert pool.P == 64
    ws = torch.tensor([0, 64, 128, 3 * 64 + 1])
    blocks = torch.randn(4, 2, 2 * pool.head_dim)
    backend.write_state(1, "attn", ws, 2, blocks)
    assert torch.equal(backend.read_state(1, "attn", ws, 2), blocks)


def test_state_block_of_a_negative_slot_is_the_cleared_scratch():
    """``-1`` is the gather sentinel, and every set re-clears the row it lands on: the ring
    stores an EMPTY group (zero kv, -inf score) rather than whatever was written last."""
    backend, pool, _, _ = _stack()
    blocks = torch.randn(1, 2, 2 * pool.head_dim)
    backend.write_state(1, "attn", torch.tensor([-1]), 2, blocks)   # discarded, not an error
    got = backend.read_state(1, "attn", torch.tensor([-1]), 2)
    assert got.shape == (1, 2, 2 * pool.head_dim)
    assert torch.equal(got[..., : pool.head_dim], torch.zeros(1, 2, pool.head_dim))
    assert torch.isinf(got[..., pool.head_dim:]).all() and (got[..., pool.head_dim:] < 0).all()


def test_state_write_under_a_tier_with_no_ring_is_a_hard_error():
    """The pool routes a ring read/write to the ATTENTION ring, so an "idx" call would silently
    hit the wrong tier; V4.1's indexer owns no state and the backend must say so."""
    backend, _, _, _ = _stack()
    with pytest.raises(AssertionError):
        backend.write_state(1, "idx", torch.tensor([0]), 2, torch.zeros(1, 2, 1024))
    with pytest.raises(AssertionError):
        backend.read_state(1, "idx", torch.tensor([0]), 2)


# ----- decode staging ---------------------------------------------------------------------------
def test_decode_compress_rows_route_an_incomplete_step_to_scratch():
    backend, _, _, sizes = _stack()
    batch = _decode_batch([2, MRR], [259, 0])
    backend.prepare_metadata(batch)
    with get_global_ctx().forward_batch(batch):
        rows = torch.tensor([0, 1])
        pos = torch.tensor([259, 0])
        got = backend.decode_compress_rows(
            rows, pos, 2, 1, "attn", torch.tensor([True, False])
        )
    # row 0 finished a block at pos 259 -> the arithmetic row; row 1 did not -> its own scratch.
    assert int(got[0]) == 259 // 2
    assert int(got[1]) == 1 + sizes.cmp_blocks[1]


def test_eager_decode_snapshots_the_live_rows():
    """Even without a graph the metadata carries a COPY, so the next batch's allocate_paged
    cannot move the rows this forward reads."""
    backend, _, pt, sizes = _stack()
    batch = _decode_batch([2, MRR], [259, 0])
    backend.prepare_metadata(batch)
    assert batch.attn_metadata.full_snap is None          # deferred, not taken yet
    with get_global_ctx().forward_batch(batch):
        snap = backend.snapshot()
        assert backend.snapshot() is snap                 # materialized once, then cached
    assert snap.dtype == torch.int64
    assert torch.equal(snap[0, :300], pt[2, :300].to(torch.int64))
    assert int(snap[1, 0]) == sizes.full_token - P  # dummy row -> the reserved tail page
    pt[2, :300] = -7                                # a later allocate mutating the live table
    assert torch.equal(snap[0, :300], torch.arange(300, dtype=torch.int64))


def test_prefill_metadata_carries_segments():
    """Prefill addressing -- (offset, extend_len, table_idx, start_pos) per request, tiling the
    flat token stream -- rides the metadata, so the model's forward never reads Req fields."""
    backend, _, _, _ = _stack()
    reqs = [
        Req(input_ids=torch.zeros(300, dtype=torch.int32), table_idx=2, cached_len=256,
            output_len=1, uid=0, sampling_params=SamplingParams(), cache_handle=None),
        Req(input_ids=torch.zeros(5, dtype=torch.int32), table_idx=1, cached_len=0,
            output_len=1, uid=1, sampling_params=SamplingParams(), cache_handle=None),
    ]
    batch = Batch(reqs=reqs, phase="prefill")
    batch.padded_reqs = reqs
    backend.prepare_metadata(batch)
    md = batch.attn_metadata
    assert md.segments == [(0, 44, 2, 256), (44, 5, 1, 0)]
    assert md.full_snap is None
    assert int(md.get_last_indices(2)[0]) == 43 and int(md.get_last_indices(2)[1]) == 48


def test_staging_width_tracks_the_engine_ceiling():
    backend, _, _, _ = _stack()
    backend.init_capture_graph(max_seq_len=1024, bs_list=[1, 2])
    batch = _decode_batch([2, MRR], [259, 0])
    backend.prepare_for_replay(batch)
    assert batch.attn_metadata.stage_width == 1024
    with get_global_ctx().forward_batch(batch):
        assert backend.snapshot().shape[1] == 1024


def test_capture_stages_the_dummy_row_and_recapture_is_clean():
    backend, pool, _, sizes = _stack()
    backend.init_capture_graph(max_seq_len=1024, bs_list=[2])
    batch = _decode_batch([MRR, MRR], [0, 0])
    backend.prepare_for_capture(batch)
    snap = batch.attn_metadata.full_snap
    assert int(snap[0, 0]) == sizes.full_token - P
    assert (pool.translate_full_to_window(snap[:, :4]) >= 0).all()

    backend.reset_capture()
    assert backend.capture is None and backend.capture_bs == []
    backend.init_capture_graph(max_seq_len=512, bs_list=[2])
    assert backend.capture.full_snap.shape == (2, 512)
    assert (backend.capture.full_snap == -1).all()


def test_window_ctx_is_layer_invariant_and_row_isolated():
    backend, _, _, _ = _stack()
    batch = _decode_batch([2, MRR], [259, 0])
    backend.prepare_metadata(batch)
    with get_global_ctx().forward_batch(batch):
        pos = batch.positions
        rows = torch.arange(2)
        md = batch.attn_metadata
        ws, prev, ring = md.window_ctx(pos, rows)
        assert ws.shape == (2,) and prev.shape == (2,) and ring.shape == (2, 1, P)
        # row 1 sits at position 0: exactly one live ring column, the rest masked
        assert int((ring[1, 0] >= 0).sum()) == 1
        # row 0 is past a full window: every ring column is live
        assert int((ring[0, 0] >= 0).sum()) == P
        # NEVER cached: a cached tensor would leave these gathers out of the captured graph.
        again = md.window_ctx(pos, rows)
        assert again[2] is not ring and torch.equal(again[2], ring)


# ----- kernel-level ---------------------------------------------------------------------------
@requires_cuda
def test_attend_is_finite_over_window_and_compressed_slots():
    """The two pools the kernel reads: the band's compressed tier, and (ratio-0 layers) the
    window pool aliased in its place."""
    backend, pool, _, _ = _stack(device=torch.device("cuda"))
    n_heads, head_dim = 16, pool.head_dim
    dev = pool.device
    q = torch.randn(1, 1, n_heads, head_dim, device=dev, dtype=torch.bfloat16)
    sink = torch.zeros(n_heads, device=dev, dtype=torch.float32)
    kv = torch.randn(4, head_dim, device=dev, dtype=torch.bfloat16)
    slots = torch.arange(4, device=dev)
    topk = torch.tensor([[[0, 1, 2, 3, 0, 1, 2, 3]]], device=dev, dtype=torch.int32)

    # Layer 1 is a band owner: window (4) + compressed (4) columns.
    backend.store_window(kv, 1, slots)
    out = backend.attend(q, 1, topk, 4, sink, head_dim ** -0.5)
    assert out.shape == q.shape and torch.isfinite(out).all()

    # Layer 0 is ratio-0: no compressed tier, so the window pool is aliased into the argument
    # that keeps the kernel's two-pool stride assert happy (n_window == topk here).
    backend.store_window(kv, 0, slots)
    out0 = backend.attend(q, 0, topk[:, :, :4], 4, sink, head_dim ** -0.5, has_compression=False)
    assert out0.shape == q.shape and torch.isfinite(out0).all()
