"""P2b integration: CacheManager hybrid path (match_req -> cache_req donate -> prefix hit).
CPU, real LinearStatePool + page_table, hand-built Reqs. Exercises the two-currency wiring
without the full scheduler/engine."""
from __future__ import annotations

import contextlib
import itertools
import random
import time
from types import SimpleNamespace

import pytest
import torch

from freetoken.core import Req, SamplingParams
from freetoken.kvcache.linear_state_pool import LinearStatePool
from freetoken.models.config import LinearGatedDeltaGroupConfig
from freetoken.scheduler.cache import CacheManager


@contextlib.contextmanager
def _deterministic_clock():
    """Port of tests/kvcache/radix/driver.deterministic_clock (B7): the trees stamp nodes with
    one time.monotonic_ns() read per walk, so a real clock's coarse resolution adds ties
    BETWEEN walks and makes stamp-ordering asserts flaky."""
    saved = time.monotonic_ns
    time.monotonic_ns = itertools.count(1_000_000_001).__next__   # type: ignore[assignment]
    try:
        yield
    finally:
        time.monotonic_ns = saved                                 # type: ignore[assignment]


@pytest.fixture(autouse=True)
def det_clock():
    with _deterministic_clock():
        yield


def _pool(num_slots=16):
    g = LinearGatedDeltaGroupConfig(
        name="linear", layer_ids=(0,), num_key_heads=2, num_value_heads=4,
        key_head_dim=16, value_head_dim=16, conv_kernel_dim=4, output_gate="silu",
    )
    return LinearStatePool(group=g, num_slots=num_slots, dtype=torch.bfloat16,
                           device=torch.device("cpu"), tp_size=1)


def _pend(ids):
    # int32 to match production Req.input_ids dtype (fast_compare_key needs consistent dtype)
    t = torch.tensor(ids, dtype=torch.int32)
    return SimpleNamespace(input_ids=t, input_len=len(ids))


def test_hybrid_cache_manager_donate_then_hit():
    pool = _pool()
    page_table = torch.zeros(4, 64, dtype=torch.int32)
    cm = CacheManager(64, 1, page_table, "hybrid_radix", linear_state_pool=pool)
    assert cm.is_hybrid

    # cold match on an empty tree
    mr = cm.match_req(_pend([1, 2, 3, 4, 5]))
    assert mr.cuda_handle.cached_len == 0 and mr.mamba_value is None

    # admit req A: allocate live + ping-pong, stage KV pages, mark a ×N snapshot at boundary 4
    live, pp = pool.alloc(1)[0], tuple(pool.alloc(2))
    page_table[0, :4] = torch.tensor([100, 101, 102, 103], dtype=torch.int32)
    reqA = Req(input_ids=torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32), table_idx=0,
               cached_len=4, output_len=1, uid=0, sampling_params=SamplingParams(),
               cache_handle=mr.cuda_handle)
    reqA.linear_slot_idx, reqA.mamba_ping_pong = live, pp
    reqA.mamba_next_track_idx = 1            # flipped from 0 in build_fla_metadata; frozen = pp[0]
    reqA.mamba_last_track_seqlen = 4
    cm.lock(mr.cuda_handle)

    free_before = pool.num_free_slots
    cm.cache_req(reqA, finished=False)       # donate a clone of pp[0] at boundary 4
    # copy-on-donate: one slot is alloc'd for the tree's clone and the request hands nothing over
    assert pool.num_free_slots == free_before - 1  # the clone; the pair below is still the request's
    assert reqA.mamba_ping_pong == pp              # the request keeps both ping-pong slots

    # req B shares the [1,2,3,4] prefix -> HIT: returns the donated snapshot + reused KV
    mrB = cm.match_req(_pend([1, 2, 3, 4, 9]))
    assert mrB.cuda_handle.cached_len == 4
    assert mrB.mamba_value not in (live, *pp)      # the tree serves its own clone, not the request's slot
    assert mrB.cuda_handle.get_matched_indices().tolist() == [100, 101, 102, 103]


def test_hybrid_finish_donates_live_slot():
    pool = _pool()
    page_table = torch.zeros(4, 64, dtype=torch.int32)
    cm = CacheManager(64, 1, page_table, "hybrid_radix", linear_state_pool=pool)

    mr = cm.match_req(_pend([7, 8, 9, 10]))
    live, pp = pool.alloc(1)[0], tuple(pool.alloc(2))
    page_table[1, :3] = torch.tensor([200, 201, 202], dtype=torch.int32)
    req = Req(input_ids=torch.tensor([7, 8, 9, 10], dtype=torch.int32), table_idx=1,
              cached_len=3, output_len=1, uid=1, sampling_params=SamplingParams(),
              cache_handle=mr.cuda_handle)
    req.linear_slot_idx, req.mamba_ping_pong = live, pp
    cm.lock(mr.cuda_handle)

    free_before = pool.num_free_slots
    cm.cache_req(req, finished=True)         # donate a clone of the live slot (final state)
    # copy-on-donate: the tree holds a private clone; every slot the request held comes back
    assert pool.num_free_slots == free_before + 2   # clone alloc'd (-1), live + pair freed (+3)
    mr2 = cm.match_req(_pend([7, 8, 9, 10]))
    assert mr2.cuda_handle.cached_len == 3 and mr2.mamba_value not in (live, *pp)


def test_free_req_slots_idempotent():
    """C2: a finish/abort double-free of the same request must NOT push its GDN slots twice."""
    pool = _pool()
    pt = torch.zeros(4, 64, dtype=torch.int32)
    cm = CacheManager(64, 1, pt, "hybrid_radix", linear_state_pool=pool)
    live, pp = pool.alloc(1)[0], tuple(pool.alloc(2))
    req = Req(input_ids=torch.tensor([1, 2, 3], dtype=torch.int32), table_idx=0, cached_len=2,
              output_len=1, uid=0, sampling_params=SamplingParams(), cache_handle=None)
    req.linear_slot_idx, req.mamba_ping_pong = live, pp
    base = pool.num_free_slots
    cm._free_req_slots(req)
    assert pool.num_free_slots == base + 3        # live + 2 ping-pong returned once
    cm._free_req_slots(req)                        # second free (abort/finish race)
    assert pool.num_free_slots == base + 3         # idempotent: nothing pushed twice


def test_rebuild_reclaims_donated_gdn_slots():
    """C5: a runtime cache rebuild must return the discarded tree's GDN snapshot slots (idle)."""
    pool = _pool(num_slots=16)
    pt = torch.zeros(4, 64, dtype=torch.int32)
    cm = CacheManager(64, 1, pt, "hybrid_radix", linear_state_pool=pool)
    mr = cm.match_req(_pend([7, 8, 9, 10]))
    live, pp = pool.alloc(1)[0], tuple(pool.alloc(2))
    pt[1, :3] = torch.tensor([200, 201, 202], dtype=torch.int32)
    req = Req(input_ids=torch.tensor([7, 8, 9, 10], dtype=torch.int32), table_idx=1, cached_len=3,
              output_len=1, uid=1, sampling_params=SamplingParams(), cache_handle=mr.cuda_handle)
    req.linear_slot_idx, req.mamba_ping_pong = live, pp
    cm.lock(mr.cuda_handle)
    cm.cache_req(req, finished=True)              # donates `live` to the tree, frees ping-pong
    assert pool.num_free_slots < pool.num_slots - 1   # a slot is now tree-owned
    cm.rebuild(64, pt)                            # idle rebuild discards the tree
    assert pool.num_free_slots == pool.num_slots - 1  # all GDN slots reclaimed (no leak)


def test_prefill_chunk_ends_on_a_page_boundary():
    """A hybrid chunk must end page-aligned: the snapshot commit skips any other boundary."""
    from freetoken.scheduler.prefill import ChunkedReq, PrefillAdder
    from freetoken.scheduler.table import TableManager
    from freetoken.scheduler.utils import PendingReq

    pool = _pool()
    pt = torch.zeros(4, 512, dtype=torch.int32)
    cm = CacheManager(64, 64, pt, "hybrid_radix", linear_state_pool=pool)
    assert cm.prefill_chunk_align == 64
    tm = TableManager(max_running_reqs=4, page_table=pt)
    pending = PendingReq(0, torch.arange(300, dtype=torch.int32), SamplingParams(max_tokens=1))

    adder = PrefillAdder(token_budget=100, reserved_size=0, cache_manager=cm, table_manager=tm)
    req = adder.try_add_one(pending)
    assert isinstance(req, ChunkedReq) and req.extend_len == 64

    # a budget below one page keeps the unaligned chunk rather than stalling the request
    adder = PrefillAdder(token_budget=40, reserved_size=0, cache_manager=cm, table_manager=tm)
    assert adder.try_add_one(pending).extend_len == 40


def _chunked_setup(num_slots=16, prompt_len=300, page_size=64, num_pages=32, uid=0):
    """Hybrid manager + table + one pending prompt, sized for chunked-prefill flows."""
    from freetoken.scheduler.table import TableManager
    from freetoken.scheduler.utils import PendingReq

    pool = _pool(num_slots)
    pt = torch.zeros(8, num_pages * page_size, dtype=torch.int32)
    cm = CacheManager(num_pages, page_size, pt, "hybrid_radix", linear_state_pool=pool)
    tm = TableManager(max_running_reqs=8, page_table=pt)
    pending = PendingReq(uid, torch.arange(prompt_len, dtype=torch.int32),
                         SamplingParams(max_tokens=1))
    return cm, pool, pt, tm, pending


def _forward_chunk(adder, pending, track_len, next_track_idx):
    """One chunked-prefill step in production order: create the chunk (the hybrid per-chunk
    donation of the PRIOR chunk fires inside try_add_one), make its pages real, stamp the
    xCHUNK track state _build_track_metadata leaves behind (boundary + flip, only when the
    extend crosses a boundary), advance cached_len (engine complete_one). Returns
    (req, inherited_track_len) -- the mark the chunk inherited before its own forward,
    i.e. what the donation consumed."""
    from freetoken.scheduler.prefill import ChunkedReq

    req = adder.try_add_one(pending)
    inherited_track = req.mamba_last_track_seqlen
    adder.cache_manager.allocate_paged([req])
    if track_len is not None:
        req.mamba_last_track_seqlen = track_len
        req.mamba_next_track_idx = next_track_idx
    req.complete_one()
    pending.chunked_req = req if isinstance(req, ChunkedReq) else None
    return req, inherited_track


def _snapshot_chain(cm):
    """The tree's boundary nodes along the single inserted chain, root-first."""
    chain, node = [], cm.prefix_cache.root
    while node.children:
        assert len(node.children) == 1
        node = next(iter(node.children.values()))
        chain.append(node)
    return chain


def test_prefill_continuation_donates_prior_boundary_before_inheriting():
    """AMENDMENT 2: creating a continuation commits the PRIOR chunk's tracked x64 boundary
    BEFORE inheriting its state (hybrid only): the continuation sees the post-commit handle
    and ping-pong tuple and a consumed track mark. Committing at the drain instead would
    double-free: the continuation has already copied (overlap) the pre-commit state."""
    from freetoken.scheduler.prefill import ChunkedReq, PrefillAdder
    from freetoken.scheduler.table import TableManager
    from freetoken.scheduler.utils import PendingReq

    pool = _pool()
    pt = torch.zeros(4, 512, dtype=torch.int32)
    cm = CacheManager(64, 64, pt, "hybrid_radix", linear_state_pool=pool)
    tm = TableManager(max_running_reqs=4, page_table=pt)
    pending = PendingReq(0, torch.arange(300, dtype=torch.int32), SamplingParams(max_tokens=1))

    adder = PrefillAdder(token_budget=100, reserved_size=0, cache_manager=cm, table_manager=tm)
    chunk1 = adder.try_add_one(pending)
    assert isinstance(chunk1, ChunkedReq) and chunk1.extend_len == 64
    assert chunk1.mamba_last_track_seqlen is None  # fresh admit starts untracked
    chunk1.mamba_last_track_seqlen = 64            # the x64 track fired during chunk 1's forward
    chunk1.mamba_next_track_idx = 1                # flipped from 0 in build_fla_metadata; frozen = pp[0]
    cm.allocate_paged([chunk1])                    # mirror _launch_req: make the chunk's pages real
    chunk1.complete_one()                          # cached_len 0 -> 64
    pp0 = chunk1.mamba_ping_pong[0]
    pending.chunked_req = chunk1

    # B3-2b: a budget bounce (image-whole path) AFTER the donation, then retry: the second
    # pass's donation is a no-op (the mark was consumed) and the inherited state is identical.
    pending.mm_items = [SimpleNamespace(offsets=[(100, 200)])]
    bounced = PrefillAdder(token_budget=100, pass_budget=300, reserved_size=0,
                           cache_manager=cm, table_manager=tm, keep_images_whole=True)
    assert bounced.try_add_one(pending) is None    # bounced in _add_one_req, donation already ran
    bounce_state = (pool.num_free_slots, chunk1.cache_handle.cached_len,
                    chunk1.mamba_last_track_seqlen, chunk1.mamba_ping_pong)
    pending.mm_items = None

    adder = PrefillAdder(token_budget=100, reserved_size=0, cache_manager=cm, table_manager=tm)
    req2 = adder.try_add_one(pending)
    assert req2 is not None
    assert (pool.num_free_slots, chunk1.cache_handle.cached_len,
            chunk1.mamba_last_track_seqlen, chunk1.mamba_ping_pong) == bounce_state
    # the donation consumed the mark and rebound the chain BEFORE the inheritance
    assert req2.mamba_last_track_seqlen is None
    assert req2.cached_len == 64
    assert req2.cache_handle.cached_len == 64      # rebound to the donated boundary node
    # copy-on-donate (#287): the tree holds a PRIVATE CLONE; the request keeps its slot
    # for the next track -- no replacement alloc, no shared ownership.
    assert req2.mamba_ping_pong[0] == pp0
    donated0 = next(iter(cm.prefix_cache.root.children.values())).mamba_value
    assert donated0 is not None and donated0 != pp0

    # the boundary is reusable by any request sharing the prefix
    mr = cm.match_req(_pend(list(range(64)) + [99]))
    assert mr.cuda_handle.cached_len == 64
    assert mr.mamba_value == donated0

    # chain integration: the final chunk still commits its own boundary at the drain, and the
    # aligned finish donates the live slot at the next boundary -- no double donation
    cm.allocate_paged([req2])
    req2.complete_one()                            # cached_len 64 -> 128
    live_id = req2.linear_slot_idx
    free_before = pool.num_free_slots
    cm.cache_req(req2, finished=True)
    assert pool.num_free_slots == free_before + 2  # the pair freed; live donated at 128
    mr2 = cm.match_req(_pend(list(range(128)) + [99]))
    assert mr2.cuda_handle.cached_len == 128
    assert mr2.mamba_value is not None and mr2.mamba_value != live_id   # tree's private clone
    cm.prefix_cache.check_integrity()


def test_chunked_req_donates_each_intermediate_boundary():
    """AMENDMENT 2 T-A: a 3-chunk ChunkedReq flow must attach each intermediate chunk's
    tracked x64 snapshot to the tree at its page-aligned boundary as the NEXT chunk is
    created. Pre-fix the ChunkedReq drain skip left ZERO intermediate snapshots -- only the
    final chunk's own boundary ever landed (fails-before)."""
    from freetoken.scheduler.prefill import PrefillAdder

    cm, pool, pt, tm, pending = _chunked_setup(prompt_len=300)
    adder = PrefillAdder(token_budget=128, reserved_size=0, cache_manager=cm, table_manager=tm)
    chunk1, _ = _forward_chunk(adder, pending, 64, 1)       # chunk [0,128): tracks 64, flip -> 1
    pp0 = chunk1.mamba_ping_pong[0]

    adder = PrefillAdder(token_budget=128, reserved_size=0, cache_manager=cm, table_manager=tm)
    chunk2, inherited = _forward_chunk(adder, pending, 192, 0)   # chunk [128,256): tracks 192
    # creating chunk 2 donated chunk 1's boundary 64 before inheriting its state
    a = next(iter(cm.prefix_cache.root.children.values()))
    # copy-on-donate: the node holds a private clone of the frozen slot, not the slot itself
    assert a.length == 64 and a.mamba_value is not None and a.mamba_value != pp0
    assert inherited is None                                # chunk 1's mark was consumed by the donation
    assert chunk2.cache_handle.cached_len == 64             # rebound to the donated node
    assert chunk2.mamba_ping_pong[0] == pp0                 # the request keeps its slot (clone semantics)
    pp1 = chunk2.mamba_ping_pong[1]

    adder = PrefillAdder(token_budget=128, reserved_size=0, cache_manager=cm, table_manager=tm)
    chunk3, inherited3 = _forward_chunk(adder, pending, None, 0)   # final chunk [256,300): no x64 track
    b = next(iter(a.children.values()))
    assert a.length + b.length == 192 and b.mamba_value is not None and b.mamba_value != pp1
    assert b.mamba_ref_count == 1 and a.mamba_ref_count == 0     # newest locked, prior evictable
    assert inherited3 is None                               # chunk 2's mark consumed too
    assert chunk3.cache_handle.cached_len == 192
    assert chunk3.mamba_ping_pong == chunk2.mamba_ping_pong      # inherited the post-donation pair
    # pool: 15 allocatable = 3 req slots + 2 tree snapshots + 10 free
    assert pool.num_free_slots == 10
    assert cm.prefix_cache.mamba_evictable_size == 1 and cm.prefix_cache.mamba_protected == 1
    cm.prefix_cache.check_integrity()


def test_midhistory_divergence_matches_deepest_live_boundary():
    """AMENDMENT 2 T-B: after a 3-chunk prefill donates boundaries 64 and 192, a repeat of
    the full prefix reuses to 192, and a request sharing only the first ~1.5 chunks
    (divergence at 100) matches the deepest LIVE boundary below the divergence: cached_len
    == 64. Pre-fix: 0 -- a full re-prefill (fails-before)."""
    from freetoken.scheduler.prefill import PrefillAdder

    cm, pool, pt, tm, pending = _chunked_setup(prompt_len=300)
    for k, track in enumerate((64, 192, None)):
        adder = PrefillAdder(token_budget=128, reserved_size=0, cache_manager=cm,
                             table_manager=tm)
        nxt = None if track is None else (1 if k == 0 else 0)
        chunk, _ = _forward_chunk(adder, pending, track, nxt)
    chain = _snapshot_chain(cm)
    pp0, pp1 = chain[0].mamba_value, chain[1].mamba_value
    assert (chain[0].length, chain[0].length + chain[1].length) == (64, 192)
    assert pp0 is not None and pp1 is not None

    # full-prefix repeat: reuse up to the deepest committed boundary
    mr = cm.match_req(_pend(list(range(300))))
    assert mr.cuda_handle.cached_len == 192 and mr.mamba_value == pp1
    # mid-history rewrite: the client replays [0,100) then diverges
    ids2 = list(range(100)) + list(range(9000, 9008))
    mr2 = cm.match_req(_pend(ids2))
    assert mr2.cuda_handle.cached_len == 64                 # deepest live boundary <= 100
    assert mr2.mamba_value == pp0


def test_per_chunk_donation_then_finish_keeps_pool_integrity():
    """AMENDMENT 2 T-C (double-free regression). Reconstructed hazard: at the DRAIN the
    continuation has already copied (overlap) the pre-commit handle and ping-pong tuple, so
    a drain-point cache_req (a) leaves the donated snapshot slot id in the continuation's
    stale tuple, which _free_req_slots frees again at finish while the tree owns it, and
    (b) frees [stale_handle.cached_len, prefix_len) -- the pages the prior commit already
    adopted into the tree -- a second time. The fix commits at continuation creation, so
    the donate-then-finish flow must keep both pools exactly conserved."""
    from freetoken.scheduler.prefill import PrefillAdder

    cm, pool, pt, tm, pending = _chunked_setup(prompt_len=300)
    chunks = []
    for k, track in enumerate((64, 192, None)):
        adder = PrefillAdder(token_budget=128, reserved_size=0, cache_manager=cm,
                             table_manager=tm)
        nxt = None if track is None else (1 if k == 0 else 0)
        chunk, _ = _forward_chunk(adder, pending, track, nxt)
        chunks.append(chunk)
    chunk1 = chunks[0]

    # B3-2a: re-committing an already-committed chunk (its L was consumed -> None) must be
    # a no-op: the early return fires before any tree/pool/refcount mutation.
    tree_before = [(n.length, n.mamba_value, n.mamba_ref_count, n.ref_count)
                   for n in _snapshot_chain(cm)]
    free_before_recommit = pool.num_free_slots
    cm.cache_req(chunk1, finished=False)
    assert pool.num_free_slots == free_before_recommit
    assert [(n.length, n.mamba_value, n.mamba_ref_count, n.ref_count)
            for n in _snapshot_chain(cm)] == tree_before

    free_before_finish = pool.num_free_slots
    cm.cache_req(chunk, finished=True)     # finish: unaligned live-donate (300) skipped
    # exactly live + both ping-pong slots returned once; the tree keeps its 2 donated slots
    assert pool.num_free_slots == free_before_finish + 3
    tree = cm.prefix_cache
    assert tree.mamba_evictable_size + tree.mamba_protected == 2
    assert chunk.mamba_ping_pong is None and chunk.linear_slot_idx is None
    cm.check_integrity()                   # page + GDN-slot conservation, tree structure
    cm._free_req_slots(chunk)              # abort/finish race: second free must be a no-op
    assert pool.num_free_slots == free_before_finish + 3


def test_donation_fund_pressure_evicts_stalest_validated_boundary():
    """AMENDMENT 2 T-D: a 9-slot pool (8 allocatable: live+2pp per req + 1 padding sink)
    holds only 4 unlocked boundary donations; a 10-chunk prompt must fire evict_mamba
    mid-chain and kill the STALEST-VALIDATED boundary first (FIFO by snapshot_lru -- the
    amended Fix-3 ordering becomes load-bearing), never the just-locked tip."""
    from freetoken.scheduler.prefill import PrefillAdder

    cm, pool, pt, tm, pending = _chunked_setup(num_slots=9, prompt_len=1280)
    next_idx = 0
    chunk = None
    for k in range(10):                     # chunks of 128; boundary k = k*128 + 64
        adder = PrefillAdder(token_budget=128, reserved_size=0, cache_manager=cm,
                             table_manager=tm)
        chunk, _ = _forward_chunk(adder, pending, k * 128 + 64, 1 - next_idx)
        next_idx = 1 - next_idx
    cm.cache_req(chunk, finished=False)     # the final chunk's drain commit (boundary 1216)
    chain = _snapshot_chain(cm)
    assert len(chain) == 10 and [c.length for c in chain] == [64] + [128] * 9
    dead, live, tip = chain[:5], chain[5:9], chain[9]
    assert all(c.mamba_value is None for c in dead)      # FIFO by last validation
    assert all(c.mamba_value is not None for c in live)
    assert tip.mamba_value is not None and tip.mamba_ref_count == 1   # just-locked tip survives
    assert pool.num_free_slots == 0                      # steady state: 3 req + 5 tree = 8
    cm.prefix_cache.check_integrity()
    # the aligned finish (1280) donates the live state at boundary 1280; full conservation
    cm.cache_req(chunk, finished=True)
    # copy-on-donate: funding the finish's private clone with a full pool evicts the
    # stalest-validated boundary first (the amended Fix-3 ordering), then the request's
    # three slots come back -- the tree ends with the live clone + the 4 surviving boundaries.
    assert pool.num_free_slots == 3
    assert cm.prefix_cache.mamba_evictable_size == 5 and cm.prefix_cache.mamba_protected == 0
    cm.check_integrity()


def test_finish_does_not_double_donate_final_boundary():
    """AMENDMENT 2 T-E: the final chunk's own boundary is donated ONCE (at its drain
    commit); the finish's pending-frozen branch must not re-attach at any position (the
    commit consumed the mark) and the unaligned live donate is skipped, so the tree keeps
    exactly the two donated boundaries with their original slots."""
    from freetoken.scheduler.prefill import PrefillAdder

    cm, pool, pt, tm, pending = _chunked_setup(prompt_len=200)
    adder = PrefillAdder(token_budget=128, reserved_size=0, cache_manager=cm, table_manager=tm)
    chunk1, _ = _forward_chunk(adder, pending, 64, 1)       # chunk [0,128): tracks 64
    pp0 = chunk1.mamba_ping_pong[0]
    adder = PrefillAdder(token_budget=128, reserved_size=0, cache_manager=cm, table_manager=tm)
    chunk2, _ = _forward_chunk(adder, pending, 192, 0)      # final chunk [128,200): tracks 192
    pp1 = chunk2.mamba_ping_pong[1]
    cm.cache_req(chunk2, finished=False)                    # the drain commit donates 192
    chain = _snapshot_chain(cm)
    # copy-on-donate: the tree's snapshots are private clones, never the request's slot ids
    donated = [c.mamba_value for c in chain]
    assert all(d is not None for d in donated) and pp0 not in donated and pp1 not in donated
    assert chain[1].mamba_ref_count == 1                    # the drain commit locked the tip
    cm.cache_req(chunk2, finished=True)                     # L is None: no pending-frozen re-donate
    assert [c.mamba_value for c in chain] == donated        # unchanged: no double donation
    assert pool.num_free_slots == 13                        # 15 - 2 tree-owned
    cm.check_integrity()


def test_finish_pending_frozen_dedup_keeps_existing_snapshot():
    """AMENDMENT 2 T-E (dedup backstop): a finish whose tracked L names an ALREADY-donated
    boundary must dedup (mamba_exist) -- the tree keeps the ORIGINAL snapshot and the
    donated slot stays req-owned (freed with the pair), never double-owned. The forced
    stale L is defensive legacy state, unreachable in the new design (the drain commit
    clears the mark; a decode-time anchor can only freeze at page-aligned positions
    above every prefill boundary) -- kept to pin the dedup path."""
    from freetoken.scheduler.prefill import PrefillAdder

    cm, pool, pt, tm, pending = _chunked_setup(prompt_len=200)
    adder = PrefillAdder(token_budget=128, reserved_size=0, cache_manager=cm, table_manager=tm)
    chunk1, _ = _forward_chunk(adder, pending, 64, 1)
    pp0 = chunk1.mamba_ping_pong[0]
    adder = PrefillAdder(token_budget=128, reserved_size=0, cache_manager=cm, table_manager=tm)
    chunk2, _ = _forward_chunk(adder, pending, 192, 0)
    pp1 = chunk2.mamba_ping_pong[1]
    cm.cache_req(chunk2, finished=False)                    # drain commit donates 192
    donated = [c.mamba_value for c in _snapshot_chain(cm)]  # the tree's private clones
    # force the collision: a stale L naming the already-donated boundary 64
    chunk2.mamba_last_track_seqlen = 64
    cm.cache_req(chunk2, finished=True)
    chain = _snapshot_chain(cm)
    assert [c.mamba_value for c in chain] == donated        # the original snapshots survive
    assert pool.num_free_slots == 13                        # the colliding clone returned unused
    cm.check_integrity()


def test_aligned_final_boundary_finish_live_donate():
    """B3-3: an ALIGNED final boundary (192 = chunks 64+128). The drain commit donates the
    final chunk's tracked L=128; the finish's live-donate then fires (192 is page-aligned)
    onto a FRESH attach node -- a single chain can never dedup its own live donate, since
    every chain boundary is < cached_len -- so the live slot is DONATED and only the pair
    is freed (+2). A re-prefill of the same prompt (phase B) whose admission matches 128
    (a 191-token match cannot reach the 192 node's 64-token key) finishes with the
    live-donate landing on the EXISTING 192 node -> mamba_exist=True -> keep_live=False:
    live + pair freed (+3), the tree's snapshots untouched (no double-donate/double-free)."""
    from freetoken.scheduler.prefill import ChunkedReq, PrefillAdder
    from freetoken.scheduler.utils import PendingReq

    cm, pool, pt, tm, pending = _chunked_setup(prompt_len=192)
    adder = PrefillAdder(token_budget=100, reserved_size=0, cache_manager=cm, table_manager=tm)
    chunk1, _ = _forward_chunk(adder, pending, 64, 1)       # chunk [0,64): tracks 64
    pp0 = chunk1.mamba_ping_pong[0]
    adder = PrefillAdder(token_budget=128, reserved_size=0, cache_manager=cm, table_manager=tm)
    chunk2, _ = _forward_chunk(adder, pending, 128, 0)      # FINAL [64,192): extend 128 -> L=128
    assert not isinstance(chunk2, ChunkedReq)
    pp1 = chunk2.mamba_ping_pong[1]
    cm.cache_req(chunk2, finished=False)                    # the drain commit donates L=128
    chain = _snapshot_chain(cm)
    assert [n.length for n in chain] == [64, 64]            # spans; cumulative ends 64, 128
    assert sum(n.length for n in chain) == 128              # the drain-donated boundary L=128
    # copy-on-donate: the chain holds private clones; the request's own ids never enter the tree
    donated = [n.mamba_value for n in chain]
    assert all(d is not None for d in donated) and pp0 not in donated and pp1 not in donated

    live_id = chunk2.linear_slot_idx
    free_before = pool.num_free_slots
    cm.cache_req(chunk2, finished=True)                     # the aligned live-donate fires at 192
    chain = _snapshot_chain(cm)
    # the tree holds a CLONE of the live state at 192 (donated clone != the request's live slot)
    assert len(chain) == 3 and chain[2].mamba_value is not None
    assert chain[2].mamba_value != live_id and chain[2].mamba_value not in donated
    assert pool.num_free_slots == free_before + 2           # 3 req slots freed, 1 clone alloc'd
    cm.check_integrity()

    # phase B: a re-prefill of the same prompt; its aligned finish live-donate dedups
    pending_b = PendingReq(1, torch.arange(192, dtype=torch.int32), SamplingParams(max_tokens=1))
    adder = PrefillAdder(token_budget=128, reserved_size=0, cache_manager=cm, table_manager=tm)
    req_b = adder.try_add_one(pending_b)
    assert req_b is not None and not isinstance(req_b, ChunkedReq) and req_b.cached_len == 128
    cm.allocate_paged([req_b])
    req_b.complete_one()                                    # cached_len 128 -> 192
    free_before_b = pool.num_free_slots
    cm.cache_req(req_b, finished=True)                      # the live-donate dedups at 192
    assert pool.num_free_slots == free_before_b + 3         # live + pair freed; nothing donated
    assert [n.mamba_value for n in _snapshot_chain(cm)] == donated + [chain[2].mamba_value]   # untouched
    assert req_b.mamba_ping_pong is None and req_b.linear_slot_idx is None
    cm.check_integrity()


def test_naive_cache_does_not_align_prefill_chunks():
    """The alignment hook is hybrid-only; every other cache keeps the raw budget chunk."""
    from freetoken.scheduler.prefill import ChunkedReq, PrefillAdder
    from freetoken.scheduler.table import TableManager
    from freetoken.scheduler.utils import PendingReq

    pt = torch.zeros(4, 512, dtype=torch.int32)
    cm = CacheManager(64, 64, pt, "radix")
    assert cm.prefill_chunk_align == 1
    tm = TableManager(max_running_reqs=4, page_table=pt)
    adder = PrefillAdder(token_budget=100, reserved_size=0, cache_manager=cm, table_manager=tm)
    pending = PendingReq(0, torch.arange(300, dtype=torch.int32), SamplingParams(max_tokens=1))
    chunk1 = adder.try_add_one(pending)
    assert isinstance(chunk1, ChunkedReq) and chunk1.extend_len == 100

    # B3-1: the is_hybrid gate keeps the per-chunk donation hybrid-only: a non-hybrid
    # continuation must not cache_req at creation (the drain skip still covers it), so the
    # tree stays empty and the admission handle is not rebound until the final chunk.
    cm.allocate_paged([chunk1])
    chunk1.complete_one()                             # cached_len 0 -> 100
    pending.chunked_req = chunk1
    adder = PrefillAdder(token_budget=100, reserved_size=0, cache_manager=cm, table_manager=tm)
    chunk2 = adder.try_add_one(pending)
    assert isinstance(chunk2, ChunkedReq) and chunk2.cached_len == 100
    assert len(cm.prefix_cache.root_node.children) == 0    # no insert at continuation creation
    assert chunk2.cache_handle is chunk1.cache_handle      # the admission handle, not rebound

    # the final chunk commits at its own drain: only then does the tree fill
    cm.allocate_paged([chunk2])
    chunk2.complete_one()                             # cached_len 100 -> 200
    pending.chunked_req = chunk2
    adder = PrefillAdder(token_budget=100, reserved_size=0, cache_manager=cm, table_manager=tm)
    final = adder.try_add_one(pending)                # [200,300): the last chunk is a plain Req
    assert not isinstance(final, ChunkedReq)
    cm.allocate_paged([final])
    final.complete_one()                              # cached_len 200 -> 300
    cm.cache_req(final, finished=False)
    assert len(cm.prefix_cache.root_node.children) == 1    # the final drain commit inserted
    assert final.cache_handle is not chunk1.cache_handle   # rebound by the final commit
    assert final.cache_handle.cached_len == 256            # align_down(300, 64)


@pytest.mark.parametrize("full_chunks", [2, 3])
@pytest.mark.parametrize("tail", [1, 59, 64, 65])
@pytest.mark.parametrize("finish_early", [False, True])
def test_chunked_prefill_retains_resumable_snapshot(monkeypatch, full_chunks, tail, finish_early):
    """A short final extend must retain the prior state; a longer one must replace it."""
    import freetoken.core as core
    from freetoken.attention.linear import build_fla_metadata
    from freetoken.scheduler.decode import DecodeManager
    from freetoken.scheduler.prefill import ChunkedReq, PrefillManager
    from freetoken.scheduler.scheduler import Scheduler
    from freetoken.scheduler.table import TableManager
    from freetoken.scheduler.utils import PendingReq

    pool = _pool()
    pt = torch.zeros(2, 1024, dtype=torch.int32)
    cm = CacheManager(32, 64, pt, "hybrid_radix", linear_state_pool=pool)
    tm = TableManager(max_running_reqs=1, page_table=pt)
    pm = PrefillManager(cm, tm, DecodeManager(page_size=64))
    monkeypatch.setattr(core, "_GLOBAL_CTX", core.Context(page_size=64, linear_state_pool=pool))
    prompt = torch.arange(full_chunks * 128 + tail, dtype=torch.int32)
    pm.pending_list = [PendingReq(0, prompt, SamplingParams(max_tokens=8))]
    last_batch = None
    states = {}
    final = None

    while pm.runnable or last_batch is not None:
        # Match overlap ordering: prepare the continuation before draining the prior chunk.
        batch = pm.schedule_next_batch(128)
        if batch is not None:
            batch.padded_reqs = batch.reqs
            cm.allocate_paged(batch.reqs)
            req = batch.reqs[0]
            meta = build_fla_metadata(batch, torch.device("cpu"))
            if meta.track_dst is not None:
                slot = meta.track_dst.item()
                value = len(states) + 1
                states[req.mamba_last_track_seqlen] = value
                pool.recurrent_states[:, slot].fill_(value)
                pool.conv_states[:, slot].fill_(value)
            pool.recurrent_states[:, req.linear_slot_idx].fill_(-9)
            pool.conv_states[:, req.linear_slot_idx].fill_(-9)
            req.complete_one()
        if last_batch is not None:
            previous = last_batch.reqs[0]
            if not isinstance(previous, ChunkedReq):
                final = previous
                if not finish_early:
                    cm.cache_req(final, finished=False)
        last_batch = batch

    # Before the finish path can donate an aligned live state, require the frozen checkpoint.
    expected = full_chunks * 128 - 64 if tail <= 64 else full_chunks * 128 + 64
    if not finish_early:
        match = cm.match_req(_pend(prompt.tolist() + [999]))
        assert match.cuda_handle.cached_len == expected
        assert torch.all(pool.recurrent_states[:, match.mamba_value] == states[expected])
        assert torch.all(pool.conv_states[:, match.mamba_value] == states[expected])

    # Finish/abort may occur before the prefill commit; use the real idempotent cleanup path.
    stub = SimpleNamespace(cache_manager=cm, table_manager=tm, _prefill_start={})
    Scheduler._free_req_resources(stub, final)
    Scheduler._free_req_resources(stub, final)
    match = cm.match_req(_pend(prompt.tolist() + [999]))
    expected_finish = len(prompt) if tail == 64 else expected
    assert match.cuda_handle.cached_len == expected_finish
    value = -9 if tail == 64 else states[expected]
    assert torch.all(pool.recurrent_states[:, match.mamba_value] == value)
    assert torch.all(pool.conv_states[:, match.mamba_value] == value)
    assert tm.available_size == 1
    cm.check_integrity()
    assert pool.num_free_slots + cm.prefix_cache.mamba_evictable_size == pool.num_slots - 1
    cm.ensure_mamba_slots(pool.num_slots - 1)
    cm.check_integrity()
    assert pool.num_free_slots == pool.num_slots - 1
    assert len(cm.free_slots) == cm.num_pages


def test_pool_sizing_covers_4mr_floor():
    """C6: pool must reserve the 4-slot-per-request non-evictable floor even at a tiny ratio."""
    from types import SimpleNamespace
    from freetoken.kvcache.linear_state_pool import _linear_pool_num_slots
    for mr in (1, 8, 64):
        c = SimpleNamespace(max_running_req=mr, cache_type="hybrid_radix",
                            linear_state_cache_ratio=0.1)
        assert _linear_pool_num_slots(c) >= 4 * mr + 1, (mr, _linear_pool_num_slots(c))


def test_chunk_commit_dedup_refreshes_boundary_lru():
    """Fix-3, scheduler level: a repeat chunked-prefill that dedups at an existing boundary
    re-validates it, so the deepest shared boundary leaves the commit walk's uniform tic
    strictly behind the shallower boundary (pre-fix they tie and the LRU order is arbitrary)."""
    pool = _pool()
    pt = torch.zeros(4, 512, dtype=torch.int32)
    cm = CacheManager(64, 64, pt, "hybrid_radix", linear_state_pool=pool)

    ids129 = list(range(1, 130))            # Req needs device_len > cached_len

    def _commit(uid, table_idx, cached_len, track_len, pages_base):
        mr = cm.match_req(_pend(ids129[:cached_len]))
        live, pp = pool.alloc(1)[0], tuple(pool.alloc(2))
        pt[table_idx, :cached_len] = torch.arange(pages_base, pages_base + cached_len,
                                                  dtype=torch.int32)
        req = Req(input_ids=torch.tensor(ids129, dtype=torch.int32), table_idx=table_idx,
                  cached_len=cached_len, output_len=1, uid=uid,
                  sampling_params=SamplingParams(), cache_handle=mr.cuda_handle)
        req.linear_slot_idx, req.mamba_ping_pong = live, pp
        req.mamba_next_track_idx = 1               # frozen = pp[0]
        req.mamba_last_track_seqlen = cached_len   # the x64 boundary this chunk tracked
        cm.lock(mr.cuda_handle)
        cm.cache_req(req, finished=False)
        return mr, req

    _commit(0, 0, 64, 64, 100)                     # req 1: chunk commit at boundary 64 (B1)
    _commit(1, 1, 128, 128, 200)                   # req 2: chunk commit at boundary 128 (B2)
    _commit(2, 2, 128, 128, 300)                   # req 3: same prefix -> DEDUP at B2

    b1 = next(iter(cm.prefix_cache.root.children.values()))
    b2 = next(iter(b1.children.values()))
    assert b1.length == 64 and b2.length == 64 and b2.mamba_value is not None
    assert b2.timestamp > b1.timestamp

    # the boundary snapshots stay live (mamba-locked by the open chunk reqs) across the
    # repeated chunked-prefill requests, so a re-prefill of the same prefix still resumes
    assert b1.mamba_value is not None and b2.mamba_value is not None
    mr = cm.match_req(_pend(ids129))          # match_req strips the last token: 128 tokens
    assert mr.cuda_handle.cached_len == 128 and mr.mamba_value is not None
    cm.prefix_cache.check_integrity()


def test_pool_exhaustion_evicts_the_stale_validated_boundary():
    """Amended Fix-3, scheduler level: a 7-slot pool (6 allocatable -- the pool keeps one
    padding sink) makes ensure_mamba_slots actually exhaust at req4's chunk commit (the
    real commit order: insert -> unlock -> match -> lock -> ensure). At the ensure B64 was
    last validated at req3's dedup, the off-path decoy D1 at reqD, and B128 by req4's own
    admission match -- all three share the commit walk's tic. Post-fix the victim is the
    stalest-VALIDATED one (B64) and the just-re-validated B128 survives; pre-fix the decoy's
    stale walk-tic timestamp makes IT the victim and the ordering is lost (fails-before)."""
    pool = _pool(num_slots=7)
    pt = torch.zeros(5, 512, dtype=torch.int32)
    cm = CacheManager(64, 64, pt, "hybrid_radix", linear_state_pool=pool)
    ids193 = list(range(1, 194))            # Req needs device_len > cached_len
    idsD = list(range(1000, 1065))          # decoy branch: different first page, off-path

    def _commit(uid, table_idx, ids, cached_len, track_len, pages_base, finish):
        mr = cm.match_req(_pend(ids[:cached_len + 1]))
        live, pp = pool.alloc(1)[0], tuple(pool.alloc(2))
        pt[table_idx, :cached_len] = torch.arange(pages_base, pages_base + cached_len,
                                                  dtype=torch.int32)
        req = Req(input_ids=torch.tensor(ids[:cached_len + 1], dtype=torch.int32),
                  table_idx=table_idx, cached_len=cached_len, output_len=1, uid=uid,
                  sampling_params=SamplingParams(), cache_handle=mr.cuda_handle)
        req.linear_slot_idx, req.mamba_ping_pong = live, pp
        req.mamba_next_track_idx = 1               # frozen = pp[0]
        req.mamba_last_track_seqlen = track_len
        cm.lock(mr.cuda_handle)
        cm.cache_req(req, finished=finish)

    _commit(0, 0, ids193, 64, 64, 100, True)       # B64: live slot donated at the x64 boundary
    _commit(1, 1, ids193, 128, 128, 200, True)     # B128 born; its finish dedup validates it
    _commit(2, 2, ids193, 64, 64, 300, True)       # repeat @64: B64 dedup-validated (newest pre-r4)
    _commit(3, 3, idsD, 64, 64, 600, True)         # off-path decoy D1: validated once, never re-walked
    _commit(4, 4, ids193, 192, 192, 400, False)    # fresh B192 -> ensure_mamba_slots must evict

    b64 = next(iter(cm.prefix_cache.root.children.values()))
    b128 = next(iter(b64.children.values()))
    b192 = next(iter(b128.children.values()))
    d1 = next(c for c in cm.prefix_cache.root.children.values() if c is not b64)
    assert (b64.length, b128.length, b192.length, d1.length) == (64, 64, 64, 64)
    assert b64.mamba_value is None                 # the stalest-VALIDATED tie-class peer died
    assert b128.mamba_value is not None            # re-validated by req4's admission match
    assert d1.mamba_value is not None              # validated after B64: survives victim #1
    assert b192.mamba_value is not None            # the just-donated tip is locked, untouched
    mr = cm.match_req(_pend(ids193))               # match_req strips the last token: 192
    assert mr.cuda_handle.cached_len == 192 and mr.mamba_value == b192.mamba_value
    cm.prefix_cache.check_integrity()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))


# ---------------------------------------------------------------- host KV tier

HOST_PAGE = 4
QSA_LAYERS, QSA_TOTAL_LAYERS, QSA_IDX_LAYERS, QSA_IDX_DIM, QSA_RATIO = 2, 4, 2, 6, 2
QSA_ROWS = HOST_PAGE // QSA_RATIO
QSA_PAGES = 8


def _tp(monkeypatch):
    """MHAKVCache reads the TP size when it sizes the local K/V slab."""
    from freetoken.distributed.info import DistributedInfo

    monkeypatch.setattr("freetoken.kvcache.mha_pool.get_tp_info",
                        lambda: DistributedInfo(rank=0, size=1))


def _qsa_pool(mrope: bool = False):
    from freetoken.kvcache.qsa_pool import QSAKVCache
    return QSAKVCache(
        num_kv_heads=1, num_layers=QSA_TOTAL_LAYERS, head_dim=4, num_pages=QSA_PAGES,
        page_size=HOST_PAGE, dtype=torch.bfloat16, device=torch.device("cpu"),
        index_head_dim=QSA_IDX_DIM, num_index_layers=QSA_IDX_LAYERS, index_ratio=QSA_RATIO,
        num_req_slots=2, layer_ids=(1, 3), mrope=mrope,
    )


def _fill(qsa, page, seed):
    kv = qsa.page_kv_view(page)
    kv[0].fill_(seed)
    kv[1].fill_(-seed)
    for layer in range(QSA_IDX_LAYERS):
        qsa.cmp_k_cache(layer)[page * QSA_ROWS:(page + 1) * QSA_ROWS].fill_(seed * 10 + layer)


def _snapshot(qsa, page):
    return (
        qsa.page_kv_view(page).clone(),
        torch.stack([qsa.cmp_k_cache(l)[page * QSA_ROWS:(page + 1) * QSA_ROWS]
                     for l in range(QSA_IDX_LAYERS)]).clone(),
    )


def _walk_leaf(cache):
    node = cache.root
    while node.children:
        node = next(iter(node.children.values()))
    return node


def test_host_tier_restores_a_real_qsa_prefix_bit_exactly(monkeypatch):
    """The whole seam: spill on evict, match, restore into DIFFERENT pages, same bytes."""
    _tp(monkeypatch)
    monkeypatch.setenv("FREETOKEN_KV_HOST_TIER_PAGES", "4")
    qsa, lsp = _qsa_pool(), _pool()
    cm = CacheManager(QSA_PAGES, HOST_PAGE, torch.zeros(2, 64, dtype=torch.int32),
                      "hybrid_radix", linear_state_pool=lsp, swa_pool=qsa)

    assert cm._host_bridge is not None, "the opt-in variable must build the bridge"
    assert cm._host_tier.geometry.index_layers == QSA_IDX_LAYERS
    assert cm._host_tier.geometry.rope_pos is False
    assert cm._host_tier.geometry.num_layers == QSA_LAYERS

    for page, seed in ((0, 11), (1, 22)):
        _fill(qsa, page, seed)
    before = {page: _snapshot(qsa, page) for page in (0, 1)}

    ids = torch.arange(2 * HOST_PAGE, dtype=torch.int32)
    values = (torch.arange(2 * HOST_PAGE, dtype=torch.int32) // HOST_PAGE) * HOST_PAGE
    cache = cm.prefix_cache
    cache.insert(ids, values, mamba_value=1)
    cache.evict_full(2 * HOST_PAGE)

    assert cache.host_resident_size == 2 * HOST_PAGE, "the leaf is spilled, not dropped"
    assert cm._host_tier.resident_entries == 1
    held = cm._host_alloc_pages(2)                     # another request now owns pages 0, 1
    assert [int(v) for v in held] == [0, 1]

    m = cache.match_prefix(ids)

    assert m.cached_len == 2 * HOST_PAGE and m.mamba_value == 1
    assert cache.host_resident_size == 0
    restored = [int(v) // HOST_PAGE for v in _walk_leaf(cache).value[::HOST_PAGE]]
    assert restored == [2, 3], "restored into the pages that were actually free"
    for got, src in zip(restored, (0, 1)):
        kv, idx = _snapshot(qsa, got)
        assert torch.equal(kv, before[src][0]), "K/V byte-for-byte"
        assert torch.equal(idx, before[src][1]), "index shadow byte-for-byte"


def test_an_off_deployment_builds_no_tier_and_keeps_the_old_eviction(monkeypatch):
    _tp(monkeypatch)
    monkeypatch.delenv("FREETOKEN_KV_HOST_TIER_PAGES", raising=False)
    qsa, lsp = _qsa_pool(), _pool()
    cm = CacheManager(QSA_PAGES, HOST_PAGE, torch.zeros(2, 64, dtype=torch.int32),
                      "hybrid_radix", linear_state_pool=lsp, swa_pool=qsa)

    assert cm._host_bridge is None and cm._host_tier is None
    assert cm.prefix_cache.host_spill is None
    assert cm.prefix_cache.host_materialize is None

    ids = torch.arange(2 * HOST_PAGE, dtype=torch.int32)
    values = (torch.arange(2 * HOST_PAGE, dtype=torch.int32) // HOST_PAGE) * HOST_PAGE
    cm.prefix_cache.insert(ids, values, mamba_value=1)
    er = cm.prefix_cache.evict_full(2 * HOST_PAGE)

    assert cm.prefix_cache.host_resident_size == 0
    assert torch.equal(er.kv_indices, values), "the pages come back for the caller to free"


def test_rebuild_clears_the_host_tier_before_replacing_the_tree(monkeypatch):
    _tp(monkeypatch)
    monkeypatch.setenv("FREETOKEN_KV_HOST_TIER_PAGES", "4")
    qsa, lsp = _qsa_pool(), _pool()
    cm = CacheManager(QSA_PAGES, HOST_PAGE, torch.zeros(2, 64, dtype=torch.int32),
                      "hybrid_radix", linear_state_pool=lsp, swa_pool=qsa)
    ids = torch.arange(2 * HOST_PAGE, dtype=torch.int32)
    values = (torch.arange(2 * HOST_PAGE, dtype=torch.int32) // HOST_PAGE) * HOST_PAGE
    cm.prefix_cache.insert(ids, values, mamba_value=1)
    cm.prefix_cache.evict_full(2 * HOST_PAGE)
    assert cm._host_tier.resident_entries == 1

    cm.rebuild(QSA_PAGES, torch.zeros(2, 64, dtype=torch.int32))

    assert cm._host_tier.resident_entries == 0, "entries describe the discarded tree"
    assert cm._host_tier.resident_pages == 0
    assert cm.prefix_cache.host_resident_size == 0


def test_the_rope_rows_move_with_the_page_on_a_mrope_pool(monkeypatch):
    """mrope's per-token 3-axis position is slot-addressed state: leave it behind on a restore
    and every restored token ropes at the position of whoever owns the page now."""
    _tp(monkeypatch)
    monkeypatch.setenv("FREETOKEN_KV_HOST_TIER_PAGES", "4")
    qsa, lsp = _qsa_pool(mrope=True), _pool()
    cm = CacheManager(QSA_PAGES, HOST_PAGE, torch.zeros(2, 64, dtype=torch.int32),
                      "hybrid_radix", linear_state_pool=lsp, swa_pool=qsa)
    assert cm._host_tier.geometry.rope_pos is True

    for page, seed in ((0, 5), (1, 9)):
        qsa.rope_positions[page * HOST_PAGE:(page + 1) * HOST_PAGE] = torch.tensor(
            [[seed, page, axis] for axis in range(HOST_PAGE)], dtype=torch.int32)
    before = {page: qsa.rope_positions[page * HOST_PAGE:(page + 1) * HOST_PAGE].clone()
              for page in (0, 1)}

    ids = torch.arange(2 * HOST_PAGE, dtype=torch.int32)
    values = (torch.arange(2 * HOST_PAGE, dtype=torch.int32) // HOST_PAGE) * HOST_PAGE
    cache = cm.prefix_cache
    cache.insert(ids, values, mamba_value=1)
    cache.evict_full(2 * HOST_PAGE)
    held = cm._host_alloc_pages(2)
    assert [int(v) for v in held] == [0, 1]

    assert cache.match_prefix(ids).cached_len == 2 * HOST_PAGE

    restored = [int(v) // HOST_PAGE for v in _walk_leaf(cache).value[::HOST_PAGE]]
    assert restored == [2, 3]
    for got, src in zip(restored, (0, 1)):
        assert torch.equal(qsa.rope_positions[got * HOST_PAGE:(got + 1) * HOST_PAGE], before[src])


def _host_cache(monkeypatch, pages: str):
    _tp(monkeypatch)
    monkeypatch.setenv("FREETOKEN_KV_HOST_TIER_PAGES", pages)
    qsa, lsp = _qsa_pool(), _pool()
    cm = CacheManager(QSA_PAGES, HOST_PAGE, torch.zeros(2, 64, dtype=torch.int32),
                      "hybrid_radix", linear_state_pool=lsp, swa_pool=qsa)
    return qsa, cm


def _page_prefix(first_token: int, n_pages: int = 1):
    ids = torch.arange(first_token, first_token + n_pages * HOST_PAGE, dtype=torch.int32)
    values = (torch.arange(n_pages * HOST_PAGE, dtype=torch.int32) // HOST_PAGE) * HOST_PAGE
    return ids, values


def test_the_tier_s_own_lru_eviction_unlinks_the_node_it_drops(monkeypatch):
    """The tier frees its own room. That victim's node must be unlinked, or the tree keeps a
    host-resident leaf whose bytes are gone and every later match reads a prefix that is not
    there."""
    qsa, cm = _host_cache(monkeypatch, "1")
    cache = cm.prefix_cache
    a_ids, a_pages = _page_prefix(100)
    cache.insert(a_ids, a_pages, mamba_value=1)
    cache.evict_full(HOST_PAGE)
    assert cache.host_resident_size == HOST_PAGE and cm._host_tier.resident_entries == 1

    b_ids, b_pages = _page_prefix(200)
    cache.insert(b_ids, b_pages, mamba_value=2)
    cache.evict_full(HOST_PAGE)          # no room: the tier evicts A to take B

    assert cm._host_tier.resident_entries == 1
    assert cache.host_resident_size == HOST_PAGE, "A's length must leave the cache's account"
    cache.check_integrity()
    hit = cache.match_prefix(a_ids)
    assert hit.cached_len == 0, "A's prefix is gone, not silently readable"
    assert cache.match_prefix(b_ids).cached_len == HOST_PAGE


def test_a_node_larger_than_the_tier_declines_instead_of_crashing(monkeypatch):
    """Spill is a question, not an order: a tier too small for this node must answer 'no' and
    let the cache evict as it always did. An assert here aborts the engine on the first
    eviction of any prefix longer than the configured tier."""
    qsa, cm = _host_cache(monkeypatch, "1")          # one page of host tier
    cache = cm.prefix_cache
    ids, values = _page_prefix(300, n_pages=2)       # ...against a two-page node
    cache.insert(ids, values, mamba_value=3)

    erased = cache.evict_full(2 * HOST_PAGE)

    assert cache.host_resident_size == 0
    cache.check_integrity()
    assert [int(v) for v in erased.kv_indices] == [int(v) for v in values], "pages come back"
    assert cm._host_tier.resident_entries == 0


def test_an_unrelated_eviction_leaves_the_host_resident_node_alone(monkeypatch):
    """Eviction must never reclaim a host-resident node: its KV is in the tier, so freeing the
    snapshot would strand it, and the tier -- not the tree -- is what reclaims its pages. Pin it
    because both eviction paths in this file are inherited from an upstream that moves."""
    qsa, cm = _host_cache(monkeypatch, "4")
    cache = cm.prefix_cache
    host_ids, host_pages = _page_prefix(100)
    cache.insert(host_ids, host_pages, mamba_value=1)
    cache.evict_full(HOST_PAGE)
    assert cache.host_resident_size == HOST_PAGE and cm._host_tier.resident_entries == 1

    gpu_ids, gpu_pages = _page_prefix(200)
    cache.insert(gpu_ids, gpu_pages, mamba_value=2)
    # The host-resident node holds an unlocked snapshot too, so it is a candidate for this
    # reclaim unless the eviction skips it -- and taking its snapshot would strand its KV.
    erased = cache.evict_mamba(1)

    assert [int(v) for v in erased.kv_indices] == [int(v) for v in gpu_pages]
    assert cache.host_resident_size == HOST_PAGE, "the host-resident node kept its account"
    assert cm._host_tier.resident_entries == 1, "and the tier kept its entry"
    cache.check_integrity()
    assert cache.match_prefix(gpu_ids).cached_len == 0
    assert cache.match_prefix(host_ids).cached_len == HOST_PAGE, "still restorable"
    assert cache.host_resident_size == 0, "materializing it is what retires the entry"


def _on_page(first_token: int, page: int):
    ids = torch.arange(first_token, first_token + HOST_PAGE, dtype=torch.int32)
    return ids, torch.full((HOST_PAGE,), page * HOST_PAGE, dtype=torch.int32)


def test_a_tier_lru_drop_returns_what_it_reclaims(monkeypatch):
    """The tier drops entries on its own LRU. Everything the unlinking frees -- the node's GDN
    snapshot slot and any tombstone parent's pages -- belongs to the manager's pools, or the
    next idle integrity check finds them missing."""
    qsa, cm = _host_cache(monkeypatch, "4")
    cache = cm.prefix_cache
    # Pages and state slots really come out of their pools: ids the tree invented would let the
    # free-list assertions pass while the accounting they exercise stays wrong.
    page_parent, page_child = (int(p) for p in cm._host_alloc_pages(2))
    parent_ids, parent_pages = _on_page(100, page_parent)
    child_ids, child_pages = _on_page(100 + HOST_PAGE, page_child)
    mamba_parent, mamba_child = cm.linear_state_pool.alloc(2)
    cache.insert(parent_ids, parent_pages, mamba_value=mamba_parent)
    cache.insert(torch.cat([parent_ids, child_ids]),
                 torch.cat([parent_pages, child_pages]), mamba_value=mamba_child)
    cache.evict_mamba(1)                     # the internal parent becomes a KV-only tombstone
    spilled = cache.evict_full(HOST_PAGE)    # ...and the child leaf spills (it holds a snapshot)
    cm._free(spilled.kv_indices)             # its pages go back, its bytes to the tier
    assert cache.host_resident_size == HOST_PAGE and cm._host_tier.resident_entries == 1

    slots_before, pages_before = cm.linear_state_pool.num_free_slots, len(cm.free_slots)
    key = next(iter(cache._host_nodes))
    cm._host_tier_dropped(key)                # exactly what the tier's LRU callback does

    assert cache.host_resident_size == 0
    assert cm.linear_state_pool.num_free_slots == slots_before + 1, "the snapshot slot comes back"
    assert len(cm.free_slots) == pages_before + 1, "the tombstone parent's pages come back"
    cm.check_integrity()


def test_a_refused_materialize_on_the_match_path_returns_its_resources(monkeypatch):
    """The walk retires a node the tier will not serve again and parks what the unlink frees;
    the manager drains it before it touches a pool, or the next idle check finds it missing."""
    qsa, cm = _host_cache(monkeypatch, "4")
    cache = cm.prefix_cache
    page = int(cm._host_alloc_pages(1)[0])
    ids, values = _on_page(100, page)
    slot = cm.linear_state_pool.alloc(1)[0]
    cache.insert(ids, values, mamba_value=slot)
    cm._free(cache.evict_full(HOST_PAGE).kv_indices)   # the node spills: pages back, bytes in tier
    assert cache.host_resident_size == HOST_PAGE

    # The bridge's method is captured by the cache at construction, so cut the seam the walk
    # itself uses: what the refusal costs is the point here, not how the bridge words it.
    monkeypatch.setattr(cache, "_materialize", lambda node: False)
    slots_before, pages_before = cm.linear_state_pool.num_free_slots, len(cm.free_slots)
    assert cache.match_prefix(ids).cached_len == 0
    cm.ensure_mamba_slots(1)                           # the drain site is what returns it

    assert cache.host_resident_size == 0
    assert cm.linear_state_pool.num_free_slots == slots_before + 1, "the snapshot slot is back"
    assert len(cm.free_slots) == pages_before, "the node held no pages of its own"
    cm.check_integrity()


def test_the_manager_reports_the_tier_only_when_it_is_enabled(monkeypatch):
    """None, not zeros: an operator must be able to tell 'off' from 'on but idle'."""
    _qsa, off = _host_cache(monkeypatch, "0")
    assert off.host_tier_stats() is None

    _qsa, on = _host_cache(monkeypatch, "4")
    snap = on.host_tier_stats()
    assert snap is not None and snap["spills"] == 0, "enabled, nothing spilled yet"
    assert snap["capacity_pages"] == 4
    ids, values = _page_prefix(100)
    on.prefix_cache.insert(ids, values, mamba_value=on.linear_state_pool.alloc(1)[0])
    on._free(on.prefix_cache.evict_full(HOST_PAGE).kv_indices)   # spill it: pages back, bytes host
    assert on.host_tier_stats()["spills"] == 1, "the counter follows the real spill"
    assert on.host_tier_stats()["resident_entries"] == 1


def test_rebuild_drops_parked_frees_instead_of_returning_them_twice(monkeypatch):
    """rebuild resets every pool wholesale, so what the discarded cache had parked must go with
    it: draining those pages and slots into the fresh lists would count them twice."""
    qsa, cm = _host_cache(monkeypatch, "4")
    cache = cm.prefix_cache
    page = int(cm._host_alloc_pages(1)[0])
    ids, values = _on_page(100, page)
    slot = cm.linear_state_pool.alloc(1)[0]
    cache.insert(ids, values, mamba_value=slot)
    cm._free(cache.evict_full(HOST_PAGE).kv_indices)
    monkeypatch.setattr(cache, "_materialize", lambda node: False)
    assert cache.match_prefix(ids).cached_len == 0
    assert cache._host_pending_mamba == [slot] and len(cache._host_pending_kv) >= 0, "parked"

    cm.rebuild(QSA_PAGES, torch.zeros(2, 64, dtype=torch.int32))

    assert cm.prefix_cache is not cache, "rebuild replaces the tree"
    assert cm.prefix_cache._host_pending_mamba == [], "the new tree has nothing parked"
    assert cache._host_pending_mamba == [slot], (
        "the discarded cache keeps its own queue and nothing drains it: those pages are already "
        "in the freshly reset free lists, so returning them again would count them twice"
    )
    assert cm.host_tier_stats() is not None, "the tier survives rebuild, sized from the new pools"
    cm.check_integrity()


def test_the_per_step_drain_site_also_returns_what_a_match_parked(monkeypatch):
    """ensure_mamba_slots is not the only drain site: the ordinary per-step allocation path must
    return parked resources too, or a long run with no mamba pressure would accumulate them."""
    qsa, cm = _host_cache(monkeypatch, "4")
    cache = cm.prefix_cache
    page = int(cm._host_alloc_pages(1)[0])
    ids, values = _on_page(140, page)
    slot = cm.linear_state_pool.alloc(1)[0]
    cache.insert(ids, values, mamba_value=slot)
    cm._free(cache.evict_full(HOST_PAGE).kv_indices)
    monkeypatch.setattr(cache, "_materialize", lambda node: False)
    assert cache.match_prefix(ids).cached_len == 0

    slots_before = cm.linear_state_pool.num_free_slots
    cm.allocate_paged([])                              # the per-step site

    assert cm.linear_state_pool.num_free_slots == slots_before + 1
    cm.check_integrity()


def test_a_spill_keeps_the_gdn_snapshot_so_the_prefix_stays_resumable(monkeypatch):
    """A hybrid prefix resumes only from a live snapshot on its node, so moving the KV to the
    host must not also release the GDN state that decodes it."""
    qsa, cm = _host_cache(monkeypatch, "4")
    cache = cm.prefix_cache
    page = int(cm._host_alloc_pages(1)[0])
    ids, values = _on_page(160, page)
    slot = cm.linear_state_pool.alloc(1)[0]
    cache.insert(ids, values, mamba_value=slot)
    free_before = cm.linear_state_pool.num_free_slots

    evicted = cache.evict_full(HOST_PAGE)              # KV pages move to the host tier
    cm._free(evicted.kv_indices)                       # ...and the scheduler returns the pages

    assert cm.host_tier_stats()["resident_entries"] == 1
    assert cm.linear_state_pool.num_free_slots == free_before, (
        "the GDN snapshot must stay allocated while its KV is host-resident"
    )
    cm.check_integrity()


def test_a_tier_too_big_for_the_host_leaves_serving_untouched(monkeypatch):
    """Asking for more host pages than the machine can address (or than it has) must come back as
    "tier off", not as a startup abort: it is an optional hit-rate layer."""
    _qsa, cm = _host_cache(monkeypatch, str(1 << 46))   # 2**46 pages: unmappable on purpose

    assert cm.host_tier_stats() is None, "the tier refused to build and the manager carried on"
    assert cm.prefix_cache is not None
    cm.check_integrity()


def _per_token_values(pages):
    """The pool's per-token value form: every token names its page's first slot."""
    starts = torch.tensor([p * HOST_PAGE for p in pages], dtype=torch.int32)
    return (starts.unsqueeze(1) + torch.arange(HOST_PAGE, dtype=torch.int32)).flatten()


def test_a_two_page_prefix_comes_back_on_the_pages_its_slots_belong_to(monkeypatch):
    """Multi-page entries: one page per entry cannot see a slot-order slip, and a prefix whose
    pages are handed back in the wrong order is the quietest way to serve wrong bytes."""
    qsa, cm = _host_cache(monkeypatch, "4")
    cache = cm.prefix_cache
    pages = (0, 1)
    written = []
    for i, page in enumerate(pages):                 # distinct bytes per page, not per entry
        _fill(qsa, page, 5 + 3 * i)
        written.append(_snapshot(qsa, page))
    ids = torch.arange(200, 200 + len(pages) * HOST_PAGE, dtype=torch.int32)
    cache.insert(ids, _per_token_values(pages), mamba_value=1)

    cache.evict_full(len(pages) * HOST_PAGE)
    assert cache.host_resident_size == len(pages) * HOST_PAGE, "spilled, not dropped"
    cm._host_alloc_pages(len(pages))                 # the device reuses the original pages

    hit = cache.match_prefix(ids)

    assert hit.cached_len == len(ids)
    served = [int(v) // HOST_PAGE for v in hit.kv_indices[::HOST_PAGE]]
    assert len(served) == len(pages)
    for page, want in zip(served, written):
        got = _snapshot(qsa, page)
        for a, b in zip(got, want):
            assert torch.equal(a, b), f"page {page} of the restored prefix holds other bytes"


def test_a_random_walk_of_evictions_and_matches_never_serves_the_wrong_bytes(monkeypatch):
    """Integration fuzz: the tier on, random inserts/evictions, every served prefix checked.

    The bridge tests pin the copy paths; this pins the thing the red line actually cares about --
    a prefix that the cache answers for, after any number of spills and restores, must hand back
    pages whose bytes are the ones that were written for it. Page supply goes through the
    manager's own free list, so ownership stays consistent by construction (check_integrity
    asserts free + cached == total), and the walk ends by proving the tier really was used.
    """
    rng = random.Random(20260922)
    qsa, cm = _host_cache(monkeypatch, "3")
    cache = cm.prefix_cache
    docs: dict[int, tuple[torch.Tensor, tuple[int, ...], tuple]] = {}
    serial = 0

    def content(page: int, seed: int) -> tuple:
        _fill(qsa, page, seed)
        return _snapshot(qsa, page)

    def check_every_doc(where: str) -> None:
        cm.check_integrity()
        for doc, (ids, pages_at_insert, written) in docs.items():
            hit = cache.match_prefix(ids)
            if hit.cached_len != len(ids):
                continue                      # evicted for good; nothing to serve
            served = [int(v) // HOST_PAGE for v in hit.kv_indices[::HOST_PAGE]]
            for page, want in zip(served, written):
                got = _snapshot(qsa, page)
                for a, b in zip(got, want):
                    assert torch.equal(a, b), (
                        f"{where}: doc {doc} served page {page} with other bytes "
                        f"(inserted on {pages_at_insert})\nwalk: {log}"
                    )

    log: list[str] = []
    for _ in range(60):
        cm.check_integrity()
        roll = rng.random()
        if roll < 0.45:
            # Multi-page entries matter: with one page per entry a slot-order slip is invisible.
            free = cm._host_alloc_pages(rng.choice((1, 2)))   # only taken when this step uses it
            if not free.numel():
                continue
            slots = cm.linear_state_pool.alloc(1)   # a claimed GDN slot must be a real one
            if not slots:
                cm._free(free * HOST_PAGE)           # no slot to donate: give the pages back
                continue
            pages = tuple(int(p) for p in free)
            seed = 1 + serial * 3
            base = 10000 + 100 * serial
            ids = torch.arange(base, base + len(pages) * HOST_PAGE, dtype=torch.int32)
            written = tuple(content(page, seed + i) for i, page in enumerate(pages))
            log.append(f"insert doc {serial} pages={pages}")
            cache.insert(ids, _per_token_values(pages), mamba_value=slots[0])
            docs[serial] = (ids, pages, written)
            serial += 1
        elif roll < 0.80:
            log.append("evict")
            erased = cache.evict_full(HOST_PAGE)
            cm._free(erased.kv_indices)
            if erased.mamba_slots:
                cm.linear_state_pool.free(erased.mamba_slots)
        check_every_doc(f"step {len(log)}")

    tiers = cm._host_tier.stats_snapshot()
    assert tiers["spills"] > 0 and tiers["restores"] > 0, (
        f"the walk must exercise the tier to prove anything: {tiers}\n{log}"
    )
    check_every_doc("final")


@pytest.mark.parametrize("full_chunks", [2, 3])
@pytest.mark.parametrize("tail", [1, 59, 64, 65])
@pytest.mark.parametrize("finish_early", [False, True])
def test_chunked_prefill_retains_resumable_snapshot(monkeypatch, full_chunks, tail, finish_early):
    """A short final extend must retain the prior state; a longer one must replace it."""
    import freetoken.core as core
    from freetoken.attention.linear import build_fla_metadata
    from freetoken.scheduler.decode import DecodeManager
    from freetoken.scheduler.prefill import ChunkedReq, PrefillManager
    from freetoken.scheduler.scheduler import Scheduler
    from freetoken.scheduler.table import TableManager
    from freetoken.scheduler.utils import PendingReq

    pool = _pool()
    pt = torch.zeros(2, 1024, dtype=torch.int32)
    cm = CacheManager(32, 64, pt, "hybrid_radix", linear_state_pool=pool)
    tm = TableManager(max_running_reqs=1, page_table=pt)
    pm = PrefillManager(cm, tm, DecodeManager(page_size=64))
    monkeypatch.setattr(core, "_GLOBAL_CTX", core.Context(page_size=64, linear_state_pool=pool))
    prompt = torch.arange(full_chunks * 128 + tail, dtype=torch.int32)
    pm.pending_list = [PendingReq(0, prompt, SamplingParams(max_tokens=8))]
    last_batch = None
    states = {}
    final = None

    while pm.runnable or last_batch is not None:
        # Match overlap ordering: prepare the continuation before draining the prior chunk.
        batch = pm.schedule_next_batch(128)
        if batch is not None:
            batch.padded_reqs = batch.reqs
            cm.allocate_paged(batch.reqs)
            req = batch.reqs[0]
            meta = build_fla_metadata(batch, torch.device("cpu"))
            if meta.track_dst is not None:
                slot = meta.track_dst.item()
                value = len(states) + 1
                states[req.mamba_last_track_seqlen] = value
                pool.recurrent_states[:, slot].fill_(value)
                pool.conv_states[:, slot].fill_(value)
            pool.recurrent_states[:, req.linear_slot_idx].fill_(-9)
            pool.conv_states[:, req.linear_slot_idx].fill_(-9)
            req.complete_one()
        if last_batch is not None:
            previous = last_batch.reqs[0]
            if not isinstance(previous, ChunkedReq):
                final = previous
                if not finish_early:
                    cm.cache_req(final, finished=False)
        last_batch = batch

    # Before the finish path can donate an aligned live state, require the frozen checkpoint.
    expected = full_chunks * 128 - 64 if tail <= 64 else full_chunks * 128 + 64
    if not finish_early:
        match = cm.match_req(_pend(prompt.tolist() + [999]))
        assert match.cuda_handle.cached_len == expected
        assert torch.all(pool.recurrent_states[:, match.mamba_value] == states[expected])
        assert torch.all(pool.conv_states[:, match.mamba_value] == states[expected])

    # Finish/abort may occur before the prefill commit; use the real idempotent cleanup path.
    stub = SimpleNamespace(cache_manager=cm, table_manager=tm, _prefill_start={})
    Scheduler._free_req_resources(stub, final)
    Scheduler._free_req_resources(stub, final)
    match = cm.match_req(_pend(prompt.tolist() + [999]))
    expected_finish = len(prompt) if tail == 64 else expected
    assert match.cuda_handle.cached_len == expected_finish
    value = -9 if tail == 64 else states[expected]
    assert torch.all(pool.recurrent_states[:, match.mamba_value] == value)
    assert torch.all(pool.conv_states[:, match.mamba_value] == value)
    assert tm.available_size == 1
    cm.check_integrity()
    assert pool.num_free_slots + cm.prefix_cache.mamba_evictable_size == pool.num_slots - 1
    cm.ensure_mamba_slots(pool.num_slots - 1)
    cm.check_integrity()
    assert pool.num_free_slots == pool.num_slots - 1
    assert len(cm.free_slots) == cm.num_pages


def test_free_slots_injection_is_rejected_loudly():
    """free_slots values land in device index tensors unchecked, so a stale/duplicated/garbage
    page must be filtered at the injection point instead of an MMU fault much later."""
    pool = _pool()
    page_table = torch.zeros(4, 64, dtype=torch.int32)
    cm = CacheManager(64, 1, page_table, "hybrid_radix", linear_state_pool=pool)

    # a legitimately allocated-and-returned page still goes back exactly once
    allocated, cm.free_slots = cm.free_slots[:4].clone(), cm.free_slots[4:]
    cm._free(allocated)
    assert len(cm.free_slots) == 64
    assert int((cm.free_slots == 0).sum()) == 1

    # out-of-pool garbage must not enter free_slots
    cm._free(torch.tensor([999, -3], dtype=torch.int32))
    assert len(cm.free_slots) == 64

    # a page that is already free being returned again is a double free: dropped
    cm._free(torch.tensor([5], dtype=torch.int32))
    assert int((cm.free_slots == 5).sum()) == 1

    # duplicates within a single free batch collapse to one entry
    cm.free_slots = cm.free_slots[cm.free_slots != 7]
    cm._free(torch.tensor([7, 7], dtype=torch.int32))
    assert int((cm.free_slots == 7).sum()) == 1
    assert len(cm.free_slots) == 64, "every filtered batch leaves the pool whole"

    # page-aligned pool rejects unaligned offsets
    cm2 = CacheManager(8, 2, torch.zeros(4, 64, dtype=torch.int32), "hybrid_radix",
                       linear_state_pool=_pool())
    cm2._free(torch.tensor([3], dtype=torch.int32))
    assert int((cm2.free_slots == 3).sum()) == 0
