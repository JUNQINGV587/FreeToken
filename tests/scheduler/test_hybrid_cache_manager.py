"""P2b integration: CacheManager hybrid path (match_req -> cache_req donate -> prefix hit).
CPU, real LinearStatePool + page_table, hand-built Reqs. Exercises the two-currency wiring
without the full scheduler/engine."""
from __future__ import annotations

from types import SimpleNamespace

import torch
import pytest

from freetoken.core import Req, SamplingParams
from freetoken.kvcache.linear_state_pool import LinearStatePool
from freetoken.models.config import LinearGatedDeltaGroupConfig
from freetoken.scheduler.cache import CacheManager


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
    cm.cache_req(reqA, finished=False)       # donate pp[0] at boundary 4; replace it in the pair
    # pp[0] donated to the tree; a fresh replacement was alloc'd -> net free-slot count unchanged
    assert pool.num_free_slots == free_before - 1  # one replacement alloc'd (donated slot now tree-owned)
    assert reqA.mamba_ping_pong[0] != pp[0]        # slot 0 replaced; pp[0] now lives in the tree

    # req B shares the [1,2,3,4] prefix -> HIT: returns the donated snapshot + reused KV
    mrB = cm.match_req(_pend([1, 2, 3, 4, 9]))
    assert mrB.cuda_handle.cached_len == 4
    assert mrB.mamba_value == pp[0]
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

    cm.cache_req(req, finished=True)         # donate the live slot directly (final state)
    # ping-pong pair freed; live slot kept (now owned by the tree)
    mr2 = cm.match_req(_pend([7, 8, 9, 10]))
    assert mr2.cuda_handle.cached_len == 3 and mr2.mamba_value == live


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


def test_naive_cache_does_not_align_prefill_chunks():
    """The alignment hook is hybrid-only; every other cache keeps the raw budget chunk."""
    from freetoken.scheduler.prefill import PrefillAdder
    from freetoken.scheduler.table import TableManager
    from freetoken.scheduler.utils import PendingReq

    pt = torch.zeros(4, 512, dtype=torch.int32)
    cm = CacheManager(64, 64, pt, "radix")
    assert cm.prefill_chunk_align == 1
    tm = TableManager(max_running_reqs=4, page_table=pt)
    adder = PrefillAdder(token_budget=100, reserved_size=0, cache_manager=cm, table_manager=tm)
    pending = PendingReq(0, torch.arange(300, dtype=torch.int32), SamplingParams(max_tokens=1))
    assert adder.try_add_one(pending).extend_len == 100


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
    stub = SimpleNamespace(cache_manager=cm, table_manager=tm)
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


def _qsa_pool():
    from freetoken.kvcache.qsa_pool import QSAKVCache
    return QSAKVCache(
        num_kv_heads=1, num_layers=QSA_TOTAL_LAYERS, head_dim=4, num_pages=QSA_PAGES,
        page_size=HOST_PAGE, dtype=torch.bfloat16, device=torch.device("cpu"),
        index_head_dim=QSA_IDX_DIM, num_index_layers=QSA_IDX_LAYERS, index_ratio=QSA_RATIO,
        num_req_slots=2, layer_ids=(1, 3),
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
