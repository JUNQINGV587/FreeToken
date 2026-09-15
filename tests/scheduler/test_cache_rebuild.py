"""The cache-rebuild path where it has an in-process seam: the scheduler's idle gate and the
pool/table re-point it performs. The destructive orchestration underneath (graph teardown, pool
resize, page-table refresh, graph re-capture) has no seam worth stubbing and is covered against a
real server by tests/e2e/test_cache_rebuild.py; the maintenance state machine the HTTP layer runs
on top lives in tests/server/test_rebuild_maintenance.py."""

from __future__ import annotations

import torch


def _page_table(max_running_reqs: int, width: int) -> torch.Tensor:
    return torch.zeros((max_running_reqs + 1, width), dtype=torch.int32, device=torch.device("cpu"))


def _setup_context(page_size: int) -> None:
    """Initialize global context if not already done."""
    from freetoken.core import Context, get_global_ctx, set_global_ctx

    try:
        get_global_ctx()
    except AssertionError:
        # Create minimal context
        ctx = Context(page_size=page_size)
        set_global_ctx(ctx)


def test_cache_manager_rebuild_resets_pages_and_prefix():
    from freetoken.scheduler.cache import CacheManager

    _setup_context(page_size=2)

    pt = _page_table(4, 64)
    cm = CacheManager(num_pages=8, page_size=2, page_table=pt, type="radix")
    # mutate state so we can prove rebuild resets it
    cm.free_slots = cm.free_slots[:3]

    new_pt = _page_table(4, 128)
    cm.rebuild(num_pages=20, page_table=new_pt)

    assert cm.num_pages == 20
    assert cm.page_table is new_pt
    assert cm.free_slots.tolist() == [i * 2 for i in range(20)]
    assert cm.prefix_cache.size_info.total_size == 0
    cm.check_integrity()  # must pass: free_pages(20) + cache_pages(0) == num_pages(20)


def test_table_manager_rebuild_reallocs_token_pool_and_frees_slots():
    from freetoken.scheduler.table import TableManager

    pt = _page_table(4, 64)
    tm = TableManager(max_running_reqs=4, page_table=pt)
    tm.allocate(); tm.allocate()  # consume 2 slots

    new_pt = _page_table(4, 128)
    tm.rebuild(new_pt)

    assert tm.page_table is new_pt
    assert tm.token_pool.shape == new_pt.shape
    assert tm.available_size == 4  # all slots free again


def _stub_scheduler(*, prefill_runnable: bool, decode_runnable: bool, pending: object | None):
    """A Scheduler shell (no __init__/GPU) wired just enough to drive normal_loop's
    rebuild-drain branch. _execute_pending_rebuild is replaced with a recorder."""
    from types import SimpleNamespace

    from freetoken.scheduler.scheduler import Scheduler

    sched = Scheduler.__new__(Scheduler)
    sched.prefill_manager = SimpleNamespace(runnable=prefill_runnable)
    sched.decode_manager = SimpleNamespace(runnable=decode_runnable)
    sched._pending_rebuild = pending
    sched.receive_msg = lambda blocking: []
    sched._schedule_next_batch = lambda: None
    sched._process_last_data = lambda data: None
    calls = []

    def _exec():
        calls.append(True)
        sched._pending_rebuild = None

    sched._execute_pending_rebuild = _exec
    return sched, calls


def test_normal_loop_executes_pending_rebuild_when_idle():
    # Non-overlap mode (DISABLE_OVERLAP_SCHEDULING) must drain a queued rebuild at the idle
    # safe point, else it hangs until the HTTP request times out.
    from freetoken.scheduler.scheduler import Scheduler

    sched, calls = _stub_scheduler(prefill_runnable=False, decode_runnable=False, pending=object())
    Scheduler.normal_loop(sched)
    assert calls == [True]
    assert sched._pending_rebuild is None


def test_normal_loop_defers_pending_rebuild_while_busy():
    # A queued rebuild must NOT run while prefill/decode is still in flight.
    from freetoken.scheduler.scheduler import Scheduler

    pending = object()
    sched, calls = _stub_scheduler(prefill_runnable=False, decode_runnable=True, pending=pending)
    Scheduler.normal_loop(sched)
    assert calls == []
    assert sched._pending_rebuild is pending  # still queued
































def test_rebuild_cache_refreshes_prefill_budget(monkeypatch):
    # A rebuild that shrank the DSV4 window pool must shrink Scheduler.prefill_budget to the new
    # prefill_chunk_budget, or the next long prompt is chunked against the stale (larger) cap.
    from types import SimpleNamespace

    from freetoken.scheduler.scheduler import Scheduler

    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)

    sched = Scheduler.__new__(Scheduler)
    sched.prefill_manager = SimpleNamespace(runnable=False)
    sched.decode_manager = SimpleNamespace(runnable=False)
    sched.device = torch.device("cpu")
    sched.config = SimpleNamespace(tp_info=SimpleNamespace(size=1), max_extend_tokens=100_000)
    sched.engine = SimpleNamespace(
        rebuild_runtime_cache=lambda **kw: None, num_pages=32, page_table=None
    )
    # engine.page_table unchanged across the (stubbed) rebuild -> no token_pool re-point.
    sched.table_manager = SimpleNamespace(page_table=None)
    # DSV4-like manager: prefill_chunk_budget tracks the (about-to-shrink) window pool; no shared
    # page table, so rebuild_cache's prefix-cache rebuild branch is skipped.
    cache_manager = SimpleNamespace(
        prefill_chunk_budget=5000, rebuild=lambda *a: None, check_integrity=lambda: None)
    sched.cache_manager = cache_manager
    sched.table_manager.rebuild = lambda pt: None
    sched.table_manager.token_pool = None
    sched.prefill_budget = min(sched.config.max_extend_tokens, cache_manager.prefill_chunk_budget)
    assert sched.prefill_budget == 5000

    cache_manager.prefill_chunk_budget = 1000  # the (stubbed) engine rebuild shrank the pool
    Scheduler.rebuild_cache(sched, num_pages=16)
    assert sched.prefill_budget == 1000  # tracks the shrunk cap, not the stale 5000


def _pool_engine(calls, *, moe_slots=512, num_pages=4096, mamba_slots=9):
    """An engine carrying only the three pools _resize_pools touches, each logging its resize."""
    from types import SimpleNamespace

    engine = SimpleNamespace(num_pages=num_pages)
    engine.moe_offload_cache = SimpleNamespace(
        cache_size=moe_slots, rebuild=lambda n: calls.append(("moe", n)))
    engine.linear_state_pool = SimpleNamespace(
        num_slots=mamba_slots, rebuild=lambda n: calls.append(("mamba", n)))
    engine._resize_kv_pool = lambda config, pages, swa: calls.append(("kv", pages, swa))
    return engine


def test_resize_pools_frees_shrinking_pools_before_growing_ones():
    """#5: MoE 512 -> 1024 slots with KV 4096 -> 1024 pages OOMed on a full card because the MoE
    pool was reallocated while the KV pool still held its old pages."""
    from freetoken.engine.engine import Engine

    calls = []
    Engine._resize_pools(_pool_engine(calls), None, moe_cache_size=1024, num_pages=1024,
                         num_mamba_slots=4, num_swa_pages=None)
    assert calls == [("kv", 1024, None), ("mamba", 5), ("moe", 1024)]

    calls.clear()  # the rollback of that request: MoE shrinks back first, then KV and mamba grow
    Engine._resize_pools(_pool_engine(calls, moe_slots=1024, num_pages=1024, mamba_slots=5), None,
                         moe_cache_size=512, num_pages=4096, num_mamba_slots=8, num_swa_pages=None)
    assert calls == [("moe", 512), ("kv", 4096, None), ("mamba", 9)]


def test_resize_pools_keeps_request_order_among_growers_and_window_only_uses_current_pages():
    from freetoken.engine.engine import Engine

    calls = []
    Engine._resize_pools(_pool_engine(calls), None, moe_cache_size=600, num_pages=None,
                         num_mamba_slots=None, num_swa_pages=7)
    assert calls == [("moe", 600), ("kv", 4096, 7)]
