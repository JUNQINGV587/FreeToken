"""The idle-boundary MoE stats log in ``Scheduler.run_when_idle``.

When the engine's offload cache collects stats, each idle boundary must log the
decode-cache summary plus per-layer miss rates for the busy window that just ended
and reset the counters, so the next window starts clean. With collect_stats off
(or no offload cache at all) it must stay silent and never touch the cache.
"""
from __future__ import annotations

from types import SimpleNamespace

import freetoken.scheduler.scheduler as sched_mod
from freetoken.scheduler.scheduler import Scheduler


class _FakeMoeCache:
    def __init__(self, collect_stats: bool = True, layer_calls: int = 3):
        self.collect_stats = collect_stats
        self.reset_called = 0
        self._calls = layer_calls

    def decode_miss_stats(self) -> dict:
        return {
            "layer_calls": self._calls,
            "active_per_layer": 5.0,
            "missing_per_layer": 1.0,
            "miss_rate": 0.2,
            "prefill_hit_rows": 7,
            "prefill_rows": 10,
        }

    def decode_miss_stats_per_layer(self) -> dict:
        return {"per_layer": [{"layer": 0, "miss_rate": 0.25}]}

    def reset_stats(self) -> None:
        self.reset_called += 1


def _run(cache) -> tuple[list[str], _FakeMoeCache | None]:
    logs: list[str] = []
    real_logger = sched_mod.logger
    sched_mod.logger = SimpleNamespace(info_rank0=logs.append)
    try:
        stub = SimpleNamespace(
            engine=SimpleNamespace(moe_offload_cache=cache),
            cache_manager=SimpleNamespace(check_integrity=lambda: None),
        )
        Scheduler.run_when_idle(stub)
    finally:
        sched_mod.logger = real_logger
    return logs, cache


def test_idle_logs_one_busy_window_and_resets():
    cache = _FakeMoeCache()
    logs, _ = _run(cache)
    moe_lines = [l for l in logs if l.startswith("MoE decode cache")]
    assert len(moe_lines) == 2
    assert "miss rate: 0.200000" in moe_lines[0]
    assert "prefill hit rows: 7/10" in moe_lines[0]
    assert "0=0.250000" in moe_lines[1]
    assert cache.reset_called == 1


def test_idle_resets_without_logging_when_no_calls():
    # CUDA-graph warm-up alone can leave counters without real layer calls: still
    # reset so the next window is clean, but log nothing.
    cache = _FakeMoeCache(layer_calls=0)
    logs, _ = _run(cache)
    assert not [l for l in logs if l.startswith("MoE decode cache")]
    assert cache.reset_called == 1


def test_idle_silent_and_untouched_when_collect_stats_off():
    cache = _FakeMoeCache(collect_stats=False)
    logs, _ = _run(cache)
    assert not [l for l in logs if l.startswith("MoE decode cache")]
    assert cache.reset_called == 0


def test_idle_tolerates_engine_without_offload_cache():
    logs, _ = _run(None)
    assert logs == ["Scheduler is idle, waiting for new reqs..."]
