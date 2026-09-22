"""Tests for the opt-in prefill staging profile (``FREETOKEN_PREFILL_PROFILE``).

The profiler exists to attribute the ~1.2 s host-CPU floor of a prefill batch to a phase
(see ``freetoken/moe/prefill_profile.py``). It is called from the hot path -- six phases
per decoder layer, 48 layers per chunk -- so the two properties that must hold are the
cheap ones: with the env unset it records nothing and reads no clock, and with it set each
chunk reports its own totals and then resets.

Both are pinned here with a fake clock, so the assertions are exact instead of timing
dependent.
"""

import pytest

from freetoken.moe import prefill_profile
from freetoken.moe.prefill_profile import PrefillProfiler, _truthy


class _FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def perf_counter(self) -> float:
        return self.now


@pytest.fixture
def clock(monkeypatch):
    fake = _FakeClock()
    monkeypatch.setattr(prefill_profile, "time", fake)
    return fake


def test_truthy_parsing():
    for value in ("1", "true", "TRUE", " yes ", "on"):
        assert _truthy(value)
    for value in ("", "0", "false", "no", "off", "2"):
        assert not _truthy(value)


def test_disabled_records_nothing_and_reads_no_clock(clock):
    prof = PrefillProfiler(False)
    with prof.phase("attn_core"):
        clock.now += 5.0  # a disabled profiler must not see this
    prof.begin_chunk()
    assert prof.end_chunk(layers=48) is None
    assert prof._totals == {}
    assert prof._counts == {}


def test_disabled_phase_is_the_shared_null_context():
    prof = PrefillProfiler(False)
    assert prof.phase("anything") is prefill_profile._NULL_PHASE
    with prof.phase("anything"):
        pass  # __exit__ must not swallow or re-raise anything


def test_enabled_accumulates_per_phase(clock):
    prof = PrefillProfiler(True)
    prof.begin_chunk()
    for _ in range(3):
        with prof.phase("stage_wait"):
            clock.now += 0.25
    with prof.phase("moe_gemm"):
        clock.now += 0.5
    line = prof.end_chunk(layers=48, extra="rows=12288 hits=8032")
    assert line is not None
    assert "stage_wait=750.0/3" in line
    assert "moe_gemm=500.0/1" in line
    assert "layers=48" in line
    assert "rows=12288 hits=8032" in line


def test_chunk_resets_accumulators(clock):
    prof = PrefillProfiler(True)
    prof.begin_chunk()
    with prof.phase("attn_core"):
        clock.now += 1.0
    first = prof.end_chunk(layers=48)
    prof.begin_chunk()
    with prof.phase("attn_core"):
        clock.now += 0.125
    second = prof.end_chunk(layers=48)
    assert first is not None and second is not None
    assert "attn_core=1000.0/1" in first
    assert "attn_core=125.0/1" in second
    assert "chunk=1" in first and "chunk=2" in second


def test_host_covers_the_whole_chunk_and_unaccounted_is_the_remainder(clock):
    prof = PrefillProfiler(True)
    prof.begin_chunk()
    with prof.phase("attn_core"):
        clock.now += 0.2
    with prof.phase("moe_total"):
        clock.now += 0.3
    clock.now += 0.5  # time outside every recorded phase
    line = prof.end_chunk(layers=48)
    assert line is not None
    assert "host=1000.0ms" in line
    assert "unaccounted=500.0ms" in line


def test_phases_are_printed_in_a_stable_order(clock):
    prof = PrefillProfiler(True)
    prof.begin_chunk()
    for name in ("moe_total", "attn_core", "zzz_custom", "entry", "attn_mix"):
        with prof.phase(name):
            clock.now += 0.001
    line = prof.end_chunk(layers=48)
    assert line is not None
    head = line.split("|")[1]
    order = [part.split("=")[0] for part in head.split()]
    recorded = ("entry", "attn_mix", "attn_core", "moe_total", "zzz_custom")
    # Every top-level phase is printed even at zero, so compare only the recorded ones:
    # they must follow the module's order, and an unknown phase must land after them.
    assert tuple(name for name in order if name in set(recorded)) == recorded
    assert order.index("zzz_custom") == len(order) - 1


def test_get_profiler_follows_the_module_flag(monkeypatch):
    monkeypatch.setattr(prefill_profile, "PROFILE_ENABLED", False)
    assert prefill_profile.get_profiler().enabled is False
    monkeypatch.setattr(prefill_profile, "PROFILE_ENABLED", True)
    assert prefill_profile.get_profiler().enabled is True


def test_disabled_bump_records_nothing():
    prof = PrefillProfiler(False)
    prof.begin_chunk()
    prof.bump("stage_driver", 3)
    assert prof._ops == {}


def test_bump_counts_accumulate_and_are_reported(clock):
    prof = PrefillProfiler(True)
    prof.begin_chunk()
    prof.bump("compact_launch")
    prof.bump("compact_launch", 2)
    prof.bump("stage_plan_tensor", 3)
    line = prof.end_chunk(layers=48)
    assert line is not None
    assert "ops=compact_launch:3,stage_plan_tensor:3" in line


def test_bump_counts_do_not_disturb_the_phase_partition(clock):
    prof = PrefillProfiler(True)
    prof.begin_chunk()
    with prof.phase("attn_core"):
        clock.now += 0.25
    prof.bump("attn_core_calls", 48)
    line = prof.end_chunk(layers=48)
    assert line is not None
    # counters are a separate segment: the phase segment still carries the timing partition
    assert "host=250.0ms unaccounted=0.0ms" in line
    head = line.split("|")[1]
    assert "attn_core=250.0/1" in head
    assert "ops=" not in head


def test_ops_are_reset_per_chunk(clock):
    prof = PrefillProfiler(True)
    prof.begin_chunk()
    prof.bump("stage_driver", 5)
    first = prof.end_chunk(layers=48)
    prof.begin_chunk()
    second = prof.end_chunk(layers=48)
    assert first is not None and second is not None
    assert "ops=" in first
    assert "ops=" not in second


def test_relaunch_control_needs_both_flags():
    # The relaunch is a measurement control and must not arm on its own env var alone.
    assert not prefill_profile._relaunch_enabled(False, "1")
    assert not prefill_profile._relaunch_enabled(True, "0")
    assert not prefill_profile._relaunch_enabled(True, "")
    assert prefill_profile._relaunch_enabled(True, "on")


class _FakeEvent:
    """Event stand-in: records which stream it was armed on, and a settable timestamp."""

    def __init__(self) -> None:
        self.streams = []
        self.done = False
        self.ms = 0.0

    def record(self, stream=None) -> None:
        # A freshly recorded event is NOT complete: the device completes it later, which
        # is what makes the deferred dump non-blocking.
        self.streams.append(stream)
        self.done = False

    def query(self) -> bool:
        return self.done

    def elapsed_time(self, other) -> float:
        return other.ms - self.ms


class _FakeOrigin:
    """Origin event whose elapsed_time reads the recorded timestamp off the peer."""

    def __init__(self) -> None:
        self.ms = 0.0

    def record(self, stream=None) -> None:
        pass

    def query(self) -> bool:
        return True

    def elapsed_time(self, other) -> float:
        return other.ms - self.ms


def _timeline_with_fakes():
    pool = []
    origin = _FakeOrigin()

    def factory():
        ev = origin if not pool else _FakeEvent()
        pool.append(ev)
        return ev

    return prefill_profile.LayerTimeline(event_factory=factory), pool


class _FakeTimeline:
    """Stand-in for LayerTimeline in tests that only care about the profiler facade."""

    def begin_chunk(self) -> None:
        pass

    def record(self, kind, layer, stream=None) -> None:
        pass

    def finish_chunk(self) -> None:
        pass


def test_batch_bucket_edges():
    assert prefill_profile.batch_bucket(1) == 0
    assert prefill_profile.batch_bucket(64 * 1024 - 1) == 0
    assert prefill_profile.batch_bucket(64 * 1024) == 1
    assert prefill_profile.batch_bucket(1024 * 1024 - 1) == 1
    assert prefill_profile.batch_bucket(1024 * 1024) == 2


def test_analyze_timeline_separates_device_stall_from_host_time():
    # Two layers, copy lands late for both: the device gap equals the copy's lateness.
    rows = [
        {"chunk_begin": 0.0, "wait_done": 1.0, "gemm_end": 3.0,
         "copy_begin": 0.5, "copy_end": 1.0, "h_wait_done": 0.001, "h_gemm_end": 0.003},
        {"chunk_begin": 0.0, "wait_done": 8.0, "gemm_end": 10.0,
         "copy_begin": 3.5, "copy_end": 8.0, "h_wait_done": 0.004, "h_gemm_end": 0.010},
    ]
    out = prefill_profile.analyze_timeline(rows)
    assert out["layers"] == 2
    # Layer 0 waits from the chunk origin, layer 1 from layer 0's gemm_end.
    assert out["stall_sum_ms"] == pytest.approx(1.0 + 5.0)
    assert out["copy_sum_ms"] == pytest.approx(0.5 + 4.5)
    assert out["late_p50_ms"] == pytest.approx(5.0)
    # Only layer 1 has a previous gemm_end, so exactly one host gap is measured.
    assert out["host_gap_sum_ms"] == pytest.approx(1.0)
    assert out["stall_frac_of_host"] == pytest.approx(6.0)


def test_analyze_timeline_reports_device_idle_as_a_small_stall():
    # The device was already drained: wait_done fires just after the previous gemm_end,
    # while the host took 10 ms to get there -> the host, not the copy, is the limit.
    rows = [
        {"chunk_begin": 0.0, "wait_done": 0.1, "gemm_end": 0.2,
         "copy_begin": 0.05, "copy_end": 0.1, "h_wait_done": 0.0001, "h_gemm_end": 0.0002},
        {"chunk_begin": 0.0, "wait_done": 0.21, "gemm_end": 0.3,
         "copy_begin": 0.15, "copy_end": 0.2, "h_wait_done": 0.0102, "h_gemm_end": 0.0104},
    ]
    out = prefill_profile.analyze_timeline(rows)
    assert out["stall_sum_ms"] == pytest.approx(0.11)
    assert out["stall_frac_of_host"] < 0.03


def test_analyze_timeline_is_empty_safe():
    out = prefill_profile.analyze_timeline([])
    assert out["layers"] == 0
    assert out["stall_sum_ms"] == 0.0
    assert out["stall_frac_of_host"] == 0.0


def test_timeline_requires_the_profile_gate():
    # Like the relaunch control: the timeline must not arm on its own env var alone.
    assert not prefill_profile._relaunch_enabled(False, "1")
    assert prefill_profile._relaunch_enabled(True, "on")


def test_timeline_records_every_kind_for_every_layer():
    tl, _ = _timeline_with_fakes()
    tl.begin_chunk()
    for layer in range(3):
        for kind in prefill_profile.LayerTimeline.KINDS:
            tl.record(kind, layer, stream=f"copy" if kind.startswith("copy") else None)
    rows = tl._rows
    assert len(rows) == 3
    for row in rows:
        for kind in prefill_profile.LayerTimeline.KINDS:
            assert kind in row


def test_timeline_rejects_unknown_kinds():
    tl, _ = _timeline_with_fakes()
    tl.begin_chunk()
    with pytest.raises(ValueError):
        tl.record("nope", 0)


def test_timeline_defers_the_dump_until_the_events_complete(clock, monkeypatch):
    monkeypatch.setattr(prefill_profile, "TIMELINE_OUT", "")
    tl, pool = _timeline_with_fakes()
    tl.begin_chunk()
    for layer in range(2):
        for kind in prefill_profile.LayerTimeline.KINDS:
            tl.record(kind, layer)
    tl.finish_chunk()
    assert len(tl._queue) == 1
    # The parked chunk's last event is still in flight -> nothing is dumped (no blocking).
    assert tl.flush() == 0
    assert len(tl._queue) == 1
    pool[-1].done = True
    assert tl.flush() == 1
    assert tl._queue == []
    assert len(tl.summaries) == 1


def test_timeline_queue_is_bounded():
    tl, _ = _timeline_with_fakes()
    for _ in range(6):
        tl.begin_chunk()
        for kind in prefill_profile.LayerTimeline.KINDS:
            tl.record(kind, 0)
        tl.finish_chunk()
    assert len(tl._queue) <= 4
    assert tl.dropped >= 2


def test_batch_stats_are_reported_and_reset(clock):
    prof = PrefillProfiler(True)
    prof.begin_chunk()
    prof.batch([1024, 2048])
    prof.batch([2 * 1024 * 1024], driver_ms=0.4)
    first = prof.end_chunk(layers=1)
    assert "batch=entries:3,bytes:" in first
    assert "lt64k:2,to1m:0,gt1m:1" in first
    assert "drv_max:0.40ms" in first
    prof.begin_chunk()
    assert "batch=" not in prof.end_chunk(layers=1)


def test_series_only_records_while_the_timeline_is_on(clock, monkeypatch):
    monkeypatch.setattr(prefill_profile, "LayerTimeline", _FakeTimeline)
    plain = PrefillProfiler(True)
    plain.begin_chunk()
    with plain.phase("attn_core", series=True):
        pass
    assert plain._series == {}
    timed = PrefillProfiler(True, timeline_enabled=True)
    timed.begin_chunk()
    timed.set_layer(7)
    with timed.phase("attn_core", series=True):
        pass
    assert [layer for layer, _ in timed._series["attn_core"]] == [7]


def test_disabled_profiler_touches_no_event_or_series(monkeypatch):
    prof = PrefillProfiler(False, timeline_enabled=True)
    def _boom(*_a, **_k):
        raise AssertionError("must not create CUDA events while disabled")
    monkeypatch.setattr(prefill_profile, "LayerTimeline", _boom)
    prof.begin_chunk()
    prof.set_layer(0)
    prof.evt("wait_done", 0)
    prof.batch([1])
    with prof.phase("attn_core", series=True):
        pass
    assert prof.end_chunk(layers=1) is None
