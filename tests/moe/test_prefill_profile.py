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
