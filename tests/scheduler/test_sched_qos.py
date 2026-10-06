"""Scheduler QoS: time-based decode share and the contention chunk cap (dsv41 port).

Decode share (FREETOKEN_DECODE_SHARE): a step that carried prefill and took T wall
seconds earns f*T of decode-only allowance, so an already-decoding request keeps
advancing while a long prompt chunks through prefill (dsv41 measured 0.1 t/s collapse
without it, ~3.3 t/s with f=0.25). Two behaviours must travel with the mechanism, and
the tests pin both:

  * the waiting exception (dsv41 D130): a FRESH waiting request is never held back by
    the allowance -- without it short TTFT measured 117 s;
  * the 10 s allowance cap, so one giant prefill step cannot bank an unbounded decode
    monopoly.

Contention chunk cap (FREETOKEN_LONG_PREFILL_WHEN_WAITING): with >= 2 requests competing
for prefill passes the chunk clamps so a long prompt yields the queue sooner; a lone
prompt keeps the full budget.

The policy tests are pure Python (injected clock); the wiring tests drive the shipped
Scheduler._schedule_next_batch on a stripped Scheduler, mirroring test_interleave.py.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from freetoken.scheduler.chunk_policy import CONTENTION_CAP_ENV
from freetoken.scheduler.decode_share import CAP_ENV, SHARE_ENV, DecodeSharePolicy
from freetoken.scheduler.prefill import PrefillManager
from freetoken.scheduler.scheduler import Scheduler


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, dt: float) -> None:
        self.now += dt


def _policy(share=0.25, cap_s=10.0, clock=None):
    return DecodeSharePolicy(share, cap_s, clock or FakeClock())


# --------------------------------------------------------------------------- #
# from_env
# --------------------------------------------------------------------------- #
def test_from_env_off_by_default(monkeypatch):
    """Unset, 0, negative, and unparseable all mean OFF -- the historical order."""
    for raw in (None, "", "0", "-0.5", "abc"):
        monkeypatch.delenv(SHARE_ENV, raising=False)
        if raw is not None:
            monkeypatch.setenv(SHARE_ENV, raw)
        assert DecodeSharePolicy.from_env() is None, raw


def test_from_env_builds_with_default_cap(monkeypatch):
    monkeypatch.setenv(SHARE_ENV, "0.25")
    monkeypatch.delenv(CAP_ENV, raising=False)
    p = DecodeSharePolicy.from_env()
    assert p.share == 0.25 and p.cap_s == 10.0


def test_from_env_cap_override_and_invalid_cap_falls_back(monkeypatch):
    monkeypatch.setenv(SHARE_ENV, "0.5")
    monkeypatch.setenv(CAP_ENV, "3.5")
    assert DecodeSharePolicy.from_env().cap_s == 3.5
    monkeypatch.setenv(CAP_ENV, "junk")
    assert DecodeSharePolicy.from_env().cap_s == 10.0


# --------------------------------------------------------------------------- #
# allowance accounting (pure, injected clock)
# --------------------------------------------------------------------------- #
def test_prefill_span_grants_share_times_dt():
    clock = FakeClock()
    p = _policy(clock=clock)
    p.close_span()  # first slot opens the clock; nothing to settle
    p.note_step(had_prefill=True)
    clock.advance(8.0)  # the prefill-carrying step took 8 s of wall
    p.close_span()
    assert p.allow_s == pytest.approx(2.0), "f * T = 0.25 * 8"
    assert p.granted_s == pytest.approx(2.0)


def test_allowance_is_capped():
    clock = FakeClock()
    p = _policy(cap_s=10.0, clock=clock)
    p.close_span()
    p.note_step(had_prefill=True)
    clock.advance(1000.0)  # one monstrous prefill step must not bank 250 s
    p.close_span()
    assert p.allow_s == 10.0


def test_non_prefill_span_consumes_allowance():
    clock = FakeClock()
    p = _policy(clock=clock)
    p.close_span()
    p.note_step(had_prefill=True)
    clock.advance(8.0)
    p.close_span()  # allow = 2.0
    p.note_step(had_prefill=False)  # decode-only span
    clock.advance(1.0)
    p.close_span()
    assert p.allow_s == pytest.approx(1.0)
    assert p.consumed_s == pytest.approx(1.0)
    clock.advance(10.0)  # longer than the remaining allowance: clamps at 0, not negative
    p.close_span()
    assert p.allow_s == 0.0
    assert p.consumed_s == pytest.approx(2.0)


def test_first_settlement_has_no_span():
    p = _policy()
    p.close_span()
    assert p.allow_s == 0.0


# --------------------------------------------------------------------------- #
# wants_decode truth table
# --------------------------------------------------------------------------- #
def test_wants_decode_requires_allowance_decode_and_no_fresh_waiting():
    p = _policy()
    assert not p.wants_decode(decoding=True, waiting_new=False), "no allowance yet"
    p.allow_s = 1.0
    assert p.wants_decode(decoding=True, waiting_new=False)
    assert not p.wants_decode(decoding=True, waiting_new=True), (
        "D130: a fresh waiting request rides along in the next pass; holding it back "
        "measured 117 s short TTFT on dsv41"
    )
    assert not p.wants_decode(decoding=False, waiting_new=False), "nothing to promote"


def test_disabled_policy_never_wants_decode():
    p = DecodeSharePolicy(0.0)
    p.allow_s = 100.0
    assert not p.wants_decode(decoding=True, waiting_new=False)


def test_snapshot_reports_the_ledger():
    p = _policy(cap_s=7.5)
    p.allow_s = 1.23456
    p.throttled = 3
    snap = p.snapshot()
    assert snap["share"] == 0.25 and snap["cap_s"] == 7.5
    assert snap["allow_s"] == 1.235 and snap["throttled"] == 3
    assert snap["granted_s"] == 0.0 and snap["consumed_s"] == 0.0


# --------------------------------------------------------------------------- #
# wiring into the REAL Scheduler._schedule_next_batch (stripped-scheduler harness,
# same pattern as test_interleave.py -- SimpleNamespace batches, no GPU)
# --------------------------------------------------------------------------- #
PREFILL_BATCH = SimpleNamespace(is_prefill=True, prompt_admissions=[])
DECODE_BATCH = SimpleNamespace(is_prefill=False, prompt_admissions=[])


def _stripped_scheduler(*, share=None, waiting_new=False, decode_runnable=True):
    s = Scheduler.__new__(Scheduler)
    s.prefill_budget = 8192
    s.prefill_manager = SimpleNamespace(
        schedule_next_batch=lambda budget: PREFILL_BATCH,
        has_new_waiting=waiting_new,
    )
    s.decode_manager = SimpleNamespace(
        schedule_next_batch=lambda: DECODE_BATCH if decode_runnable else None,
        runnable=decode_runnable,
    )
    s._prepare_batch = lambda value: value
    s.send_result = lambda messages: None
    if share is not None:
        s._decode_share = share
    return s


def test_share_grants_decode_slot_after_a_prefill_span_then_prefill_resumes():
    clock = FakeClock()
    share = _policy(clock=clock)
    s = _stripped_scheduler(share=share, waiting_new=False)
    # Slot 1: no allowance yet, prefill keeps priority.
    assert Scheduler._schedule_next_batch(s) is PREFILL_BATCH
    assert share.throttled == 0
    clock.advance(8.0)  # prefill step took 8 s -> 2 s allowance
    # Slot 2: the allowance buys a decode-only slot even though prefill is pending.
    assert Scheduler._schedule_next_batch(s) is DECODE_BATCH
    assert share.throttled == 1
    clock.advance(4.0)  # decode ran past the 2 s allowance
    # Slot 3: allowance exhausted -> prefill resumes; the prefill span also re-grants.
    assert Scheduler._schedule_next_batch(s) is PREFILL_BATCH
    assert share.throttled == 1
    assert share.consumed_s == pytest.approx(2.0)


def test_fresh_waiting_request_is_never_held_back():
    """The D130 exception: allowance or not, a request that has not started prefill
    rides along in the next pass."""
    clock = FakeClock()
    share = _policy(clock=clock)
    share.allow_s = 9.0  # plenty of allowance banked
    s = _stripped_scheduler(share=share, waiting_new=True)
    assert Scheduler._schedule_next_batch(s) is PREFILL_BATCH
    assert share.throttled == 0


def test_chunked_continuation_does_not_count_as_fresh_waiting():
    """Only fresh arrivals get the exception; delaying a continuation's next chunk is
    precisely what the share is for."""
    share = _policy()
    share.allow_s = 9.0
    s = _stripped_scheduler(share=share, waiting_new=False)
    assert Scheduler._schedule_next_batch(s) is DECODE_BATCH
    assert share.throttled == 1


def test_share_without_runnable_decode_stays_prefill():
    share = _policy()
    share.allow_s = 9.0
    s = _stripped_scheduler(share=share, decode_runnable=False)
    assert Scheduler._schedule_next_batch(s) is PREFILL_BATCH
    assert share.throttled == 0


def test_scheduler_without_share_keeps_historical_order():
    """A stripped Scheduler with no _decode_share attribute must not break (the
    accounting tests build one)."""
    s = _stripped_scheduler(share=None)
    assert not hasattr(s, "_decode_share")
    assert Scheduler._schedule_next_batch(s) is PREFILL_BATCH


# --------------------------------------------------------------------------- #
# PrefillManager.has_new_waiting
# --------------------------------------------------------------------------- #
def _manager(pending):
    return PrefillManager(
        cache_manager=None,
        table_manager=None,
        decode_manager=SimpleNamespace(inflight_tokens=0),
        pending_list=pending,
    )


def test_has_new_waiting_distinguishes_fresh_from_continuation():
    assert not _manager([]).has_new_waiting
    fresh = SimpleNamespace(chunked_req=None)
    continuation = SimpleNamespace(chunked_req=object())
    assert _manager([fresh]).has_new_waiting
    assert not _manager([continuation]).has_new_waiting, (
        "a chunked continuation is 'prefilling', not 'waiting' -- dsv41's exception "
        "covers only requests that have not started"
    )
    assert _manager([continuation, fresh]).has_new_waiting


# --------------------------------------------------------------------------- #
# _qos_stats_snapshot sampler (bound to a stand-in self, same pattern as
# test_host_tier_stats_sample.py)
# --------------------------------------------------------------------------- #
sample = Scheduler._qos_stats_snapshot


def test_qos_sample_is_none_when_both_knobs_off(monkeypatch):
    monkeypatch.delenv(CONTENTION_CAP_ENV, raising=False)
    assert sample(SimpleNamespace()) is None


def test_qos_sample_reports_share_ledger_and_throttles(monkeypatch):
    monkeypatch.delenv(CONTENTION_CAP_ENV, raising=False)
    share = _policy()
    share.allow_s = 2.0
    pm = SimpleNamespace(pending_list=[1, 2], contention_capped_passes=3)
    me = SimpleNamespace(_decode_share=share, prefill_manager=pm)
    snap = sample(me)
    assert snap["decode_share"]["allow_s"] == 2.0
    assert snap["chunk_cap"] == {"cap_tokens": 0, "contending": True, "capped_passes": 3}
    assert sample(me) is None, "throttled: the frontend keeps the last known value"


def test_qos_sample_reports_chunk_cap_without_share(monkeypatch):
    monkeypatch.setenv(CONTENTION_CAP_ENV, "7168")
    pm = SimpleNamespace(pending_list=[], contention_capped_passes=0)
    snap = sample(SimpleNamespace(prefill_manager=pm))
    assert snap["decode_share"] is None
    assert snap["chunk_cap"]["cap_tokens"] == 7168
    assert snap["chunk_cap"]["contending"] is False
