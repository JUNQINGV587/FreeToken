"""The /v1/stats sampler: off, throttled, and never fatal.

Bound to a stand-in self so the test exercises the shipped method (and its throttle attribute)
rather than a copy of the logic.
"""
from types import SimpleNamespace

import pytest

from freetoken.scheduler.scheduler import Scheduler

sample = Scheduler._host_tier_stats_snapshot


class FakeCM:
    def __init__(self, answer):
        self.answer = answer
        self.calls = 0

    def host_tier_stats(self):
        self.calls += 1
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


def test_absent_or_disabled_tier_samples_as_none():
    assert sample(SimpleNamespace()) is None, "no cache manager at all"
    cm = FakeCM(None)
    assert sample(SimpleNamespace(cache_manager=cm)) is None, "tier off"
    assert cm.calls == 1, "and it did ask"
    # A disabled tier must not engage the throttle on behalf of the enabled one later, so the
    # timestamp stays unset and the next reply asks again.
    assert sample(SimpleNamespace(cache_manager=cm)) is None and cm.calls == 2


def test_a_raising_snapshot_returns_none_instead_of_breaking_the_stream():
    boom = RuntimeError("ledger corrupted")
    assert sample(SimpleNamespace(cache_manager=FakeCM(boom))) is None


def test_a_live_snapshot_is_returned_once_per_second():
    snap = {"spills": 3}
    cm = FakeCM(snap)
    me = SimpleNamespace(cache_manager=cm)
    assert sample(me) == snap, "the first sample goes through"
    assert sample(me) is None, "throttled: the frontend keeps the last known value"
    assert cm.calls == 1, "a throttled sample does not even ask"
