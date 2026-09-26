"""The vLLM custom-allreduce donor is opt-in: its 1-stage spin-read kernel produced Xid 31
MMU faults under PCIe P2P+H2D contention twice on this fleet (2026-09-16 reproduced,
2026-09-23 suspected), so it must stay off unless FREETOKEN_CUSTOM_ALL_REDUCE=1."""
from __future__ import annotations

from types import SimpleNamespace

from freetoken.distributed.impl import enable_custom_all_reduce


def test_donor_off_by_default(monkeypatch):
    monkeypatch.delenv("FREETOKEN_CUSTOM_ALL_REDUCE", raising=False)
    tp_info = SimpleNamespace(size=2, rank=0)
    assert enable_custom_all_reduce(tp_info, None, 1 << 20) is False


def test_donor_off_with_explicit_zero(monkeypatch):
    monkeypatch.setenv("FREETOKEN_CUSTOM_ALL_REDUCE", "0")
    tp_info = SimpleNamespace(size=2, rank=0)
    assert enable_custom_all_reduce(tp_info, None, 1 << 20) is False


def test_single_rank_never_loads_the_donor(monkeypatch):
    monkeypatch.setenv("FREETOKEN_CUSTOM_ALL_REDUCE", "1")
    tp_info = SimpleNamespace(size=1, rank=0)
    assert enable_custom_all_reduce(tp_info, None, 1 << 20) is False


# --- Single-stream guard (semantic port of sglang #31135 host side) -------------


class _FakeEvent:
    def __init__(self):
        self.recorded_on = []
        self.waited_by = []

    def record(self, stream):
        self.recorded_on.append(stream)

    def wait(self, stream):
        self.waited_by.append(stream)


class _FakeStream:
    def __init__(self, ptr):
        self.cuda_stream = ptr
        self.waits = []

    def wait_event(self, ev):
        self.waits.append(ev)
        ev.wait(self)


def _make_impl(monkeypatch, tmp_stream_holder):
    """CustomAllReduceImpl with a mock donor that always takes the custom path."""
    import torch

    from freetoken.distributed.impl import CustomAllReduceImpl

    monkeypatch.setattr(torch.cuda, "current_stream", lambda: tmp_stream_holder[0])
    monkeypatch.setattr(torch.cuda, "Event", _FakeEvent)

    ca = SimpleNamespace(
        disabled=False,
        _IS_CAPTURING=False,
        should_custom_ar=lambda x: True,
        custom_all_reduce=lambda x: x,
    )
    inner = SimpleNamespace(all_reduce=lambda x: x, all_gather=lambda x: x)
    return CustomAllReduceImpl(inner, ca)


def test_guard_noop_on_same_stream(monkeypatch):
    holder = [_FakeStream(0x1000)]
    impl = _make_impl(monkeypatch, holder)
    for _ in range(5):
        impl.all_reduce(object())
    assert impl._ar_event is None  # never serialized
    assert impl._ar_stream_ptr == 0x1000
    assert impl.calls == 5 and impl.calls_custom == 5


def test_guard_serializes_stream_switch(monkeypatch):
    s1, s2 = _FakeStream(0x1000), _FakeStream(0x2000)
    holder = [s1]
    impl = _make_impl(monkeypatch, holder)
    impl.all_reduce(object())
    holder[0] = s2
    impl.all_reduce(object())  # switch: record on s1, wait on s2
    ev = impl._ar_event
    assert ev is not None and ev.recorded_on == [s1] and ev.waited_by == [s2]
    assert s2.waits == [ev]
    impl.all_reduce(object())  # steady on s2: no further events
    assert ev.recorded_on == [s1]


def test_guard_skipped_for_fallback_and_capture(monkeypatch):
    s1, s2 = _FakeStream(0x1000), _FakeStream(0x2000)
    holder = [s1]
    impl = _make_impl(monkeypatch, holder)
    # declined custom path (oversize prefill tensor): no guard, no stream query
    impl.ca.should_custom_ar = lambda x: False
    impl.ca.custom_all_reduce = lambda x: None
    impl.all_reduce(object())
    assert impl._ar_stream_ptr == -1 and impl.calls_custom == 0
    # capture path: guard skipped even though the custom path is taken
    impl.ca.should_custom_ar = lambda x: True
    impl.ca.custom_all_reduce = lambda x: x
    impl.ca._IS_CAPTURING = True
    holder[0] = s2
    impl.all_reduce(object())
    assert impl._ar_stream_ptr == -1 and impl._ar_event is None
