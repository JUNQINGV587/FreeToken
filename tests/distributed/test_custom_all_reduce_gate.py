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
