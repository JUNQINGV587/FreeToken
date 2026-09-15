from __future__ import annotations

from types import SimpleNamespace

import torch

from freetoken.scheduler import scheduler as sched_mod
from freetoken.scheduler.scheduler import Scheduler


def test_gpu_mem_bytes_is_sampled_at_most_once_per_second(monkeypatch):
    """memory_reserved() walks the allocator, so it must not run on every decode step."""
    calls: list[object] = []

    def fake_reserved(device):
        calls.append(device)
        return 1234

    clock = {"t": 100.0}
    monkeypatch.setattr(torch.cuda, "memory_reserved", fake_reserved)
    monkeypatch.setattr(sched_mod.time, "monotonic", lambda: clock["t"])

    stub = SimpleNamespace(device=SimpleNamespace(type="cuda"))
    assert Scheduler._gpu_mem_bytes(stub) == 1234
    assert len(calls) == 1

    clock["t"] = 100.5  # inside the window -> last-known value, no resample
    assert Scheduler._gpu_mem_bytes(stub) == 1234
    assert len(calls) == 1

    clock["t"] = 101.5  # window elapsed -> resample
    assert Scheduler._gpu_mem_bytes(stub) == 1234
    assert len(calls) == 2


def test_gpu_mem_bytes_is_zero_on_cpu():
    stub = SimpleNamespace(device=SimpleNamespace(type="cpu"))
    assert Scheduler._gpu_mem_bytes(stub) == 0
