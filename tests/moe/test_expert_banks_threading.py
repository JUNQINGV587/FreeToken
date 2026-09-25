from __future__ import annotations

import pytest
from freetoken.moe import expert_banks


@pytest.mark.parametrize("fail", [False, True])
def test_single_threaded_torch_copies_restores_threads(monkeypatch, fail):
    current = 32
    calls: list[int] = []

    monkeypatch.setattr(expert_banks.torch, "get_num_threads", lambda: current)
    monkeypatch.setattr(expert_banks.torch, "set_num_threads", calls.append)

    if fail:
        with pytest.raises(RuntimeError):
            with expert_banks._single_threaded_torch_copies():
                raise RuntimeError("loader failed")
    else:
        with expert_banks._single_threaded_torch_copies():
            pass

    assert calls == [1, current]
