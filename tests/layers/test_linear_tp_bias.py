"""A row-parallel linear with a bias must add that bias exactly once at TP>1.

Regression: the quant method added the full bias to every rank's own partial, and the
all-reduce that follows then summed it once per rank -- so at TP=2 every biased row-parallel
output carried twice its bias. A bias-free text tower hides it completely, which is why it
survived until the Qwen VL vision tower (every one of its linears is biased) ran at TP=2:
the image features came out systematically offset and the model hallucinated.

The TP=2 case is emulated with two threads and a barrier-and-sum all-reduce, so this runs on
CPU with no process group. Each rank's layer gets its own communicator, so no thread reads
the process-wide TP info.
"""
from __future__ import annotations

import threading

import pytest
import torch

import freetoken.distributed.info as info
from freetoken.distributed.info import DistributedInfo
from freetoken.layers.linear import LinearOProj, LinearRowParallel

IN_FEATURES, OUT_FEATURES, ROWS = 8, 4, 3


class _BarrierSum:
    """``all_reduce`` between exactly two ranks, each driving its own layer."""

    def __init__(self, size: int) -> None:
        self._size = size
        self._barrier = threading.Barrier(size)
        self._partials: list[torch.Tensor | None] = [None] * size

    def all_reduce(self, x: torch.Tensor, rank: int) -> torch.Tensor:
        self._partials[rank] = x
        self._barrier.wait()
        out = torch.stack([p for p in self._partials]).sum(dim=0)  # type: ignore[arg-type]
        self._barrier.wait()  # nobody overwrites a partial before its peer has read it
        return out


class _RankedComm:
    """Stands in for the layer's DistributedCommunicator, pinned to one emulated rank."""

    def __init__(self, shared: _BarrierSum, rank: int) -> None:
        self._shared, self._rank = shared, rank

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        return self._shared.all_reduce(x, self._rank)


def _make(monkeypatch, cls, tp_size: int, rank: int, full_weight: torch.Tensor, bias: torch.Tensor, comm=None):
    # setattr (not assignment) so the process-wide TP info is restored for the other tests
    monkeypatch.setattr(info, "_TP_INFO", DistributedInfo(rank=rank, size=tp_size))
    layer = cls(IN_FEATURES, OUT_FEATURES, True, quant_config=None, prefix="probe")
    cols = IN_FEATURES // tp_size
    # both classes are row parallel: a rank holds its slice of the INPUT columns and the full bias
    layer.weight = (
        full_weight if tp_size == 1 else full_weight[:, rank * cols : (rank + 1) * cols]
    ).clone()
    layer.bias = bias.clone()
    if comm is not None:
        layer._comm = comm
    return layer


@pytest.mark.parametrize("cls", [LinearRowParallel, LinearOProj])
def test_row_parallel_bias_is_added_once_at_tp2(monkeypatch, cls):
    torch.manual_seed(0)
    full_weight = torch.randn(OUT_FEATURES, IN_FEATURES)
    bias = torch.randn(OUT_FEATURES)
    x = torch.randn(ROWS, IN_FEATURES)

    expected = x @ full_weight.t() + bias
    reference = _make(monkeypatch, cls, 1, 0, full_weight, bias).forward(x)
    assert torch.allclose(reference, expected, atol=1e-5)

    shared = _BarrierSum(2)
    layers = [_make(monkeypatch, cls, 2, r, full_weight, bias, _RankedComm(shared, r)) for r in range(2)]
    out: list[torch.Tensor | None] = [None, None]

    def run(rank: int) -> None:
        x_local = x[:, rank * (IN_FEATURES // 2) : (rank + 1) * (IN_FEATURES // 2)]
        out[rank] = layers[rank].forward(x_local)

    threads = [threading.Thread(target=run, args=(r,)) for r in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    got = out[0]
    assert got is not None, "rank 0 produced no output"
    assert torch.allclose(got, reference, atol=1e-4), (
        f"{cls.__name__} TP2 differs from TP1 by max|diff|={(got - reference).abs().max().item():.4f} "
        f"(a doubled bias is exactly {bias.abs().max().item():.4f})"
    )
