"""Regression tests for the MoE bandwidth bench (``ft bench bw``).

The bench rig stages ONE synthetic layer and then calls ``OffloadMoeCache.copy_missing``
in a loop to time it. ``copy_missing`` consumes its staged state exactly once (it clears
``_pending_src_layer``), so the loop has to re-arm the marker per iteration. When it did
not, the second call tripped "no staged misses" and the whole bench died -- and since
``_bench_format`` only catches ``(ImportError, RuntimeError)``, the ``AssertionError``
propagated out of ``ft bench bw`` instead of degrading one format.

Both tests drive the public bench entry points, so on the unfixed tree they fail with that
``AssertionError`` rather than with a missing symbol.
"""

import pytest
import torch

from freetoken.moe.benchbw import DTYPE_WORKLOADS


def _workload():
    # The cheapest canonical geometry: 128 experts, H=2048, I=768.
    return DTYPE_WORKLOADS["bf16"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_pcie_gather_bench_returns_a_rate():
    from freetoken.moe.benchbw import measure_pcie_gather_bw

    wl = _workload()
    out = measure_pcie_gather_bw("bf16", wl, torch.device("cuda:0"), iters=2)
    assert out["bw_gbs"] > 0.0
    assert out["synth_experts"] == wl.experts


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_overlap_bench_returns_rates():
    from freetoken.moe.benchbw import measure_overlap_bw

    wl = _workload()
    out = measure_overlap_bw("bf16", wl, torch.device("cuda:0"), seconds=0.2)
    assert out["cpu_gbs"] > 0.0
    assert out["pcie_gbs"] > 0.0
