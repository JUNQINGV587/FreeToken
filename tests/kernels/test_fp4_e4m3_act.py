"""FP4 with an e4m3 block scale vs the reference ``fp4_act_quant(scale_dtype=e4m3)``.

The compressed-KV quantizer, used by ``Attention._compress_kv`` with block 16. ``fp4_e4m3_ref.pt``
comes from ``research/v41-oracle/fp4_e4m3_oracle.py`` and includes degenerate groups: all-zero
blocks and blocks whose ``amax`` sits far below the ``6 * 2**-9`` scale floor, i.e. exactly the
cases the e8m0 variant gets wrong.

    FT_V41_FP4_E4M3_DUMP  default /oracle/fp4_e4m3_ref.pt
"""

from __future__ import annotations

import os

import pytest
import torch

from freetoken.kernel.triton.dsv4.fp8_linear import fp4_act_quant_inplace
from freetoken.kernel.triton.fp4_e4m3_act import E4M3_MIN_AMAX, fp4_act_quant_e4m3_inplace

DUMP = os.environ.get("FT_V41_FP4_E4M3_DUMP", "/oracle/fp4_e4m3_ref.pt")

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and os.path.isfile(DUMP)),
    reason="needs CUDA + the fp4/e4m3 oracle dump",
)


def _reference() -> dict:
    return torch.load(DUMP, map_location="cpu", weights_only=False)


KEYS = (
    sorted(torch.load(DUMP, map_location="cpu", weights_only=False).keys())
    if os.path.isfile(DUMP)
    else []
)


@pytest.mark.parametrize("key", KEYS)
def test_roundtrip_matches_the_reference(key):
    case = _reference()[key]
    x = case["x"].to("cuda").clone()
    got = fp4_act_quant_e4m3_inplace(x, case["block"])
    want = case["y"].to("cuda")
    assert got.shape == want.shape
    assert torch.equal(got, want), f"{key}: {(got != want).sum().item()} of {got.numel()} differ"


def test_the_scale_floor_keeps_an_all_zero_group_nonzero():
    """``amax`` is floored at 6 * 2**-9, so a zero group still carries a (nonzero) scale -- but its
    dequantized values are zero, which is what the reference stores too."""
    x = torch.zeros(4, 64, dtype=torch.bfloat16, device="cuda")
    out = fp4_act_quant_e4m3_inplace(x, 16)
    assert torch.equal(out, torch.zeros_like(out))
    assert E4M3_MIN_AMAX == 6 * (2.0**-9)


def test_e4m3_scales_are_not_the_e8m0_ones():
    """The two fp4 regimes differ whenever a group's amax is not a power-of-two multiple of 6:
    this pins down that the compressed KV uses the e4m3 grid rather than rounding to a power of 2."""
    gen = torch.Generator().manual_seed(7)
    x = (torch.randn(8, 128, generator=gen) * 0.2).to(torch.bfloat16).cuda()
    # a group whose amax sits between two powers of two: 6 * 2^-9 vs 6 * 2^-8
    x[0, :16] = torch.linspace(-0.0175, 0.0175, 16).to(torch.bfloat16).cuda()
    a = fp4_act_quant_e4m3_inplace(x.clone(), 16)
    b = fp4_act_quant_inplace(x.clone(), 16)
    assert not torch.equal(a, b)
