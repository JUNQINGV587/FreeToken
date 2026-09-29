"""swiglu_clamp (GLM-5.3 ``swiglu_limit``) activation parity.

Reference: vLLM's SiluAndMulWithClamp with alpha=1, beta=0 --
``clamp(gate, max=L) * sigmoid(gate_clamped) * clamp(up, +-L)``. Checks the
Triton kernel, its distinction from swigluoai (the +1 up bias), and that the
compiled CPU MoE extension advertises the new generic act id.
"""

from __future__ import annotations

import pytest
import torch

LIMIT = 10.0


def _ref(x: torch.Tensor, limit: float = LIMIT) -> torch.Tensor:
    d = x.shape[-1] // 2
    gate = x[..., :d].float().clamp(max=limit)
    up = x[..., d:].float().clamp(min=-limit, max=limit)
    return (gate * torch.sigmoid(gate) * up).to(x.dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_triton_matches_reference():
    from freetoken.layers import swiglu_clamp_and_mul

    torch.manual_seed(0)
    # Scale up so the clamp actually engages on a good fraction of elements.
    x = torch.randn(129, 2 * 512, device="cuda", dtype=torch.bfloat16) * 8.0
    out = swiglu_clamp_and_mul(x, alpha=1.0, limit=LIMIT)
    ref = _ref(x)
    assert (out.float() - ref.float()).abs().max().item() < 2e-2
    assert (x[..., :512].float() > LIMIT).any(), "test data never hit the clamp"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_a_finite_limit_reaches_plain_silu_through_the_shared_epilogue():
    """``gated_act_and_mul`` is what every fused MoE epilogue calls. Plain silu has no alpha, but
    a limit is meaningful (DeepSeek-V4.1 routes it through here), so the dispatch must clamp
    rather than quietly drop the knob -- and an infinite limit must stay on the plain kernel."""
    from freetoken.layers import gated_act_and_mul, silu_and_mul

    torch.manual_seed(1)
    x = torch.randn(64, 2 * 256, device="cuda", dtype=torch.bfloat16) * 8.0
    out = torch.empty(64, 256, device="cuda", dtype=torch.bfloat16)

    gated_act_and_mul("silu", x, out, limit=LIMIT)
    ref = _ref(x)
    assert (out.float() - ref.float()).abs().max().item() < 2e-2
    assert (x[..., :256].float() > LIMIT).any(), "test data never hit the clamp"
    # the sigmoid sees the CLAMPED gate: swigluoai's shape (up + 1 bias) must NOT appear here
    assert not torch.allclose(out.float(), (ref + x[..., 256:].float().clamp(-LIMIT, LIMIT)).float())

    gated_act_and_mul("silu", x, out, limit=float("inf"))
    plain = torch.empty_like(out)
    silu_and_mul(x, plain)
    assert torch.equal(out, plain)

    # alpha rides along (the clamped kernel scales the sigmoid the same way silu does)
    gated_act_and_mul("silu", x, out, alpha=1.702, limit=LIMIT)
    gate = x[..., :256].float().clamp(max=LIMIT)
    ref_alpha = (gate * torch.sigmoid(gate * 1.702) * x[..., 256:].float().clamp(-LIMIT, LIMIT)).to(torch.bfloat16)
    assert (out.float() - ref_alpha.float()).abs().max().item() < 2e-2


def test_the_epilogue_reason_accepts_a_limit_on_silu_and_still_rejects_the_rest():
    from freetoken.layers.quantization.moe.base import MoEConfig, gated_epilogue_reason

    base = dict(num_experts=8, hidden=256, intermediate=128, top_k=2, scheme=None)
    assert gated_epilogue_reason(MoEConfig(**base, activation="silu", limit=10.0)) is None
    assert gated_epilogue_reason(MoEConfig(**base, activation="silu")) is None
    assert "alpha" in gated_epilogue_reason(MoEConfig(**base, activation="silu", alpha=1.702, limit=10.0))
    assert gated_epilogue_reason(MoEConfig(**base, activation="gelu", limit=10.0)) is not None
    assert gated_epilogue_reason(MoEConfig(**base, activation="silu", beta=0.5)) is not None


def test_cpu_extension_supports_swiglu_clamp():
    from freetoken.moe.cpu_executor import _cpu_act_id, compiled_extension_supports

    assert compiled_extension_supports("swiglu_clamp"), (
        "compiled _cpu_moe extension is stale -- rebuild with ACT_SWIGLU_CLAMP "
        "(python setup.py build_ext --inplace)"
    )
    # a silu expert that asks for a limit IS clamped swiglu: same math, so the CPU epilogue must
    # take the clamped act id instead of silently running the unclamped one
    from freetoken.moe.cpu_executor import _ACT_IDS

    assert _cpu_act_id("silu", 10.0) == _ACT_IDS["swiglu_clamp"]
    assert _cpu_act_id("silu", None) == _ACT_IDS["silu"]
    assert _cpu_act_id("silu", float("inf")) == _ACT_IDS["silu"]
    assert _cpu_act_id("gpt_oss_swiglu", 7.0) == _ACT_IDS["gpt_oss_swiglu"]
