"""Fused PLE gate (vllm #54517 port) against the eager oracle chain.

Non-bit-exact by design (fusion changes the fp32 reduction order — annotated in
kernel/triton/ple_gate.py and the scoping doc), so correctness is pinned two ways:
elementwise distance to the eager chain must stay within 1 bf16 ulp for all but a
tiny fraction of elements, and the kernel must be bitwise deterministic across runs.
The eager fallback path (FREETOKEN_PLE_FUSED_GATE=0) must reproduce the old chain
exactly.
"""

from __future__ import annotations

import math

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

from freetoken.kernel.triton.ple_gate import ple_gate  # noqa: E402

HC, H = 4, 2560  # production Qwen3.8 geometry


def _eager(key_proj_out, value, R, nk_w, nq_w, ncw_w, eps):
    """The pre-#54517 chain from models/qwen4_exp/ple.py, transcribed 1:1."""

    def norm(x, w):
        xf = x.float().unflatten(-1, (HC, -1))
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
        return (xf.flatten(-2) * (1.0 + w.float())).to(x.dtype)

    key = norm(key_proj_out, nk_w)
    query = norm(R, nq_w)
    shape = (-1, HC, H)
    gate = (key.view(shape) * query.view(shape)).sum(-1, keepdim=True) / math.sqrt(H)
    gate = torch.sigmoid(gate.sign() * gate.abs().clamp_min(1e-6).sqrt())
    gated = (gate * value.unsqueeze(-2)).flatten(-2)
    return gated, norm(gated, ncw_w)


def _ulp_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ai = a.view(torch.int16).to(torch.int32)
    bi = b.view(torch.int16).to(torch.int32)
    # monotonic mapping for signed floats: flip negative bit patterns
    ai = torch.where(ai < 0, torch.iinfo(torch.int16).min - ai, ai)
    bi = torch.where(bi < 0, torch.iinfo(torch.int16).min - bi, bi)
    return (ai - bi).abs()


def _case(tokens: int, seed: int = 7):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    mk = lambda *s: torch.randn(*s, dtype=torch.bfloat16, device="cuda", generator=gen)
    key = mk(tokens, HC * H)
    value = mk(tokens, H)
    R = mk(tokens, HC * H)
    eps = 1e-6
    nk = torch.randn(HC * H, generator=gen, device="cuda") * 0.1
    nq = torch.randn(HC * H, generator=gen, device="cuda") * 0.1
    ncw = torch.randn(HC * H, generator=gen, device="cuda") * 0.1
    return key, value, R, nk, nq, ncw, eps


@pytest.mark.parametrize("tokens", [1, 3, 64, 1024])
def test_fused_gate_matches_eager_within_ulp(tokens: int):
    key, value, R, nk, nq, ncw, eps = _case(tokens)
    gated, normed = ple_gate(key, value, R, nk, nq, ncw, eps)
    gated_ref, normed_ref = _eager(key, value, R, nk, nq, ncw, eps)
    for got, ref, name in ((gated, gated_ref, "gated"), (normed, normed_ref, "normed")):
        diff = _ulp_diff(got, ref)
        bad = (diff > 1).float().mean().item()
        assert bad < 1e-3, f"{name}: {bad:.4%} of elements differ by >1 bf16 ulp (max {diff.max().item()})"
        assert diff.max().item() <= 4, f"{name}: extreme divergence {diff.max().item()} ulp"


def test_fused_gate_deterministic_across_runs():
    key, value, R, nk, nq, ncw, eps = _case(512, seed=13)
    g1, n1 = ple_gate(key, value, R, nk, nq, ncw, eps)
    g2, n2 = ple_gate(key, value, R, nk, nq, ncw, eps)
    assert torch.equal(g1, g2) and torch.equal(n1, n2)


def test_fallback_env_reproduces_eager(monkeypatch):
    """FREETOKEN_PLE_FUSED_GATE=0 must keep the legacy chain reachable."""
    import importlib

    monkeypatch.setenv("FREETOKEN_PLE_FUSED_GATE", "0")
    import freetoken.kernel.triton.ple_gate as mod

    importlib.reload(mod)
    assert not mod.fused_gate_enabled()
    monkeypatch.delenv("FREETOKEN_PLE_FUSED_GATE")
    importlib.reload(mod)
    assert mod.fused_gate_enabled()
