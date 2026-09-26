"""Fused PLE conv (vllm #54517 port, decode+prefill modes) vs the HF oracle.

Oracle = ``models/qwen4_exp/ple.py::short_conv_reference`` (per-request F.conv1d
transcription). Pins:
- output within bf16 ulp tolerance of gated + oracle (tap-order fusion is not
  bit-exact, annotated in kernel/triton/ple_conv.py);
- rolled conv state BITWISE equal (the roll moves raw bf16 values, no math);
- fresh slots (no history) identical to zeroed state;
- determinism across runs; env fallback reachable.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

from freetoken.kernel.triton.ple_conv import ple_conv  # noqa: E402
from freetoken.models.qwen4_exp.ple import short_conv_reference  # noqa: E402

C, K, DIL = 512, 4, 3
STATE_LEN = (K - 1) * DIL  # 9


def _meta(lengths, fresh, device):
    from types import SimpleNamespace

    cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int64)
    return SimpleNamespace(
        seq_lens=list(lengths),
        state_slots=torch.arange(len(lengths), dtype=torch.int64, device=device),
        cu_seqlens=cu.to(device),
        fresh_slots=torch.tensor(fresh, dtype=torch.bool, device=device)
        if fresh is not None
        else None,
        is_decode=len(lengths) > 0 and all(n == 1 for n in lengths),
    )


def _ulp_frac(a: torch.Tensor, b: torch.Tensor):
    ai = a.view(torch.int16).cpu().int()
    bi = b.view(torch.int16).cpu().int()
    lo = torch.iinfo(torch.int16).min
    ai = torch.where(ai < 0, lo - ai, ai)
    bi = torch.where(bi < 0, lo - bi, bi)
    d = (ai - bi).abs()
    return d.max().item(), (d > 1).float().mean().item()


def _build(lengths, fresh, seed=5):
    device = torch.device("cuda")
    gen = torch.Generator(device="cuda").manual_seed(seed)
    total = sum(lengths)
    x = torch.randn(total, C, dtype=torch.bfloat16, device=device, generator=gen)
    gated = torch.randn(total, C, dtype=torch.bfloat16, device=device, generator=gen)
    weight = torch.randn(C, 1, K, dtype=torch.bfloat16, device=device, generator=gen)
    states = torch.randn(len(lengths), C, STATE_LEN, dtype=torch.bfloat16, device=device, generator=gen)
    meta = _meta(lengths, fresh, device)
    return x, gated, weight, states, meta


def _run_fused(x, gated, weight, states, meta):
    res = gated.clone()
    st = states.clone()
    ple_conv(
        x, res, st, weight.squeeze(1), meta.state_slots,
        mode="decode" if meta.is_decode else "prefill",
        dilation=DIL,
        query_start_loc=None if meta.is_decode else meta.cu_seqlens,
        has_initial_states=None if meta.fresh_slots is None else ~meta.fresh_slots,
    )
    return res, st


def _run_ref(x, gated, weight, states, meta):
    st = states.clone()
    out = short_conv_reference(x, meta, st, weight, DIL)
    return gated + out, st


def test_decode_matches_oracle():
    x, gated, w, states, meta = _build([1, 1, 1, 1], [False, True, False, False])
    got, got_st = _run_fused(x, gated, w, states, meta)
    ref, ref_st = _run_ref(x, gated, w, states, meta)
    mx, frac = _ulp_frac(got, ref)
    assert frac < 1e-3 and mx <= 4, f"decode output divergence: max {mx} ulp, >1ulp {frac:.5%}"
    assert torch.equal(got_st, ref_st), "decode state roll not bitwise"


def test_prefill_matches_oracle():
    x, gated, w, states, meta = _build([37, 1, 256, 5], [False, False, True, False])
    got, got_st = _run_fused(x, gated, w, states, meta)
    ref, ref_st = _run_ref(x, gated, w, states, meta)
    mx, frac = _ulp_frac(got, ref)
    assert frac < 1e-3 and mx <= 4, f"prefill output divergence: max {mx} ulp, >1ulp {frac:.5%}"
    assert torch.equal(got_st, ref_st), "prefill state writeback not bitwise"


def test_prefill_no_fresh_slots():
    x, gated, w, states, meta = _build([64, 128], None)
    got, got_st = _run_fused(x, gated, w, states, meta)
    ref, ref_st = _run_ref(x, gated, w, states, meta)
    mx, frac = _ulp_frac(got, ref)
    assert frac < 1e-3 and mx <= 4
    assert torch.equal(got_st, ref_st)


def test_deterministic_across_runs():
    x, gated, w, states, meta = _build([100, 50], [True, False], seed=9)
    a, as_ = _run_fused(x, gated, w, states, meta)
    b, bs_ = _run_fused(x, gated, w, states, meta)
    assert torch.equal(a, b) and torch.equal(as_, bs_)


def test_env_fallback_flag(monkeypatch):
    import importlib

    import freetoken.kernel.triton.ple_conv as mod

    monkeypatch.setenv("FREETOKEN_PLE_FUSED_CONV", "0")
    importlib.reload(mod)
    assert not mod.fused_conv_enabled()
    monkeypatch.delenv("FREETOKEN_PLE_FUSED_CONV")
    importlib.reload(mod)
    assert mod.fused_conv_enabled()
