"""Fused PLE gate: norm_key + norm_query + dot gate + sigmoid + gate*value + norm_conv.

Ported from vllm #54517 (``ops/ple.py::_ple_gate_kernel``, f870b92976), PDL stripped
(sm89 reports ``is_arch_support_pdl() == False`` upstream, so the flag was dead code
for us). One kernel replaces six eager ops in ``models/qwen4_exp/ple.py``.

RED-LINE NOTE (per notes/engines/20260926-qwen38-optimization-scoping.md §4): the
fusion keeps the eager dtype boundaries (every intermediate that eager materializes
in bf16 is rounded to bf16 here), but ``tl.sum``'s tree reduction order differs from
torch's eager reduction order and from the vendored ``grouped_gemma_rmsnorm`` kernel,
so output is NOT bit-identical to the previous chain — it is deterministic per run
and validated against the eager oracle within bf16 tolerance. Revert with
``FREETOKEN_PLE_FUSED_GATE=0``.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl


def fused_gate_enabled() -> bool:
    return os.environ.get("FREETOKEN_PLE_FUSED_GATE", "1") != "0"


@triton.jit
def _ple_gate_kernel(
    key_ptr,        # [T, HC*H] projected key, pre-norm
    value_ptr,      # [T, H] value, shared across groups
    hidden_ptr,     # [T, HC*H] residual streams (query source), pre-norm
    nk_ptr,         # [HC*H] norm_key weight (zero-centered, raw)
    nq_ptr,         # [HC*H] norm_query weight
    ncw_ptr,        # [HC*H] norm_conv weight
    gated_ptr,      # [T, HC*H] out
    normed_ptr,     # [T, HC*H] out
    key_rs: tl.int64,
    value_rs: tl.int64,
    eps,
    H: tl.constexpr,
    HC: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    t = tl.program_id(0)
    s = tl.program_id(1)
    lanes = tl.arange(0, BLOCK_H)
    mask = lanes < H
    offs = s * H + lanes
    dtype: tl.constexpr = key_ptr.dtype.element_ty

    k = tl.load(key_ptr + t * key_rs + offs, mask=mask, other=0.0).to(tl.float32)
    q = tl.load(hidden_ptr + t * HC * H + offs, mask=mask, other=0.0).to(tl.float32)
    nk = tl.load(nk_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    nq = tl.load(nq_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # Match eager materialization at each intermediate tensor boundary.
    k_n = (k * tl.rsqrt(tl.sum(k * k) / H + eps) * (1.0 + nk)).to(dtype)
    q_n = (q * tl.rsqrt(tl.sum(q * q) / H + eps) * (1.0 + nq)).to(dtype)
    products = (k_n.to(tl.float32) * q_n.to(tl.float32)).to(dtype)
    dot = tl.sum(products.to(tl.float32)).to(dtype).to(tl.float32)
    d = dot / tl.sqrt(float(H))
    d = d.to(dtype).to(tl.float32)
    sign = tl.where(d < 0, -1.0, 0.0)
    sign = tl.where(d > 0, 1.0, sign)
    magnitude = tl.sqrt(tl.maximum(tl.abs(d), 1e-6)).to(dtype)
    g = tl.sigmoid(sign * magnitude.to(tl.float32))
    g = g.to(dtype).to(tl.float32)

    v = tl.load(value_ptr + t * value_rs + lanes, mask=mask, other=0.0).to(tl.float32)
    gated = (g * v).to(dtype)
    gf = gated.to(tl.float32)
    ncw = tl.load(ncw_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    normed = gf * tl.rsqrt(tl.sum(gf * gf) / H + eps) * (1.0 + ncw)

    tl.store(gated_ptr + t * HC * H + offs, gated, mask=mask)
    tl.store(normed_ptr + t * HC * H + offs, normed, mask=mask)


def ple_gate(
    key: torch.Tensor,
    value: torch.Tensor,
    hidden: torch.Tensor,
    norm_key_w: torch.Tensor,
    norm_query_w: torch.Tensor,
    norm_conv_w: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (gated, normed) — the eager chain's ``gated`` and ``norm_conv(gated)``."""
    num_tokens = hidden.shape[0]
    h = value.shape[-1]
    hc = hidden.shape[-1] // h
    assert key.stride(1) == 1 and value.stride(1) == 1
    assert hidden.is_contiguous()
    if key.dtype != value.dtype or key.dtype != hidden.dtype:
        raise ValueError("key, value, and hidden must have the same dtype")
    if key.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("PLE gate supports BF16 and FP16 inputs")
    gated = torch.empty_like(hidden)
    normed = torch.empty_like(hidden)
    _ple_gate_kernel[(num_tokens, hc)](
        key,
        value,
        hidden,
        norm_key_w,
        norm_query_w,
        norm_conv_w,
        gated,
        normed,
        key.stride(0),
        value.stride(0),
        eps,
        H=h,
        HC=hc,
        BLOCK_H=triton.next_power_of_2(h),
        num_warps=4,
    )
    return gated, normed
