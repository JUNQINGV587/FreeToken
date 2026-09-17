# SPDX-License-Identifier: Apache-2.0
"""Fused hyper-connection prefill kernels for qwen4_exp (FREETOKEN_HC_PREFILL_FUSED).

Two fusions around the two cuBLAS GEMMs, which are NOT replaced:

  K1 ``hc_prefill_combine_norm``  = hc_combine + next block's grouped_gemma_rmsnorm
  G1 cuBLAS down GEMM             = unchanged (merged [lowrank+hc+pad, hc*hidden])
  K2 ``hc_prefill_up_mix``        = hc_silu + up GEMM + hc_gate_mix

Rounding-point spec (bit contract vs the unfused path, rounds kept identical):

  R1  rn     = grouped_gemma_rmsnorm(R):  bf16 -> fp32, rrms = rsqrt(sum(x*x)/G + eps)
              with the reference's single tl.sum over next_pow2(G)=4096 lanes;
              y = x*rrms; y += y*w (FMA order); -> STORE bf16
  R2  down   = F.linear(rn, Wd)           cuBLAS bf16 in / fp32 acc / bf16 STORE.
              At M=8192 cuBLAS picks ampere_bf16_s1688gemm_128x64_sliced1x2 (k8 mma
              + 2-way sliced-k, plus a workspace Memset); at M=2048/4096 it picks
              s1688gemm_128x128 (no slice). A Triton re-implementation does NOT
              reproduce these bits (probed: ~45% of outputs differ at every M and
              KB), so G1 stays cuBLAS and rn/down stay materialized.
  R3  a      = hc_silu(lora):             bf16 -> fp32, x/HC, x*sigmoid(x) -> STORE bf16
  R4  gate   = F.linear(a, Wu)            cuBLAS s16816gemm (k16 mma, NO split-k at
              every M probed: 2048/4096/8192). Per-output accumulation is one
              thread's strictly ascending sequence of k16 mma steps, which a Triton
              tl.dot k-loop (KB multiple of 16, ascending) reproduces bit-exactly
              (probed: 0 mismatches at M in {2048,4096,8192}, KB in {32,64}).
  R5  x      = hc_gate_mix(rn, gate):     gate bf16 -> fp32 sigmoid; rn bf16 -> fp32;
              acc += sigmoid(gate_c)*rn_c for c in 0..HC-1 (stream-sequential);
              acc /= HC -> STORE bf16
  R6  R'     = hc_combine(R, y, s):       inj_c = 2*sigmoid(s_c/HC) fp32;
              out = R + y*inj_c fp32 -> STORE bf16

K1 = R6 + R1: the combine result is rounded to bf16 (stored to R') BEFORE the norm
reads it, matching the unfused combine -> norm HBM boundary; the norm then uses the
reference's exact lane layout (4096 padded) and FMA chain, so rn is bit-identical.
K2 = R3 + R4 + R5: silu is rounded to bf16 per element before the dot (same value as
the materialized hc_silu output); the dot accumulates over K=320 in ascending KB=64
steps; the gate accumulator is rounded to bf16 before sigmoid (same value as the
materialized cuBLAS output); the mix keeps the reference's stream-sequential order
and single /HC at the end. All tile-shape choices (BM/BN/warps) leave the per-element
accumulation order untouched, so K2 is bit-exact for any config with KB % 16 == 0.

Gate: env FREETOKEN_HC_PREFILL_FUSED=1 AND M >= HC_PREFILL_FUSED_MIN_M AND bf16
weights. Below the M gate the caller keeps the production kernels (the up GEMM's
bit contract is only probed in the prefill regime; small-M cuBLAS may pick gemv
kernels with a different accumulation order).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from freetoken.kernel.triton.hc import _pdl_supported

# Prefill regime only; below this the dispatch keeps the production path.
HC_PREFILL_FUSED_MIN_M = 2048


@triton.jit
def _hc_prefill_combine_norm_kernel(
    res_ptr,
    block_ptr,
    inj_ptr,
    w_ptr,
    out_ptr,
    rn_ptr,
    stride_res,
    stride_block,
    stride_inj,
    stride_out,
    stride_rn,
    HC_DIM: tl.constexpr,
    HC: tl.constexpr,
    W_SHARED: tl.constexpr,
    EPS: tl.constexpr,
    launch_pdl: tl.constexpr,
) -> None:
    # One program per (row, stream); lane layout identical to
    # _grouped_gemma_rmsnorm_kernel so the reduction tree (and the bits) match.
    BLOCK_SIZE: tl.constexpr = triton.next_power_of_2(HC_DIM)

    row = tl.program_id(0).to(tl.int64)
    stream = tl.program_id(1)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < HC_DIM

    if launch_pdl:
        tl.extra.cuda.gdc_wait()

    # --- combine (R6): scalar inject, same elementwise math as _hc_combine_kernel
    s = tl.load(inj_ptr + row * stride_inj + stream)
    inj = 2.0 * tl.sigmoid(s.to(tl.float32) / HC)
    res = tl.load(res_ptr + row * stride_res + stream * HC_DIM + offs, mask, other=0.0)
    block = tl.load(block_ptr + row * stride_block + offs, mask, other=0.0)
    out = res.to(tl.float32) + block.to(tl.float32) * inj
    out = out.to(out_ptr.dtype.element_ty)  # R6 round: bf16 boundary store
    tl.store(out_ptr + row * stride_out + stream * HC_DIM + offs, out, mask)

    # --- norm (R1 of the next block) on the ROUNDED combine result
    x = out.to(tl.float32)
    rrms = tl.rsqrt(tl.sum(x * x) / HC_DIM + EPS)
    w_offs = offs if W_SHARED else stream * HC_DIM + offs
    w = tl.load(w_ptr + w_offs, mask, other=0.0)
    y = x * rrms
    y += y * w.to(tl.float32)

    if launch_pdl:
        tl.extra.cuda.gdc_launch_dependents()
    tl.store(rn_ptr + row * stride_rn + stream * HC_DIM + offs, y, mask)


def hc_prefill_combine_norm(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    hc_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused hc_combine + grouped_gemma_rmsnorm; returns (R' bf16, rn bf16)."""
    N, DIM = residual.shape
    assert DIM % hc_count == 0
    hc_dim = DIM // hc_count
    assert block_output.shape == (N, hc_dim)
    assert injection_logits.shape == (N, hc_count)
    assert residual.stride(1) == 1
    assert block_output.stride(1) == 1
    assert injection_logits.stride(1) == 1
    assert norm_weight.is_contiguous()
    assert norm_weight.numel() in (hc_dim, DIM)

    out = residual.new_empty(residual.shape)
    rn = residual.new_empty(residual.shape)
    # num_warps is part of the bit contract: the tl.sum reduction tree over the
    # 4096 padded lanes changes with the warp count (probed: w2/w8 mismatch, w4
    # matches), and the reference grouped_gemma_rmsnorm launches at the default 4.
    _hc_prefill_combine_norm_kernel[(N, hc_count)](
        residual,
        block_output,
        injection_logits,
        norm_weight,
        out,
        rn,
        residual.stride(0),
        block_output.stride(0),
        injection_logits.stride(0),
        out.stride(0),
        rn.stride(0),
        HC_DIM=hc_dim,
        HC=hc_count,
        W_SHARED=norm_weight.numel() == hc_dim,
        EPS=eps,
        launch_pdl=_pdl_supported(),
        num_warps=4,
    )
    return out, rn


@triton.jit
def _hc_prefill_up_mix_kernel(
    lora_ptr,  # [T, >=L] bf16, the down GEMM's lora slice (row stride may exceed L)
    rn_ptr,  # [T, HC*HC_DIM] bf16
    wu_ptr,  # [HC*HC_DIM, L] bf16 up weight
    x_ptr,  # [T, HC_DIM] bf16 out
    stride_lora,
    stride_rn,
    stride_wu,
    stride_x,
    M,
    L: tl.constexpr,
    HC_DIM: tl.constexpr,
    HC: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    KB: tl.constexpr,
    launch_pdl: tl.constexpr,
) -> None:
    pid_m = tl.program_id(0)
    tile_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    mask_m = offs_m < M
    offs_o = tile_n * BN + tl.arange(0, BN)
    mask_o = offs_o < HC_DIM

    if launch_pdl:
        tl.extra.cuda.gdc_wait()

    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for stream in tl.static_range(HC):
        # up GEMM for this stream's output tile: ascending k16-mma order via a
        # KB-stepped tl.dot loop, bit-matching the cuBLAS s16816 kernel (R4).
        gate = tl.zeros([BM, BN], dtype=tl.float32)
        for k0 in range(0, L, KB):
            offs_k = k0 + tl.arange(0, KB)
            mask_k = offs_k < L
            a = tl.load(
                lora_ptr + offs_m[:, None] * stride_lora + offs_k[None, :],
                mask=(mask_m[:, None]) & mask_k[None, :],
                other=0.0,
            )
            # R3: silu rounded to bf16 per element before the dot, matching the
            # materialized hc_silu output the reference GEMM consumes.
            t = a.to(tl.float32) / HC
            silu = (t * tl.sigmoid(t)).to(tl.bfloat16)
            w = tl.load(
                wu_ptr + (stream * HC_DIM + offs_o)[:, None] * stride_wu + offs_k[None, :],
                mask=mask_o[:, None] & mask_k[None, :],
                other=0.0,
            )
            gate += tl.dot(silu, tl.trans(w), out_dtype=tl.float32)
        # R4 round: the reference reads the cuBLAS bf16 store back.
        gate = gate.to(tl.bfloat16).to(tl.float32)
        rn = tl.load(
            rn_ptr + offs_m[:, None].to(tl.int64) * stride_rn + (stream * HC_DIM + offs_o)[None, :],
            mask=(mask_m[:, None]) & mask_o[None, :],
            other=0.0,
        )
        # R5: stream-sequential fp32 mix.
        acc += tl.sigmoid(gate) * rn.to(tl.float32)
    acc /= HC

    if launch_pdl:
        tl.extra.cuda.gdc_launch_dependents()
    tl.store(
        x_ptr + offs_m[:, None].to(tl.int64) * stride_x + offs_o[None, :],
        acc,
        mask=(mask_m[:, None]) & mask_o[None, :],
    )


def hc_prefill_up_mix(
    lora: torch.Tensor,
    rn: torch.Tensor,
    up_weight: torch.Tensor,
    hc_count: int,
    *,
    BM: int = 32,
    BN: int = 128,
    KB: int = 32,
    num_warps: int = 4,
    num_stages: int = 3,
) -> torch.Tensor:
    """Fused hc_silu + up GEMM + hc_gate_mix. Returns x [T, hc_dim] bf16.

    Bit-exact vs the unfused chain (spec above; accumulation order is independent of
    BM/BN/warps/stages, probed at KB in {32, 64, 128}). Perf-neutral on sm_89 (L20):
    a Triton bf16 GEMM tops out ~65 TFLOPS at this K=320 shape vs cuBLAS s16816's
    ~105, which eats the saved gate round-trip; the model wiring therefore keeps the
    cuBLAS up GEMM and this kernel is kept for re-evaluation on sm_90+.

    ``lora`` may be a row-strided view (the down GEMM's ``[:, :lowrank]`` slice);
    ``up_weight`` is the raw [hc_count*hc_dim, lowrank] linear weight.
    """
    T, DIM = rn.shape
    assert DIM % hc_count == 0
    hc_dim = DIM // hc_count
    L = lora.shape[1]
    assert lora.shape[0] == T and lora.stride(1) == 1
    assert rn.stride(1) == 1
    assert up_weight.shape == (DIM, L) and up_weight.stride(1) == 1
    assert KB % 16 == 0, "bit contract: k-loop steps must be whole k16 mma chunks"

    x = rn.new_empty(T, hc_dim)
    M = T
    grid = (triton.cdiv(M, BM), triton.cdiv(hc_dim, BN))
    _hc_prefill_up_mix_kernel[grid](
        lora,
        rn,
        up_weight,
        x,
        lora.stride(0),
        rn.stride(0),
        up_weight.stride(0),
        x.stride(0),
        M,
        L=L,
        HC_DIM=hc_dim,
        HC=hc_count,
        BM=BM,
        BN=BN,
        KB=KB,
        launch_pdl=_pdl_supported(),
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return x


__all__ = [
    "HC_PREFILL_FUSED_MIN_M",
    "hc_prefill_combine_norm",
    "hc_prefill_up_mix",
]
