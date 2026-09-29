"""Block-scaled FP8 (e4m3) linear with 32x32 weight blocks, matching DeepSeek-V4.1's reference.

The reference (``inference/model.py`` ``Linear`` + ``inference/kernel.py`` ``fp8_gemm``)
keeps *one* fp8 scale per 32x32 weight block and per 32 activations
(``fp8_block_size = 32  # one fp8 scale per 32x32 weight block / 32 activations``),
both stored as e8m0 ``ue8m0`` codes, and accumulates

  ``C += (A_fp8 @ B_fp8^T) * scale_a[m, k] * scale_b[n // 32, k]``

one 32-K block at a time in FP32 (``kernel.py:208-274``, ``block_K = group_size``).
This module reproduces that on top of the same activation quantizer the 128-block
DeepSeek-V4 path uses (``act_quant`` is already block-parameterized).

Assumes ``K % 32 == 0`` and ``N % 32 == 0`` (true for every V4.1 projection).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from freetoken.kernel.triton.dsv4.fp8_linear import (
    _splitk_reduce_kernel,
    _TL_DTYPE,
    act_quant_fp8,
)
from freetoken.kernel.triton.e4m3_compat import e4m3_kernel_view, e4m3_native_cx, e4m3_u8_to_f32

FP8 = torch.float8_e4m3fn
BLOCK = 32


@triton.jit
def _fp8_32_gemm_kernel(
    a_ptr,            # [M, K] float8_e4m3fn (quantized activation)
    w_ptr,            # [N, K] float8_e4m3fn
    sa_ptr,           # [M, K//32] uint8 (e8m0 act codes)
    sb_ptr,           # [N//32, K//32] uint8 (e8m0 weight codes)
    c_ptr,            # [M, N] compute dtype
    M, N, K,
    stride_am, stride_ak, stride_wn, stride_wk,
    stride_sam, stride_sak, stride_sbn, stride_sbk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    BLOCK: tl.constexpr, compute_type: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    m_mask = offs_m < M
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
    # one scale per 32 output rows: every lane of a 32-wide group reads the same code
    ng = offs_n // BLOCK
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    num_k = tl.cdiv(K, BLOCK_K)
    for k in range(num_k):
        a = tl.load(a_ptrs, mask=m_mask[:, None], other=0.0)
        w = tl.load(w_ptrs)
        if e4m3_native_cx():
            p = tl.dot(a, tl.trans(w), out_dtype=tl.float32)
        else:
            # bf16 dot on the same e4m3 grid: operands exact in bf16, fp32 acc
            p = tl.dot(a, tl.trans(e4m3_u8_to_f32(w).to(tl.bfloat16)), out_dtype=tl.float32)
        sa_code = tl.load(sa_ptr + offs_m * stride_sam + k * stride_sak, mask=m_mask, other=0)
        sca = tl.exp2(sa_code.to(tl.float32) - 127.0)            # [BLOCK_M]
        sb_code = tl.load(sb_ptr + ng * stride_sbn + k * stride_sbk)
        scb = tl.exp2(sb_code.to(tl.float32) - 127.0)            # [BLOCK_N], 32-row groups
        acc += p * sca[:, None] * scb[None, :]
        a_ptrs += BLOCK_K * stride_ak
        w_ptrs += BLOCK_K * stride_wk
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(compute_type), mask=m_mask[:, None])


@triton.jit
def _fp8_32_gemv_splitk_kernel(
    a_ptr,            # [K] float8_e4m3fn
    sa_ptr,           # [K//32] uint8 (e8m0 act codes)
    w_ptr,            # [N, K] float8_e4m3fn
    sb_ptr,           # [N//32, K//32] uint8 (e8m0 weight codes)
    part_ptr,         # [SPLIT_K, N] fp32
    N, K,
    stride_ak, stride_wn, stride_wk, stride_sbn, stride_sbk, stride_pk, stride_pn,
    BLOCK_N: tl.constexpr, SPLIT_K: tl.constexpr, BLOCK_K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    sn = offs_n // BLOCK
    k_per = K // SPLIT_K
    k_start = pid_k * k_per
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k0 in range(0, k_per, BLOCK_K):
        offs_k = k_start + k0 + tl.arange(0, BLOCK_K)
        a = tl.load(a_ptr + offs_k * stride_ak).to(tl.float32)
        w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
        if e4m3_native_cx():
            w = tl.load(w_ptrs, mask=n_mask[:, None], other=0.0).to(tl.float32)
        else:
            w = e4m3_u8_to_f32(tl.load(w_ptrs, mask=n_mask[:, None], other=0))
        kb = (k_start + k0) // BLOCK_K
        sb_code = tl.load(sb_ptr + sn * stride_sbn + kb * stride_sbk, mask=n_mask, other=0)
        scb = tl.exp2(sb_code.to(tl.float32) - 127.0)
        sa_code = tl.load(sa_ptr + kb)
        sca = tl.exp2(sa_code.to(tl.float32) - 127.0)
        acc += tl.sum(w * a[None, :], axis=1) * scb * sca
    tl.store(part_ptr + pid_k * stride_pk + offs_n * stride_pn, acc, mask=n_mask)


def _decode_cfg(N: int, K: int) -> tuple[int, int, int]:
    """Decode GEMV tiling: a 32-wide n block keeps the weight-scale gather trivial.

    SPLIT_K must divide the K-block count, or a split's last 32-wide K tile would read past
    its own rows (40 K blocks / 128 splits used to do exactly that), so it is capped at the
    largest power of two dividing it.
    """
    bn = 16
    n_tiles = triton.cdiv(N, bn)
    sk = max(1, 4096 // n_tiles)
    sk = 1 << (sk.bit_length() - 1)
    k_blocks = K // BLOCK
    return bn, max(1, min(sk, k_blocks & -k_blocks)), 1


def _fp8_32_gemv(a_fp8: torch.Tensor, sa: torch.Tensor, weight: torch.Tensor,
                 sb: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    N, K = weight.shape
    BLOCK_N, split_k, num_warps = _decode_cfg(N, K)
    n_tiles = triton.cdiv(N, BLOCK_N)
    part = torch.empty((split_k, N), dtype=torch.float32, device=a_fp8.device)
    _fp8_32_gemv_splitk_kernel[(n_tiles, split_k)](
        a_fp8, sa, weight, sb, part, N, K,
        a_fp8.stride(0), weight.stride(0), weight.stride(1),
        sb.stride(0), sb.stride(1), part.stride(0), part.stride(1),
        BLOCK_N=BLOCK_N, SPLIT_K=split_k, BLOCK_K=BLOCK, BLOCK=BLOCK, num_warps=num_warps,
    )
    out = torch.empty(N, dtype=out_dtype, device=a_fp8.device)
    _splitk_reduce_kernel[(triton.cdiv(N, 256),)](
        part, out, N, split_k, part.stride(0), part.stride(1),
        BLOCK=256, OUT=_TL_DTYPE[out_dtype], num_warps=2,
    )
    return out


def block_fp8_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """``y = act_quant(x, 32) @ weight^T`` with 32x32 e8m0 weight blocks (reference FP8 path).

    ``x``: ``[..., K]`` bf16; ``weight``: ``[N, K]`` float8_e4m3fn; ``scale``:
    ``[N//32, K//32]`` float8_e8m0fnu. Activation is quantized with a per-32 ue8m0 scale;
    the GEMM applies both scales one 32-K block at a time.
    """
    assert weight.dtype == FP8
    *lead, K = x.shape
    N = weight.shape[0]
    assert weight.shape[1] == K
    assert K % BLOCK == 0 and N % BLOCK == 0, (N, K)
    compute_dtype = x.dtype if x.dtype in _TL_DTYPE else torch.bfloat16
    sb = scale.view(torch.uint8) if scale.dtype == torch.float8_e8m0fnu else scale
    sb = sb.contiguous()
    w = e4m3_kernel_view(weight)

    a_fp8, sa = act_quant_fp8(x, BLOCK)  # [M,K] fp8, [M,K//32] e8m0 codes
    M = a_fp8.shape[0]

    if M == 1:
        out = _fp8_32_gemv(a_fp8[0], sa[0], w, sb, compute_dtype).reshape(*lead, N)
        if bias is not None:
            out = out + bias.to(out.dtype)
        return out

    out = torch.empty((M, N), dtype=compute_dtype, device=x.device)
    BLOCK_M = 32
    BLOCK_N = 128
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _fp8_32_gemm_kernel[grid](
        a_fp8, w, sa, sb, out,
        M, N, K,
        a_fp8.stride(0), a_fp8.stride(1), w.stride(0), w.stride(1),
        sa.stride(0), sa.stride(1), sb.stride(0), sb.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK, BLOCK=BLOCK,
        compute_type=_TL_DTYPE[compute_dtype], num_warps=4, num_stages=3,
    )
    out = out.reshape(*lead, N)
    if bias is not None:
        out = out + bias.to(out.dtype)
    return out


__all__ = ["block_fp8_linear"]
