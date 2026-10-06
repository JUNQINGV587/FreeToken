"""Routed FP4 expert compute on the offload-cache banks (DeepSeek-V4).

Composes the DeepSeek-FP4 fused-MoE kernels (``dsv4_fused_moe``) into the full
routed-expert path: ``gate_up -> swiglu(limit) -> down``, reading expert weights
directly from the resident banks (no bf16 materialization). ``slots`` maps each
(token, route) to a bank row: a cache slot for the decode GEMV path, a
streamed full-layer position (== expert id) for the grouped prefill path.
"""

from __future__ import annotations

import os

import torch
import triton

from freetoken.kernel.triton.dsv4.fp8_linear import (
    act_quant_fp8,
    act_quant_fp8_inplace,
    act_quant_fp8_roundtrip,
)
from freetoken.kernel.triton.dsv4.fused_moe import (
    _decode_dsfp4_moe_kernel,
    _e2m1_lut,
    _prefill_dsfp4_moe_kernel,
    fused_swiglu,
)
from freetoken.moe.fused import moe_align_block_size

_TL_DTYPE = None


def _compute_type(dtype: torch.dtype):
    import triton.language as tl

    return {
        torch.bfloat16: tl.bfloat16,
        torch.float16: tl.float16,
        torch.float32: tl.float32,
        # FP8_MMA A-side (native fp8 act_quant output); unused by the bf16 path.
        torch.float8_e4m3fn: tl.float8e4nv,
    }[dtype]


def _grouped_decode(
    a: torch.Tensor,            # [A_rows, K] compute dtype
    packed_cache: torch.Tensor,  # [S, N, K//2] uint8
    scale_cache: torch.Tensor,   # [S, N, K//32] e8m0
    slots: torch.Tensor,         # [T, top_k] int32 -> cache slot
    topk_weights: torch.Tensor | None,
    *,
    a_row_is_route: bool,
    mul_routed_weight: bool,
) -> torch.Tensor:
    """Grouped per-route GEMM ``c[t,r,:] = a[row] @ dequant(W[slot])^T``.

    Returns ``[T, top_k, N]``. Used for both gate_up (a_row=token) and down
    (a_row=route). FP4 weights are dequantized inline from the slot cache.
    """
    T, top_k = slots.shape
    N = packed_cache.shape[1]
    K = packed_cache.shape[2] * 2
    total_routes = T * top_k
    dtype = a.dtype
    out = torch.empty((T, top_k, N), dtype=dtype, device=a.device)
    if topk_weights is None:
        topk_weights = out.new_empty((1, 1), dtype=torch.float32)
    scale_u8 = scale_cache.view(torch.uint8)

    # Decode is a per-route GEMV, bound by the inline FP4 dequant (LUT gather + block
    # scale), not weight HBM bandwidth -- so it tops out ~29% of HBM peak. BN=16/
    # BKB=128/1 warp maximizes memory-level parallelism (many single-warp CTAs, no
    # cross-warp reduction) -> ~950 GB/s on H100 (was ~830 at BN=8/BKB=256/2). K_BYTES
    # must be a multiple of BLOCK_SIZE_KB (gate_up 2048, down 1024 -- both /128).
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_KB = 128
    _NW = 1
    assert (K // 2) % BLOCK_SIZE_KB == 0, (K, BLOCK_SIZE_KB)
    grid = (total_routes, triton.cdiv(N, BLOCK_SIZE_N))
    _decode_dsfp4_moe_kernel[grid](
        a, packed_cache, scale_u8, out, topk_weights, slots,
        _e2m1_lut(a.device.index),
        total_routes, N, K,
        a.stride(0), a.stride(1),
        packed_cache.stride(0), packed_cache.stride(1), packed_cache.stride(2),
        scale_u8.stride(0), scale_u8.stride(1), scale_u8.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        topk_weights.stride(0) if topk_weights.ndim == 2 else 0,
        topk_weights.stride(1) if topk_weights.ndim == 2 else 0,
        slots.stride(0), slots.stride(1),
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_KB=BLOCK_SIZE_KB,
        TOP_K=top_k,
        A_ROW_IS_ROUTE=a_row_is_route,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        compute_type=_compute_type(dtype),
        num_warps=_NW,
    )
    return out


def routed_experts_fp4(
    x: torch.Tensor,             # [T, H] compute dtype
    slots: torch.Tensor,         # [T, top_k] int32 -> cache slot
    topk_weights: torch.Tensor,  # [T, top_k] fp32 (incl. route_scale + renorm)
    gate_up_packed: torch.Tensor,  # [S, 2I, H//2] uint8
    gate_up_scale: torch.Tensor,   # [S, 2I, H//32] e8m0
    down_packed: torch.Tensor,     # [S, H, I//2] uint8
    down_scale: torch.Tensor,      # [S, H, I//32] e8m0
    swiglu_limit: float,
) -> torch.Tensor:
    """Full routed-expert output (summed over the top-k routes), excludes shared expert.

    Precision matches the reference ``fp4_gemm(act_quant(x, 128), W_fp4)``: the gate_up
    and down activations are FP8-round-tripped (block 128, ue8m0) before each GEMM. Since
    an fp8 value x pow2 scale is exact in bf16, the round-tripped activation entering the
    bf16 decode kernel is bit-identical to the reference's dequantized FP8 activation
    (validated max diff = 0 vs the tilelang ``fp4_gemm`` reference)."""
    T, top_k = slots.shape
    H = x.shape[1]
    two_I = gate_up_packed.shape[1]
    I = two_I // 2

    x = act_quant_fp8_roundtrip(x, 128)  # gate_up activation -> FP8 round-trip (no clone)
    gate_up = _grouped_decode(
        x, gate_up_packed, gate_up_scale, slots, None,
        a_row_is_route=False, mul_routed_weight=False,
    )  # [T, top_k, 2I]
    act = fused_swiglu(gate_up, swiglu_limit)  # [T, top_k, I]

    act = act.reshape(T * top_k, I)
    act_quant_fp8_inplace(act, 128)  # down activation -> FP8 round-trip
    down = _grouped_decode(
        act, down_packed, down_scale, slots, topk_weights,
        a_row_is_route=True, mul_routed_weight=True,
    )  # [T, top_k, H]
    return down.sum(dim=1)  # [T, H]


# Above this the grouped GEMM beats the per-route GEMV despite its padding;
# below, the GEMV's exact routes*N work wins (short streaming chunks). The
# grouped kernel sits on its dequant floor (~6.4-6.9ms/GEMM-pair at DSV4
# geometry) for any chunk size, so the crossover is where the GEMV's
# routes-proportional cost reaches that floor (H100 sweep).
_GROUPED_MIN_ROUTES = 768


def _grouped_prefill(
    a: torch.Tensor,             # [A_rows, K] compute dtype (FP8 round-tripped)
    packed_cache: torch.Tensor,  # [S, N, K//2] uint8
    scale_cache: torch.Tensor,   # [S, N, K//32] e8m0
    c: torch.Tensor,             # [T, top_k, N] output, flat-indexed over routes
    tw_flat: torch.Tensor,       # [T*top_k] fp32
    sorted_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    num_valid: int,
    kernel_top_k: int,
    mul_routed_weight: bool,
    cfg: dict,
    fp8_mma: bool = False,
    a_scale: torch.Tensor | None = None,  # [A_rows, K//128] ue8m0 (fp8_mma only)
) -> None:
    assert not fp8_mma or a_scale is not None
    N = packed_cache.shape[1]
    K = packed_cache.shape[2] * 2
    EM = sorted_ids.shape[0]
    scale_u8 = scale_cache.view(torch.uint8)
    grid = lambda META: (  # noqa: E731
        triton.cdiv(EM, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    _prefill_dsfp4_moe_kernel[grid](
        a, packed_cache, scale_u8, c, tw_flat,
        sorted_ids, expert_ids, num_tokens_post_padded,
        N, K, EM, num_valid,
        a.stride(0), a.stride(1),
        a_scale if fp8_mma else a,
        a_scale.stride(0) if fp8_mma else 0, a_scale.stride(1) if fp8_mma else 0,
        packed_cache.stride(0), packed_cache.stride(1), packed_cache.stride(2),
        scale_u8.stride(0), scale_u8.stride(1), scale_u8.stride(2),
        c.stride(-2), c.stride(-1),
        BLOCK_SIZE_M=cfg["BLOCK_SIZE_M"],
        BLOCK_SIZE_N=cfg["BLOCK_SIZE_N"],
        BLOCK_SIZE_K=cfg["BLOCK_SIZE_K"],
        GROUP_SIZE_M=cfg["GROUP_SIZE_M"],
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=kernel_top_k,
        # Output dtype, not A's: with fp8_mma the A operand is e4m3 but the
        # accumulator must store as c's dtype (else fp32->e4m3 saturation/rounding).
        compute_type=_compute_type(c.dtype),
        FP8_MMA=fp8_mma,
        num_warps=cfg.get("num_warps", 4),
        num_stages=cfg.get("num_stages", 3),
    )


def routed_experts_fp4_prefill(
    x: torch.Tensor,             # [T, H] compute dtype
    slots: torch.Tensor,         # [T, top_k] int32 -> bank row in [0, num_rows)
    topk_weights: torch.Tensor,  # [T, top_k] fp32 (incl. route_scale + renorm)
    gate_up_packed: torch.Tensor,  # [S, 2I, H//2] uint8
    gate_up_scale: torch.Tensor,   # [S, 2I, H//32] e8m0
    down_packed: torch.Tensor,     # [S, H, I//2] uint8
    down_scale: torch.Tensor,      # [S, H, I//32] e8m0
    swiglu_limit: float,
    num_rows: int,
) -> torch.Tensor:
    """Grouped-GEMM counterpart of :func:`routed_experts_fp4` for dense prefill
    chunks: one moe_align sort shared by both GEMMs, each expert's weights
    dequantized once per N-tile instead of once per route. Same FP8
    round-tripped activations; differs from the GEMV only in fp32 accumulation
    order (tl.dot tree vs sequential K-walk)."""
    T, top_k = slots.shape
    if T * top_k < _GROUPED_MIN_ROUTES:
        return routed_experts_fp4(
            x, slots, topk_weights,
            gate_up_packed, gate_up_scale, down_packed, down_scale, swiglu_limit,
        )
    H = x.shape[1]
    two_I = gate_up_packed.shape[1]
    I = two_I // 2
    routes = T * top_k
    # FT_DSFP4_FP8=1 (P4, DeepGEMM recipe on Ada): run the grouped GEMM on the
    # FP8 tensor cores -- B stays e4m3 (exact e2m1 superset), one 32-wide K
    # block per dot == one e8m0 scale block == sm89 mma.sync K=32, and the
    # scale multiplies the fp32 partial dot (exact pow2). A-side values are
    # FP8 round-tripped already, so the in-kernel bf16->e4m3 cast is exact.
    fp8_mma = os.environ.get("FT_DSFP4_FP8", "0") == "1"
    # One static config per MMA path (no autotune). sm89 sweep at v41
    # geometry (T=25088, H=5120, I=2304, 192 slots, gate_up + down): the
    # dequant-floor-bound kernel wants wide M tiles with few warps --
    # BM=128/BK=128/NW=4/NS=1 measures -30% on both GEMMs vs the old
    # BM=64/BK=64/NW=8 (67.2 vs 47.0 TF gate_up, 67.3 vs 47.2 TF down).
    # FP8_MMA keeps BK=32 (== one e8m0 block == one mma.sync K) and is
    # slower than tuned bf16 on sm89 (58.9 TF best) -- kept as an experiment
    # behind FT_DSFP4_FP8, not the default.
    cfg = dict(
        BLOCK_SIZE_M=128, BLOCK_SIZE_N=64,
        BLOCK_SIZE_K=32 if fp8_mma else 128,
        GROUP_SIZE_M=8,
        num_warps=4, num_stages=1,
    )
    sorted_ids, expert_ids, ntpp = moe_align_block_size(slots, cfg["BLOCK_SIZE_M"], num_rows)
    tw = topk_weights.reshape(-1).contiguous()

    if fp8_mma:
        # Raw e4m3 grid + separate per-128 ue8m0 scales (the round-trip folds
        # the block scale into the value, which saturates e4m3 at ±448).
        xq, xs = act_quant_fp8(x, 128)
    else:
        xq = act_quant_fp8_roundtrip(x, 128)  # gate_up activation -> FP8 round-trip
        xs = None
    gate_up = torch.empty((T, top_k, two_I), dtype=x.dtype, device=x.device)
    _grouped_prefill(
        xq, gate_up_packed, gate_up_scale, gate_up, tw,
        sorted_ids, expert_ids, ntpp, routes, top_k, False, cfg, fp8_mma, xs,
    )
    act = fused_swiglu(gate_up, swiglu_limit)  # [T, top_k, I]

    act = act.reshape(routes, I)
    if fp8_mma:
        actq, acts = act_quant_fp8(act, 128)
    else:
        act_quant_fp8_inplace(act, 128)  # down activation -> FP8 round-trip
        actq, acts = act, None
    down = torch.empty((T, top_k, H), dtype=x.dtype, device=x.device)
    _grouped_prefill(
        actq, down_packed, down_scale, down, tw,
        sorted_ids, expert_ids, ntpp, routes, 1, True, cfg, fp8_mma, acts,
    )
    return down.sum(dim=1)  # [T, H]


__all__ = ["routed_experts_fp4", "routed_experts_fp4_prefill", "_grouped_decode"]
