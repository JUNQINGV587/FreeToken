"""Host orchestration for the inline-dequant NVFP4 fused-MoE path.

Mirrors :mod:`freetoken.moe.fused` (gemm1 -> act -> gemm2 -> sum-reduce) but the two
grouped GEMMs read the NVFP4 expert cache directly and dequantize inside the K-loop,
so no BF16 copy of the experts is ever materialized.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict

import torch
import triton
import triton.language as tl

from freetoken.kernel import moe_sum_reduce_triton
from freetoken.kernel.triton.e4m3_compat import e4m3_kernel_view
from freetoken.kernel.triton.nvfp4_fused_moe import (
    _decode_nvfp4_marlin_kernel,
    _decode_nvfp4_moe_kernel,
    _e2m1_lut,
    _prefill_nvfp4_moe_kernel,
)
from freetoken.layers import gated_act_and_mul
from freetoken.moe.fused import moe_align_block_size

def _decode_env_int(name: str, default: int) -> int:
    # Read once at import: decode runs inside a CUDA graph, so a runtime flip would
    # desync the captured launch geometry from the config the launch then reads.
    value = int(os.environ.get(name, "0") or 0)
    return value if value > 0 else default


# Decode is captured into a CUDA graph, so the config must be fixed (no triton.autotune,
# which benchmarks at run time). Tuned offline against the NVFP4 decode kernels.
# These drive the original LUT-gather decode (_decode_gemm), kept only for A/B.
_DECODE_BLOCK_N = _decode_env_int("FREETOKEN_NVFP4_DECODE_BN", 64)
_DECODE_BLOCK_KB = _decode_env_int("FREETOKEN_NVFP4_DECODE_BKB", 128)
_DECODE_WARPS = _decode_env_int("FREETOKEN_NVFP4_DECODE_WARPS", 4)

# Marlin-style decode config (int32 wide loads + deferred reduction). Offline sweep over
# the qwen35/qwen3moe (I=512/768) decode shapes picked BLOCK_N=16, BLOCK_KW=16 (== 128
# k-values/iter), 4 warps -- the wide load lifts the gate/up GEMM ~43%->~51% of peak BW.
# The FREETOKEN_NVFP4_DECODE_MARLIN_* envs let a sweep retune this without a rebuild;
# unset (or 0) keeps the values above, so an env-free run is unchanged.
_DECODE_MARLIN_BLOCK_N = _decode_env_int("FREETOKEN_NVFP4_DECODE_MARLIN_BN", 16)
_DECODE_MARLIN_BLOCK_KW = _decode_env_int("FREETOKEN_NVFP4_DECODE_MARLIN_BKW", 16)
_DECODE_MARLIN_WARPS = _decode_env_int("FREETOKEN_NVFP4_DECODE_MARLIN_WARPS", 4)
# Deep-K variant: at K > 2048 (qwen4_exp gate_up, K=2560) a narrower N tile with the whole
# K strip in one program iteration measures ~13% faster (18.6 vs 21.0us); short-K shapes
# regress under it, so the split is by K, not by gemm position.
_DECODE_MARLIN_DEEPK_BLOCK_N = _decode_env_int("FREETOKEN_NVFP4_DECODE_MARLIN_DEEPK_BN", 8)
_DECODE_MARLIN_DEEPK_BLOCK_KW = _decode_env_int("FREETOKEN_NVFP4_DECODE_MARLIN_DEEPK_BKW", 128)
_DECODE_MARLIN_DEEPK_THRESHOLD = _decode_env_int("FREETOKEN_NVFP4_DECODE_MARLIN_DEEPK_THRESHOLD", 2048)


def _tl_dtype(dt: torch.dtype):
    if dt == torch.bfloat16:
        return tl.bfloat16
    if dt == torch.float16:
        return tl.float16
    return tl.float32


def _decode_gemm(
    a: torch.Tensor,
    packed: torch.Tensor,
    scale: torch.Tensor,
    glob: torch.Tensor,
    c: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    mul_routed_weight: bool,
    a_row_is_route: bool,
) -> None:
    M, top_k = topk_ids.shape
    N = packed.shape[1]
    K = packed.shape[2] * 2
    scale = e4m3_kernel_view(scale)
    total_routes = M * top_k
    grid = (total_routes, triton.cdiv(N, _DECODE_BLOCK_N))
    _decode_nvfp4_moe_kernel[grid](
        a, packed, scale, glob, c, topk_weights, topk_ids,
        _e2m1_lut(a.device.index),
        total_routes, N, K,
        a.stride(0), a.stride(1),
        packed.stride(0), packed.stride(1), packed.stride(2),
        scale.stride(0), scale.stride(1), scale.stride(2),
        glob.stride(0), glob.stride(1),
        c.stride(0), c.stride(1), c.stride(2),
        topk_weights.stride(0), topk_weights.stride(1),
        topk_ids.stride(0), topk_ids.stride(1),
        BLOCK_SIZE_N=_DECODE_BLOCK_N,
        BLOCK_SIZE_KB=_DECODE_BLOCK_KB,
        TOP_K=top_k,
        A_ROW_IS_ROUTE=a_row_is_route,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        compute_type=_tl_dtype(c.dtype),
        num_warps=_DECODE_WARPS,
    )


def _decode_gemm_marlin(
    a: torch.Tensor,
    packed: torch.Tensor,
    scale: torch.Tensor,
    glob: torch.Tensor,
    c: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    mul_routed_weight: bool,
    a_row_is_route: bool,
) -> None:
    """Marlin-style decode GEMV: int32 wide loads + deferred reduction
    (:func:`_decode_nvfp4_marlin_kernel`). ``packed`` is the uint8 ``[S, N, K//2]`` bank;
    it is reinterpreted as int32 ``[S, N, K//8]`` (contiguous, K%8==0 for NVFP4)."""
    M, top_k = topk_ids.shape
    N = packed.shape[1]
    K = packed.shape[2] * 2
    packed_i32 = packed.view(torch.int32)  # [S, N, K // 8]
    scale = e4m3_kernel_view(scale)
    total_routes = M * top_k
    deep_k = K > _DECODE_MARLIN_DEEPK_THRESHOLD
    block_n = _DECODE_MARLIN_DEEPK_BLOCK_N if deep_k else _DECODE_MARLIN_BLOCK_N
    block_kw = _DECODE_MARLIN_DEEPK_BLOCK_KW if deep_k else _DECODE_MARLIN_BLOCK_KW
    grid = (total_routes, triton.cdiv(N, block_n))
    _decode_nvfp4_marlin_kernel[grid](
        a, packed_i32, scale, glob, c, topk_weights, topk_ids,
        _e2m1_lut(a.device.index),
        total_routes, N, K,
        a.stride(0), a.stride(1),
        packed_i32.stride(0), packed_i32.stride(1), packed_i32.stride(2),
        scale.stride(0), scale.stride(1), scale.stride(2),
        glob.stride(0), glob.stride(1),
        c.stride(0), c.stride(1), c.stride(2),
        topk_weights.stride(0), topk_weights.stride(1),
        topk_ids.stride(0), topk_ids.stride(1),
        BLOCK_SIZE_N=block_n,
        BLOCK_SIZE_KW=block_kw,
        TOP_K=top_k,
        A_ROW_IS_ROUTE=a_row_is_route,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        compute_type=_tl_dtype(c.dtype),
        num_warps=_DECODE_MARLIN_WARPS,
    )


def _fused_experts_decode_nvfp4(
    gemm_fn,
    hidden_states: torch.Tensor,
    gate_up_packed: torch.Tensor,
    gate_up_scale: torch.Tensor,
    gate_up_global: torch.Tensor,
    down_packed: torch.Tensor,
    down_scale: torch.Tensor,
    down_global: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str,
    apply_router_weight_on_input: bool,
    act_alpha: float = 1.702,
    act_limit: float = 7.0,
) -> torch.Tensor:
    """Shared decode body (gemm1 -> act -> gemm2 -> sum-reduce); ``gemm_fn`` is either
    the marlin-style int32 GEMV (:func:`_decode_gemm_marlin`) or the original LUT-gather
    GEMV (:func:`_decode_gemm`), both with the same calling convention."""
    M, H = hidden_states.shape
    top_k = topk_ids.shape[1]
    two_i = gate_up_packed.shape[1]
    inter = two_i // 2
    dev, dt = hidden_states.device, hidden_states.dtype

    ic1 = torch.empty((M, top_k, two_i), device=dev, dtype=dt)
    gemm_fn(
        hidden_states, gate_up_packed, gate_up_scale, gate_up_global,
        ic1, topk_weights, topk_ids, apply_router_weight_on_input, False,
    )
    ic2 = torch.empty((M * top_k, inter), device=dev, dtype=dt)
    gated_act_and_mul(activation, ic1.view(-1, two_i), ic2, alpha=act_alpha, limit=act_limit)
    ic3 = torch.empty((M, top_k, H), device=dev, dtype=dt)
    gemm_fn(
        ic2, down_packed, down_scale, down_global,
        ic3, topk_weights, topk_ids, not apply_router_weight_on_input, True,
    )
    out = torch.empty_like(hidden_states)
    moe_sum_reduce_triton(ic3, out)
    return out


def fused_experts_decode_nvfp4_marlin(
    hidden_states: torch.Tensor,
    gate_up_packed: torch.Tensor,
    gate_up_scale: torch.Tensor,
    gate_up_global: torch.Tensor,
    down_packed: torch.Tensor,
    down_scale: torch.Tensor,
    down_global: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    act_alpha: float = 1.702,
    act_limit: float = 7.0,
) -> torch.Tensor:
    """Decode inline-NVFP4 MoE using the Marlin-style int32 wide-load GEMV."""
    return _fused_experts_decode_nvfp4(
        _decode_gemm_marlin,
        hidden_states, gate_up_packed, gate_up_scale, gate_up_global,
        down_packed, down_scale, down_global,
        topk_weights, topk_ids, activation, apply_router_weight_on_input,
        act_alpha, act_limit,
    )


def fused_experts_decode_nvfp4_serial(
    hidden_states: torch.Tensor,
    gate_up_packed: torch.Tensor,
    gate_up_scale: torch.Tensor,
    gate_up_global: torch.Tensor,
    down_packed: torch.Tensor,
    down_scale: torch.Tensor,
    down_global: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    act_alpha: float = 1.702,
    act_limit: float = 7.0,
) -> torch.Tensor:
    """Original LUT-gather decode (one program per route, full K reduction). Retained for
    A/B benchmarking against the marlin decode path; not on the production decode path."""
    return _fused_experts_decode_nvfp4(
        _decode_gemm,
        hidden_states, gate_up_packed, gate_up_scale, gate_up_global,
        down_packed, down_scale, down_global,
        topk_weights, topk_ids, activation, apply_router_weight_on_input,
        act_alpha, act_limit,
    )


def _prefill_config(M: int) -> Dict[str, int]:
    # ``BLOCK_SIZE_M`` is coupled to host-side ``moe_align_block_size`` (token padding),
    # so it cannot be picked by triton.autotune. Table from an L20 (sm_89) sweep on the
    # qwen4_exp geometry (research/bench/probe_nvfp4_prefill_tiles.py): at high route
    # density the kernel is bound by expert-weight re-reads (sum_e ceil(routes_e/BM)),
    # so BM=128 wins (+21-32% at M>=4096); near M=1024 padding compute dominates and
    # BM=32 stays. BLOCK_SIZE_KB is 32 in every row, so the per-route fp32 reduction
    # order -- and therefore the outputs -- is identical across the table.
    if M <= 64:
        cfg = dict(BLOCK_SIZE_M=16, BLOCK_SIZE_N=64, BLOCK_SIZE_KB=32,
                   GROUP_SIZE_M=1, num_warps=8, num_stages=4)
    elif M < 2048:
        cfg = dict(BLOCK_SIZE_M=32, BLOCK_SIZE_N=64, BLOCK_SIZE_KB=32,
                   GROUP_SIZE_M=8, num_warps=8, num_stages=2)
    else:
        cfg = dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=64, BLOCK_SIZE_KB=32,
                   GROUP_SIZE_M=8, num_warps=8, num_stages=2)
    bm_override = int(os.environ.get("FREETOKEN_NVFP4_PREFILL_BM", "0"))
    if bm_override > 0 and M > 64:
        cfg["BLOCK_SIZE_M"] = bm_override
    # Wide-load prototype dispatch (moe/fused_nvfp4_wide.py): env-gated
    # (FREETOKEN_NVFP4_PREFILL_WIDE=1), M >= 2048 only. With the env off the
    # wide module is never imported, so a broken/moved wide module cannot
    # affect production; if it is already loaded and its binding is installed
    # (runtime flip 1 -> 0), restore the pinned production binding.
    if os.environ.get("FREETOKEN_NVFP4_PREFILL_WIDE", "0") == "1":
        from freetoken.moe.fused_nvfp4_wide import dispatch_prefill

        return dispatch_prefill(M, cfg, globals())
    wmod = sys.modules.get("freetoken.moe.fused_nvfp4_wide")
    if wmod is not None:
        wmod.restore_prefill(globals())
    return cfg


def _prefill_gemm(
    a: torch.Tensor,
    packed: torch.Tensor,
    scale: torch.Tensor,
    glob: torch.Tensor,
    c: torch.Tensor,
    topk_weights_flat: torch.Tensor,
    sorted_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    num_valid_tokens: int,
    kernel_top_k: int,
    mul_routed_weight: bool,
    cfg: Dict[str, Any],
) -> None:
    N = packed.shape[1]
    K = packed.shape[2] * 2
    EM = sorted_ids.shape[0]
    scale = e4m3_kernel_view(scale)
    grid = lambda META: (  # noqa: E731
        triton.cdiv(EM, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    _prefill_nvfp4_moe_kernel[grid](
        a, packed, scale, glob, c, topk_weights_flat, sorted_ids, expert_ids,
        num_tokens_post_padded,
        _e2m1_lut(a.device.index),
        N, K, EM, num_valid_tokens,
        a.stride(0), a.stride(1),
        packed.stride(0), packed.stride(1), packed.stride(2),
        scale.stride(0), scale.stride(1), scale.stride(2),
        glob.stride(0), glob.stride(1),
        c.stride(1), c.stride(2),
        topk_weights_flat.stride(0),
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=kernel_top_k,
        compute_type=_tl_dtype(c.dtype),
        **cfg,
    )


def fused_experts_nvfp4(
    hidden_states: torch.Tensor,
    gate_up_packed: torch.Tensor,
    gate_up_scale: torch.Tensor,
    gate_up_global: torch.Tensor,
    down_packed: torch.Tensor,
    down_scale: torch.Tensor,
    down_global: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_experts: int,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    act_alpha: float = 1.702,
    act_limit: float = 7.0,
) -> torch.Tensor:
    """Prefill inline-NVFP4 MoE. ``topk_ids`` index rows of the bank tensors in
    ``[0, num_experts)``: full-layer banks with position == expert id (the
    materialized ``[:E]`` slot view or the overlap double buffer), raw ids."""
    M, H = hidden_states.shape
    top_k = topk_ids.shape[1]
    two_i = gate_up_packed.shape[1]
    inter = two_i // 2
    dev, dt = hidden_states.device, hidden_states.dtype
    cfg = _prefill_config(M)

    sorted_ids, expert_ids, ntpp = moe_align_block_size(topk_ids, cfg["BLOCK_SIZE_M"], num_experts)
    tw = topk_weights.reshape(-1).contiguous()
    num_valid = topk_ids.numel()

    ic1 = torch.empty((M, top_k, two_i), device=dev, dtype=dt)
    _prefill_gemm(
        hidden_states, gate_up_packed, gate_up_scale, gate_up_global, ic1,
        tw, sorted_ids, expert_ids, ntpp, num_valid, top_k,
        apply_router_weight_on_input, cfg,
    )
    ic2 = torch.empty((M * top_k, inter), device=dev, dtype=dt)
    gated_act_and_mul(activation, ic1.view(-1, two_i), ic2, alpha=act_alpha, limit=act_limit)
    ic3 = torch.empty((M, top_k, H), device=dev, dtype=dt)
    _prefill_gemm(
        ic2, down_packed, down_scale, down_global, ic3,
        tw, sorted_ids, expert_ids, ntpp, num_valid, 1,
        not apply_router_weight_on_input, cfg,
    )
    out = torch.empty_like(hidden_states)
    moe_sum_reduce_triton(ic3, out)
    return out


__all__ = [
    "fused_experts_decode_nvfp4_marlin",
    "fused_experts_decode_nvfp4_serial",
    "fused_experts_nvfp4",
]
