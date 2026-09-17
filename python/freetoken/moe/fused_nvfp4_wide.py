"""Host orchestration for the wide-load NVFP4 prefill MoE path (spec t2 prototype).

Bit-identical drop-in for the prefill path of :mod:`freetoken.moe.fused_nvfp4`:
same call signatures, same ``moe_align_block_size`` grouping, same downstream
``gated_act_and_mul`` / ``moe_sum_reduce_triton``; only the grouped-GEMM kernel
is swapped for :func:`_prefill_nvfp4_moe_wide_kernel` (int32 wide loads + ALU
dequant, KB=32 lo->hi double-dot order preserved).

Dispatch (spec section 5): selected from the single return point of
``fused_nvfp4._prefill_config`` via :func:`dispatch_prefill`, gated on
``FREETOKEN_NVFP4_PREFILL_WIDE=1`` and ``M >= 2048``. The env is unset/"0" by
default, which leaves the production kernel and config table untouched;
rollback = flip the env + restart.
"""

from __future__ import annotations

import os
from typing import Any, Dict

import torch
import triton

from freetoken.kernel import moe_sum_reduce_triton
from freetoken.kernel.triton.e4m3_compat import e4m3_kernel_view
from freetoken.kernel.triton.nvfp4_fused_moe_wide import _prefill_nvfp4_moe_wide_kernel
from freetoken.layers import gated_act_and_mul
from freetoken.moe import fused_nvfp4 as _fnv4
from freetoken.moe.fused import moe_align_block_size

_tl_dtype = _fnv4._tl_dtype

WIDE_ENV = "FREETOKEN_NVFP4_PREFILL_WIDE"
# Small-M tiers are not this round's target (spec section 5): below this M the
# dispatch always keeps the production kernel.
WIDE_MIN_M = 2048

from freetoken.utils.logger import init_logger

_logger = init_logger(__name__)
_dispatch_log_done = [False]  # log-once guard (observability only)


def wide_requested(M: int) -> bool:
    return M >= WIDE_MIN_M and os.environ.get(WIDE_ENV, "0") == "1"


def _prefill_config_wide(M: int) -> Dict[str, int]:
    """Tile table for the wide kernel. Same coupling rules as
    ``fused_nvfp4._prefill_config`` (BM is tied to ``moe_align_block_size``);
    ``BLOCK_SIZE_KB`` stays 32 in every row (bit contract C2). M>=2048 row from
    the L20 sweep at production geometry (probe_nvfp4_prefill_tiles.py with
    FREETOKEN_NVFP4_PREFILL_WIDE=1): with wide loads the expert re-read penalty
    shrinks, so BM=64 + 4 warps beats the production BM=128/8-warp tile."""
    if M <= 64:
        cfg = dict(BLOCK_SIZE_M=16, BLOCK_SIZE_N=64, BLOCK_SIZE_KB=32,
                   GROUP_SIZE_M=1, num_warps=8, num_stages=4)
    elif M < 2048:
        cfg = dict(BLOCK_SIZE_M=32, BLOCK_SIZE_N=64, BLOCK_SIZE_KB=32,
                   GROUP_SIZE_M=8, num_warps=8, num_stages=2)
    else:
        cfg = dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_KB=32,
                   GROUP_SIZE_M=8, num_warps=4, num_stages=2)
    return cfg


def _prefill_gemm_wide(
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
    """Wide-kernel twin of ``fused_nvfp4._prefill_gemm`` (identical call
    signature). ``packed`` is the uint8 ``[S, N, K//2]`` bank, reinterpreted as
    int32 ``[S, N, K//8]`` (contiguous, K%8==0 for NVFP4) like the decode path."""
    N = packed.shape[1]
    K = packed.shape[2] * 2
    EM = sorted_ids.shape[0]
    packed_i32 = packed.view(torch.int32)  # [S, N, K // 8]
    scale = e4m3_kernel_view(scale)
    grid = lambda META: (  # noqa: E731
        triton.cdiv(EM, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    _prefill_nvfp4_moe_wide_kernel[grid](
        a, packed_i32, scale, glob, c, topk_weights_flat, sorted_ids, expert_ids,
        num_tokens_post_padded,
        N, K, EM, num_valid_tokens,
        a.stride(0), a.stride(1),
        packed_i32.stride(0), packed_i32.stride(1), packed_i32.stride(2),
        scale.stride(0), scale.stride(1), scale.stride(2),
        glob.stride(0), glob.stride(1),
        c.stride(1), c.stride(2),
        topk_weights_flat.stride(0),
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=kernel_top_k,
        compute_type=_tl_dtype(c.dtype),
        **cfg,
    )


def fused_experts_nvfp4_wide(
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
    """Wide-load prefill inline-NVFP4 MoE. Mirrors
    ``fused_nvfp4.fused_experts_nvfp4`` with the wide grouped-GEMM kernel."""
    M, H = hidden_states.shape
    top_k = topk_ids.shape[1]
    two_i = gate_up_packed.shape[1]
    inter = two_i // 2
    dev, dt = hidden_states.device, hidden_states.dtype
    cfg = _prefill_config_wide(M)

    sorted_ids, expert_ids, ntpp = moe_align_block_size(topk_ids, cfg["BLOCK_SIZE_M"], num_experts)
    tw = topk_weights.reshape(-1).contiguous()
    num_valid = topk_ids.numel()

    ic1 = torch.empty((M, top_k, two_i), device=dev, dtype=dt)
    _prefill_gemm_wide(
        hidden_states, gate_up_packed, gate_up_scale, gate_up_global, ic1,
        tw, sorted_ids, expert_ids, ntpp, num_valid, top_k,
        apply_router_weight_on_input, cfg,
    )
    ic2 = torch.empty((M * top_k, inter), device=dev, dtype=dt)
    gated_act_and_mul(activation, ic1.view(-1, two_i), ic2, alpha=act_alpha, limit=act_limit)
    ic3 = torch.empty((M, top_k, H), device=dev, dtype=dt)
    _prefill_gemm_wide(
        ic2, down_packed, down_scale, down_global, ic3,
        tw, sorted_ids, expert_ids, ntpp, num_valid, 1,
        not apply_router_weight_on_input, cfg,
    )
    out = torch.empty_like(hidden_states)
    moe_sum_reduce_triton(ic3, out)
    return out


def _capture_orig_prefill_gemm():
    """Pin the production ``_prefill_gemm`` as a module constant AT IMPORT.

    Fail-closed: if the binding is already foreign at import (and not a pinned
    capture from a previous incarnation of this module, e.g. after
    ``importlib.reload``), refuse to load rather than risk capturing a wide
    binding as "original" and silently disabling the env=0 production fuse.
    """
    orig = getattr(_fnv4, "_WIDE_ORIG_PREFILL_GEMM", None)
    if orig is not None:
        return orig  # reload path: constant already pinned on the host module
    cur = _fnv4._prefill_gemm
    if getattr(cur, "__module__", None) != _fnv4.__name__ or getattr(cur, "__name__", None) != "_prefill_gemm":
        raise RuntimeError(
            "fused_nvfp4_wide: fused_nvfp4._prefill_gemm is already rebound at "
            f"import ({cur!r}); refusing to load (fail-closed)"
        )
    _fnv4._WIDE_ORIG_PREFILL_GEMM = cur
    return cur


_ORIG_PREFILL_GEMM = _capture_orig_prefill_gemm()


def restore_prefill(host_globals: Dict[str, Any]) -> None:
    """Rebind the pinned production ``_prefill_gemm`` if a wide binding is
    currently installed; no-op otherwise. No import, no mutable state.

    A stale wide incarnation (left installed across ``importlib.reload`` of
    this module) is recognized by module/name, not identity.
    """
    cur = host_globals.get("_prefill_gemm")
    if cur is None or cur is _ORIG_PREFILL_GEMM:
        return
    if cur is _prefill_gemm_wide or (
        getattr(cur, "__name__", None) == "_prefill_gemm_wide"
        and str(getattr(cur, "__module__", "")).endswith("fused_nvfp4_wide")
    ):
        host_globals["_prefill_gemm"] = _ORIG_PREFILL_GEMM


def dispatch_prefill(M: int, cfg: Dict[str, int], host_globals: Dict[str, Any]) -> Dict[str, int]:
    """Wide-path selector, called at the single return point of
    ``fused_nvfp4._prefill_config`` -- only reached when
    ``FREETOKEN_NVFP4_PREFILL_WIDE=1`` (the caller gates the import, so an
    env-off process never touches this module; it may only invoke
    :func:`restore_prefill` on an already-imported module).

    ``_prefill_gemm`` is resolved from the module globals at call time, so the
    wide path is installed by rebinding it. Stateless: both branches assign
    pinned constants, so reloads and concurrent first calls cannot poison the
    restore path.
    """
    if wide_requested(M):
        host_globals["_prefill_gemm"] = _prefill_gemm_wide
        if not _dispatch_log_done[0]:
            _dispatch_log_done[0] = True
            _logger.info_rank0(f"nvfp4 prefill: wide-load kernel active ({WIDE_ENV}=1, M={M})")
        # The caller feeds cfg["BLOCK_SIZE_M"] to moe_align_block_size, so the
        # wide tile table is returned as a whole to keep BM coupling intact.
        cfg = _prefill_config_wide(M)
        bm_override = int(os.environ.get("FREETOKEN_NVFP4_PREFILL_BM", "0"))
        if bm_override > 0 and M > 64:
            cfg["BLOCK_SIZE_M"] = bm_override
    else:
        restore_prefill(host_globals)
    return cfg


__all__ = [
    "dispatch_prefill",
    "fused_experts_nvfp4_wide",
    "restore_prefill",
    "wide_requested",
    "WIDE_ENV",
    "WIDE_MIN_M",
]
