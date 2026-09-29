"""fp8 e4m3 weight with block scales: 128x128 (DeepSeek-V3/V4) or 32x32 (DeepSeek-V4.1)."""

from __future__ import annotations

from typing import Any

import torch

from ..registry import LayerKind, register_method
from ..scheme import FP8_BLOCK as BLOCK, FP8_BLOCK32, QuantKind
from .base import LinearConfig, LinearKernel, LinearMethod

FP8 = torch.float8_e4m3fn
E8M0 = torch.float8_e8m0fnu


def _e8m0(cfg: LinearConfig) -> bool:
    return cfg.scheme is not None and cfg.scheme.weight.scale == "e8m0"


def block_of(cfg: LinearConfig) -> int:
    """The checkpoint's block size; the scheme is the only place it is recorded."""
    group = cfg.scheme.weight.group if cfg.scheme is not None else None
    return group[0] if group else BLOCK


class Dsv4Fp8BlockLinearKernel(LinearKernel):
    """DeepSeek-V4's reference path: activations quantized to fp8 with power-of-two block scales, e8m0 weight scales read as codes."""

    name = "dsv4"

    def unusable_reason(self, cfg: LinearConfig) -> str | None:
        if not _e8m0(cfg):
            return "serves e8m0 block scales only"
        return None if block_of(cfg) == BLOCK else f"serves {BLOCK}x{BLOCK} blocks only"

    def apply(self, layer: Any, x: torch.Tensor) -> torch.Tensor:
        from freetoken.kernel.triton.dsv4.fp8_linear import block_fp8_linear

        return block_fp8_linear(x, layer.weight, layer.weight_scale_inv, layer.bias)


class TritonFp8BlockLinearKernel(LinearKernel):
    """W8A16 GEMV at M=1, dynamic 1x128 W8A8 GEMM above."""

    name = "triton"

    def unusable_reason(self, cfg: LinearConfig) -> str | None:
        if _e8m0(cfg):
            return "reads float block scales; e8m0 codes go to the dsv4 kernel"
        return None if block_of(cfg) == BLOCK else f"serves {BLOCK}x{BLOCK} blocks only"

    def apply(self, layer: Any, x: torch.Tensor) -> torch.Tensor:
        from freetoken.kernel.triton.fp8_block_linear import block_fp8_linear

        return block_fp8_linear(x, layer.weight, layer.weight_scale_inv, layer.bias)


class TritonFp8Block32LinearKernel(LinearKernel):
    """DeepSeek-V4.1's reference path: 32x32 e8m0 weight blocks with per-32 activation scales."""

    name = "triton32"

    def unusable_reason(self, cfg: LinearConfig) -> str | None:
        if not _e8m0(cfg):
            return "serves e8m0 block scales only"
        return None if block_of(cfg) == FP8_BLOCK32 else f"serves {FP8_BLOCK32}x{FP8_BLOCK32} blocks only"

    def apply(self, layer: Any, x: torch.Tensor) -> torch.Tensor:
        from freetoken.kernel.triton.fp8_block32 import block_fp8_linear

        return block_fp8_linear(x, layer.weight, layer.weight_scale_inv, layer.bias)


@register_method(QuantKind.FP8_BLOCK, LayerKind.LINEAR)
class Fp8BlockLinearMethod(LinearMethod):
    candidates = (Dsv4Fp8BlockLinearKernel, TritonFp8BlockLinearKernel, TritonFp8Block32LinearKernel)

    def create_weights(self, layer: Any) -> None:
        g = self.cfg
        block = block_of(g)
        if g.in_features % block or any(o % block for o in g.output_sizes):
            raise ValueError(f"block-fp8 needs in/out sizes divisible by {block}, got K={g.in_features} N={g.output_sizes}")
        layer.weight = torch.empty(g.out_features, g.in_features, dtype=FP8)
        # e8m0 codes stay codes for the dsv4 kernel; float scales are bf16 as the readers push them today
        scale_dtype = E8M0 if _e8m0(g) else torch.bfloat16
        layer.weight_scale_inv = torch.empty(g.out_features // block, g.in_features // block, dtype=scale_dtype)
