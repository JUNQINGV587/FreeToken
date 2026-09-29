"""DeepSeek-V4.1-Flash model (engine port of ``inference/model.py``).

The V4.1 skeleton is the V4 one -- a 4-stream hyper-connection residual, MLA latents, sqrtsoftplus
routed MoE -- with three differences that this package exists to carry:

  * CSA2 attention: one sparse pass over [own 128-slot window | <=512 shared compressed slots],
    with the compressed pool published by kv-source layers 2/8/14/20 and shared across bands.
  * ModelOpt NVFP4 routed experts (per-16 E4M3 + fp32 global) instead of MXFP4.
  * Engram layers 1/14: an NVMe-backed 94 GiB-per-layer embedding lookup.

Milestones: M1 is the eager layer-0 path (window attention + hyper-connections) validated against
``inference/`` numerically; the MoE, the compressor/indexer, the paged backend and the EngramTier
follow. ``DeepseekV41ForCausalLM`` stays closed until the runtime is wired (M2).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from freetoken.kernel.triton.dsv4.hc import hc_post_combine, hc_pre_combine
from freetoken.kernel.triton.dsv4.sinkhorn import hc_split_sinkhorn
from freetoken.layers import BaseOP, RMSNorm
from freetoken.models.blocks import BaseLLMModel

from .args import DeepseekV41Args
from .attention import Attention
from .moe import MoE
from .moe import MoE


def make_identity_pre_mix(x: torch.Tensor, hc_mult: int) -> torch.Tensor:
    """Root pre-mix: the first layer's attention reads residual copy 0 (reference default)."""
    pre_mix = torch.zeros(*x.size()[:2], hc_mult, dtype=torch.float32, device=x.device)
    pre_mix[:, :, 0] = 1.0
    return pre_mix


class Block(BaseOP):
    """V4.1 decoder block: hyper-connection mixing, CSA2 attention, (next) the MoE FFN.

    The residual stream is ``hc_mult`` parallel copies. Unlike V4, the pre-mix feeding a block's
    ATTENTION is the PREVIOUS block's FFN mix, so ``forward`` threads ``pre_mix`` through; the
    block's own attention mix (``attn_pre``) is what feeds its FFN. That deferral is what keeps the
    previous layer's MoE finalize off the critical path.
    """

    def __init__(
        self,
        layer_id: int,
        args: DeepseekV41Args,
        *,
        strategy: str = "offload",
        decode_target: str = "gpu",
        quant_config=None,
        prefix: str = "",
    ):
        self.layer_id = layer_id
        self.norm_eps = args.norm_eps
        self.dim = args.dim
        self.attn = Attention(layer_id, args, quant_config=quant_config, prefix=f"{prefix}.attn")
        self.ffn = MoE(
            layer_id,
            args,
            strategy=strategy,
            decode_target=decode_target,
            quant_config=quant_config,
            prefix=f"{prefix}.ffn",
        )
        self.attn_norm = RMSNorm(args.dim, self.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, self.norm_eps)
        self.hc_mult = hc_mult = args.hc_mult
        self.hc_sinkhorn_iters = args.hc_sinkhorn_iters
        self.hc_eps = args.hc_eps
        mix_hc = (2 + hc_mult) * hc_mult
        hc_dim = hc_mult * args.dim
        # fp32 mixing weights (the reference keeps them fp32: ~15.7 GB across 40 layers).
        self.hc_attn_fn = torch.empty(mix_hc, hc_dim, dtype=torch.float32)
        self.hc_ffn_fn = torch.empty(mix_hc, hc_dim, dtype=torch.float32)
        self.hc_attn_base = torch.empty(mix_hc, dtype=torch.float32)
        self.hc_ffn_base = torch.empty(mix_hc, dtype=torch.float32)
        self.hc_attn_scale = torch.empty(3, dtype=torch.float32)
        self.hc_ffn_scale = torch.empty(3, dtype=torch.float32)

    def hc_mixes(self, x, hc_fn, hc_scale, hc_base):
        """``(pre, post, comb)`` for one mixing weight set, shaped [M, hc_mult] / [M, hc, hc]."""
        shape = x.size()
        xf = x.flatten(2).float()
        rsqrt = torch.rsqrt(xf.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(xf, hc_fn) * rsqrt
        M = shape[0] * shape[1]
        pre, post, comb = hc_split_sinkhorn(
            mixes.view(-1, mixes.size(-1)), hc_scale, hc_base, self.hc_mult,
            self.hc_sinkhorn_iters, self.hc_eps,
        )
        return pre.view(M, self.hc_mult), post.view(M, self.hc_mult), comb.view(M, self.hc_mult, self.hc_mult)

    def hc_pre(self, x, pre_mix):
        shape = x.size()
        xf = x.flatten(2).float()
        M = shape[0] * shape[1]
        y = hc_pre_combine(xf.view(M, self.hc_mult, self.dim), pre_mix.reshape(M, self.hc_mult), x.dtype)
        return y.view(*shape[:2], self.dim)

    def hc_post(self, x, residual, post, comb):
        shape = residual.size()
        M = shape[0] * shape[1]
        y = hc_post_combine(
            x.reshape(M, self.dim), residual.reshape(M, self.hc_mult, self.dim), post, comb
        )
        return y.view(shape)

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        start_pos: int = 0,
        pre_mix: torch.Tensor | None = None,
        image_mask: torch.Tensor | None = None,
        *,
        trace: dict | None = None,
    ):
        """[B, S, hc_mult, dim] -> ([B, S, hc_mult, dim], ffn_pre [B, S, hc_mult])."""
        if pre_mix is None:
            pre_mix = make_identity_pre_mix(x, self.hc_mult)
        bsz, seqlen = x.size(0), x.size(1)

        residual = x
        attn_pre, attn_post, attn_comb = self.hc_mixes(
            x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
        )
        x = self.hc_pre(x, pre_mix)
        x = self.attn_norm.forward(x)
        x = self.attn.forward(x, start_pos, trace=trace)
        x = self.hc_post(x, residual, attn_post, attn_comb)

        residual = x
        ffn_pre, ffn_post, ffn_comb = self.hc_mixes(
            x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base
        )
        ffn_pre_bs = ffn_pre.view(bsz, seqlen, self.hc_mult)
        if trace is not None:
            trace["ffn_pre"] = ffn_pre_bs
        x = self.hc_pre(x, attn_pre.view(bsz, seqlen, self.hc_mult))
        x = self.ffn_norm.forward(x)
        x = self.ffn.forward(x, image_mask)
        x = self.hc_post(x, residual, ffn_post, ffn_comb)
        return x, ffn_pre_bs


class DeepseekV41ForCausalLM(BaseLLMModel):
    """Engine entry point (registry key ``DeepseekV41ForCausalLM``).

    Weight loading and config parsing are wired (M0); the runtime -- paged KV, the shared
    compressed pool, the NVFP4 expert cache and the EngramTier -- lands with M2/M3/M5.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "DeepseekV41ForCausalLM: config parsing, the weight reader and the eager Block are in "
            "place (M0/M1); the engine runtime (paged CSA2 + NVFP4 experts) lands with M2."
        )


__all__ = ["Block", "DeepseekV41ForCausalLM", "make_identity_pre_mix"]
