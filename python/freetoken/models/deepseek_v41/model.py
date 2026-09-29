"""DeepSeek-V4.1-Flash model (engine port of ``inference/model.py``).

The V4.1 skeleton is the V4 one -- a 4-stream hyper-connection residual, MLA latents, sqrtsoftplus
routed MoE -- with three differences that this package exists to carry:

  * CSA2 attention: one sparse pass over [own 128-slot window | <=512 shared compressed slots],
    with the compressed pool published by kv-source layers 2/8/14/20 and shared across bands.
  * ModelOpt NVFP4 routed experts (per-16 E4M3 + fp32 global) instead of MXFP4.
  * Engram layers 1/14: an NVMe-backed 94 GiB-per-layer embedding lookup.

Milestones: M1 is the eager layer-0 path (window attention + hyper-connections) validated against
``inference/`` numerically; the MoE, the compressor/indexer and the EngramTier followed (M2-M3).
The paged backend and the engine batch path into ``DeepseekV41ForCausalLM.forward`` are M5.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F

from freetoken.kernel.triton.dsv4.hc import hc_post_combine, hc_pre_combine
from freetoken.kernel.triton.dsv4.sinkhorn import hc_split_sinkhorn
from freetoken.layers import (
    BaseOP,
    OPList,
    ParallelLMHead,
    RMSNorm,
    VocabParallelEmbedding,
)
from freetoken.models.blocks import BaseLLMModel

from .args import DeepseekV41Args
from .attention import Attention
from .engram import Engram, EngramLayout, NgramHashState
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
        layout: EngramLayout | None = None,
        table=None,
    ):
        self.layer_id = layer_id
        self.norm_eps = args.norm_eps
        self.dim = args.dim
        self.attn = Attention(layer_id, args, quant_config=quant_config, prefix=f"{prefix}.attn")
        # The reference keeps the engram in the Transformer but the weights live at
        # ``layers.N.engram.*``, so the module is owned here and driven from the loop below.
        self.engram = (
            Engram(layer_id, args, layout, quant_config=quant_config,
                   prefix=f"{prefix}.engram", table=table)
            if layout is not None and args.is_engram_layer(layer_id)
            else None
        )
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


class Transformer(BaseOP):
    """Embed -> ``hc_mult`` copies -> the 40 blocks -> collapse -> fp32 logits.

    Eager (M2): this is the reference's own structure, so a short prompt can be compared against
    ``inference/`` position by position. The paged runtime -- global window/compressed pools,
    per-segment prefill, graph-safe decode -- replaces the attention plumbing in M5; nothing here
    reads the engine's batch context, and every per-sequence cache is dropped by :meth:`reset`.

    The collapse before the head is the LAST block's ``hc_pre`` fed with the last ``ffn_pre`` it
    returned: the reference reuses its loop variables, and this checkpoint ships no ``hc_head_*``
    tensors (verified in the header), so there is no learned head mix to apply.
    """

    def __init__(
        self,
        args: DeepseekV41Args,
        quant_config=None,
        *,
        strategy: str = "offload",
        decode_target: str = "gpu",
        prefix: str = "",
        tokenizer=None,
        engram_table=None,
    ):
        self.args = args
        self.norm_eps = args.norm_eps
        self.hc_mult = args.hc_mult
        self.engram_layout = EngramLayout.from_args(args)
        self.embed = VocabParallelEmbedding(args.vocab_size, args.dim)
        self.layers = OPList([
            Block(
                i, args, strategy=strategy, decode_target=decode_target,
                quant_config=quant_config, prefix=f"{prefix}.layers.{i}",
                layout=self.engram_layout, table=engram_table,
            )
            for i in range(args.n_layers)
        ])
        self.norm = RMSNorm(args.dim, args.norm_eps)
        self.head = ParallelLMHead(
            args.vocab_size, args.dim, quant_config=quant_config, prefix=f"{prefix}.head"
        )
        # Not a parameter set: the hash state is DERIVED from the tokenizer, so it must stay out of
        # the checkpoint's key set (hence the underscore, which state_dict skips).
        self._engram_table = engram_table
        self._engram_hash = None
        if self.engram_layout is not None and tokenizer is not None:
            self._engram_hash = NgramHashState(args, self.engram_layout, tokenizer)

    def bind(self, device: torch.device) -> None:
        for layer in self.layers.op_list:
            layer.attn.bind(device)
            if layer.engram is not None:
                layer.engram.bind(table=self._engram_table, device=device)
        if self._engram_hash is not None:
            self._engram_hash.bind(device)

    def reset(self) -> None:
        """Drop every per-sequence cache (window rings, compressor carry, indexer keys, n-gram
        history) so the same model can run another pass from position 0."""
        for layer in self.layers.op_list:
            layer.attn.reset()
        if self._engram_hash is not None:
            self._engram_hash.reset()

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        start_pos: int = 0,
        *,
        image_mask: torch.Tensor | None = None,
        full_logits: bool = False,
        trace: dict | None = None,
    ) -> torch.Tensor:
        """``input_ids`` [B, S] -> logits [B, vocab] (last position) or [B, S, vocab]."""
        # the embedding kernel takes contiguous ids; a caller slicing a batch ([2,1] out of [2,8])
        # hands us a strided view
        input_ids = input_ids.contiguous()
        bsz, seqlen = input_ids.size()
        assert start_pos == 0 or seqlen == 1, "a warm pass advances one token at a time"
        engram_mask = None if image_mask is None else ~image_mask
        hashes = None
        if self._engram_hash is not None:
            hashes = self._engram_hash.forward(input_ids, start_pos, engram_mask)
        h = self.embed.forward(input_ids.reshape(-1)).view(bsz, seqlen, self.args.dim)
        h = h.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)

        pre_mix = make_identity_pre_mix(h, self.hc_mult)
        layer = None
        for i, layer in enumerate(self.layers.op_list):
            if layer.engram is not None:
                idx = layer.engram.layer_hash_index
                h = layer.engram.forward(h, hashes[:, :, idx, :], engram_mask)
            h, pre_mix = layer.forward(h, start_pos, pre_mix, image_mask, trace=trace)
        assert layer is not None, "a model with no layers has no state to collapse"
        h = layer.hc_pre(h, pre_mix)
        h = self.norm.forward(h)
        if not full_logits:
            h = h[:, -1:]
        logits = self.logits(h.reshape(-1, self.args.dim))
        return logits.view(bsz, seqlen, -1) if full_logits else logits

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        """The head over rows the caller has ALREADY gathered.

        ``ParallelLMHead.forward`` is written for the engine's batch path: it takes the request
        count and the per-request last-token indices off the attention metadata and all-gathers
        across TP. None of that exists here (M5 wires the batch path), and this checkpoint ties no
        embedding, so the head is exactly its quantized linear.
        """
        assert self.head.tied_embedding is None, "this checkpoint does not tie the head"
        return self.head.quant_method.apply(self.head, h)


class DeepseekV41ForCausalLM(BaseLLMModel):
    """Engine entry point (registry key ``DeepseekV41ForCausalLM``).

    Weight loading, config parsing, the eager transformer and the NVMe-backed engram tables are
    wired (M0-M3); the paged CSA2 runtime and the NVFP4 expert cache land with M5.
    """

    def __init__(self, config):
        self._config = config
        self._args: DeepseekV41Args = config.dsv41_args
        self._engram_tier = None
        self.model = Transformer(
            self._args,
            config.quant,
            strategy=config.moe_strategy,
            decode_target=config.decode_target,
            prefix="model",
            tokenizer=resolve_engram_tokenizer(config),
            engram_table=getattr(config, "engram_table", None),
        )

    def engram_layers(self) -> list:
        """The blocks that carry an engram lookup (layers 1 and 14 in this checkpoint)."""
        return [block for block in self.model.layers.op_list if block.engram is not None]

    def bind_engram_tier(self, device=None, *, model_dir: str | None = None, use_io_uring=None):
        """Serve the engram tables off NVMe and hand them to every engram layer.

        The two tables are 94 GiB each, so they are never a resident tensor: :class:`EngramTier`
        reads the ~24 rows one decode step actually touches straight out of the checkpoint shards
        (``gather`` per layer, synchronously -- see :mod:`engram_tier`). Returns the tier, or
        ``None`` for a config with no engram layers. Call it once the device is known; ``bind``
        hands the table on to each layer.
        """
        layers = self.engram_layers()
        if not layers:
            return None
        model_dir = model_dir or getattr(self._config, "checkpoint_path", None)
        if not model_dir:
            raise RuntimeError(
                "DeepseekV41ForCausalLM.bind_engram_tier needs the checkpoint directory "
                "(ModelConfig.checkpoint_path)"
            )
        from .engram_tier import EngramTier

        tier = EngramTier(
            model_dir,
            [block.engram.layer_id for block in layers],
            device=device,
            use_io_uring=use_io_uring,
        )
        for block in layers:
            view = tier.view(tier.layer_index(block.engram.layer_id))
            block.engram.bind(table=view, device=device)
        self._engram_tier = tier
        return tier

    def forward(self) -> torch.Tensor:
        raise NotImplementedError(
            "DeepseekV41ForCausalLM: the engine batch path (paged CSA2 + the NVFP4 expert cache) "
            "lands with M5 -- call Transformer.forward directly until then."
        )


def resolve_engram_tokenizer(config):
    """The tokenizer the n-gram hash's token map is built from.

    ``NgramHashState`` needs the authors' own tokenizer: the compressed map is a per-token
    normalization of the vocabulary, so an equivalent-but-not-identical tokenizer silently
    renumbers every token (see ``engram.build_compressed_token_map``). It comes from
    ``ModelConfig.engram_tokenizer_path`` when a caller pins one -- either the ``tokenizer.json``
    itself or the directory holding it -- else from ``tokenizer.json`` next to the checkpoint.
    """
    path = getattr(config, "engram_tokenizer_path", None)
    if path and os.path.isfile(path):
        from tokenizers import Tokenizer

        return Tokenizer.from_file(path)
    directory = path or getattr(config, "checkpoint_path", None)
    if directory and os.path.isfile(os.path.join(directory, "tokenizer.json")):
        from .engram import tokenizer_of

        return tokenizer_of(directory)
    if getattr(getattr(config, "dsv41_args", None), "engram_layer_ids", ()):
        raise RuntimeError(
            "DeepSeek-V4.1 has engram layers, whose n-gram hash needs the checkpoint's "
            "tokenizer.json (set ModelConfig.checkpoint_path or engram_tokenizer_path)"
        )
    return None


__all__ = [
    "Block",
    "Transformer",
    "DeepseekV41ForCausalLM",
    "make_identity_pre_mix",
    "resolve_engram_tokenizer",
]
