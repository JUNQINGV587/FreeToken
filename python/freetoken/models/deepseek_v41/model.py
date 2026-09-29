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

from freetoken.core import get_global_ctx
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
from .indexer import SharedAttentionRuntime
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
        runtime: SharedAttentionRuntime | None = None,
    ):
        self.layer_id = layer_id
        self.norm_eps = args.norm_eps
        self.dim = args.dim
        self.attn = Attention(
            layer_id, args, quant_config=quant_config, prefix=f"{prefix}.attn", runtime=runtime
        )
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
        attn_fn=None,
    ):
        """[B, S, hc_mult, dim] -> ([B, S, hc_mult, dim], ffn_pre [B, S, hc_mult]).

        ``attn_fn`` overrides the attention call alone: the paged path needs a different driver
        but the same engram / HC / FFN plumbing, and ``None`` is the eager reference call.
        """
        if pre_mix is None:
            pre_mix = make_identity_pre_mix(x, self.hc_mult)
        bsz, seqlen = x.size(0), x.size(1)

        residual = x
        attn_pre, attn_post, attn_comb = self.hc_mixes(
            x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
        )
        x = self.hc_pre(x, pre_mix)
        x = self.attn_norm.forward(x)
        x = self.attn.forward(x, start_pos, trace=trace) if attn_fn is None else attn_fn(x)
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
        # ONE band runtime for the whole stack, like the reference's module-global ``shared_attn``:
        # a kv-source layer publishes its compressed pool and its index list there, and every
        # consumer in its band (layers 3-7 <- 2, 9-13 <- 8, 15-19 <- 14, 21-39 <- 20) reads it back.
        # Per-layer instances would leave every consumer without a source's topk list.
        self._runtime = SharedAttentionRuntime()
        self.layers = OPList([
            Block(
                i, args, strategy=strategy, decode_target=decode_target,
                quant_config=quant_config, prefix=f"{prefix}.layers.{i}",
                layout=self.engram_layout, table=engram_table, runtime=self._runtime,
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

    def bind(self, device: torch.device, pool=None) -> None:
        """``pool is None`` binds the eager path (the oracle-comparison path); a pool binds paged.

        The order is deliberate: ``attn.bind(device)`` is what the eager tests and the reference
        comparison call, and it must keep meaning the same thing.
        """
        for layer in self.layers.op_list:
            layer.attn.bind(device, pool)
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

    @torch.no_grad()
    def forward_paged(
        self,
        input_ids: torch.Tensor,
        *,
        segments: list | None = None,
        flat_positions: torch.Tensor | None = None,
        pos: torch.Tensor | None = None,
        rows: torch.Tensor | None = None,
        engram_rows: torch.Tensor | None = None,
        cmp_stage_cap: int = 0,
        image_mask: torch.Tensor | None = None,
        full_logits: bool = False,
    ) -> torch.Tensor:
        """Drive the blocks along the paged CSA2 path; bind the model with a pool first.

        ``segments`` -- one ``(offset, n, table_idx, start_pos)`` per request, tiling the token
        axis -- is a ragged prefill; ``pos``/``rows`` is a decode step, one token per row, and
        ``cmp_stage_cap`` is the compressed width to stage. ``engram_rows`` is the requests'
        stable table rows for the n-gram history (defaults to ``rows``, the local ones).
        Everything else (engram, HC, FFN) is
        the eager plumbing, so a paged pass and an eager pass over the same tokens are comparable.
        """
        input_ids = input_ids.contiguous()
        bsz, seqlen = input_ids.size()
        assert (segments is None) != (pos is None), "pass either segments (prefill) or pos (decode)"
        assert all(layer.attn._paged for layer in self.layers.op_list), (
            "forward_paged needs the model bound to a pool: Transformer.bind(device, pool)"
        )
        if segments is not None:
            start_pos = segments[0][3]
            assert all(s[3] == start_pos for s in segments) or len(segments) == 1, (
                "a ragged batch carries one start_pos per request only when it is a warm pass"
            )
            assert start_pos == 0 or len(segments) == 1, (
                "a warm segment advances one request at a time"
            )
            if flat_positions is None:
                flat_positions = torch.cat([
                    s[3] + torch.arange(s[1], device=input_ids.device) for s in segments
                ])
        else:
            assert bsz == int(pos.numel()), "one pos per request row"
            if torch.cuda.is_current_stream_capturing():
                # A captured graph may not host-sync, and ``start_pos`` never reaches the paged
                # decode: the blocks only pass it to the EAGER attention call (``attn_fn`` is the
                # paged entry here), while the per-row positions travel in ``pos`` itself. So the
                # static 0 is the capture-safe value, not a guess at a position.
                start_pos = 0
            else:
                start_pos = int(pos.reshape(-1)[0])
        engram_mask = None if image_mask is None else ~image_mask
        hashes = None
        if self._engram_hash is not None:
            if segments is not None:
                hashes = self._engram_hash.forward_segments(input_ids, segments, engram_mask)
            else:
                # One hash per request ROW, at that row's own position. ``rows`` is the
                # attention-local row (snapshot/graph order) and a decode batch rebuilds it every
                # step, so the n-gram history is keyed on the request's TABLE row instead.
                erows = engram_rows if engram_rows is not None else rows
                hashes = self._engram_hash.forward(
                    input_ids, None, engram_mask, rows=erows, positions=pos.view(-1, 1)
                )
        h = self.embed.forward(input_ids.reshape(-1)).view(bsz, seqlen, self.args.dim)
        h = h.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)

        pre_mix = make_identity_pre_mix(h, self.hc_mult)
        layer = None
        for i, layer in enumerate(self.layers.op_list):
            if layer.engram is not None:
                idx = layer.engram.layer_hash_index
                h = layer.engram.forward(h, hashes[:, :, idx, :], engram_mask)
            if segments is not None:
                fn = lambda x, _l=layer: _l.attn.prefill_ragged(  # noqa: E731
                    x, segments, flat_positions
                )
            else:
                fn = lambda x, _l=layer: _l.attn.decode_step(  # noqa: E731
                    x, pos, rows, cmp_stage_cap
                )
            h, pre_mix = layer.forward(h, start_pos, pre_mix, image_mask, attn_fn=fn)
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
        self._bound = False
        self.model = Transformer(
            self._args,
            config.quant,
            strategy=config.moe_strategy,
            decode_target=config.decode_target,
            prefix="model",
            tokenizer=resolve_engram_tokenizer(config),
            engram_table=getattr(config, "engram_table", None),
        )

    def _ensure_bound(self) -> None:
        """Bind on the first forward, against whatever the global context holds.

        v41 stays runnable with NO context pool (the eval / oracle tests drive ``Transformer``
        directly), so a missing ctx or kv_cache is not an error -- it just means the eager path.
        """
        if self._bound:
            return
        pool = None
        try:
            pool = get_global_ctx().kv_cache
        except (AssertionError, AttributeError):
            pass
        device = pool.device if pool is not None else getattr(self._config, "device", None)
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.bind(torch.device(device), pool)
        self._bound = True
        # The 94 GiB engram tables are never resident weights, so a boot that did not pass a
        # table (tests/eval may) serves them off the checkpoint shards through the tier. Lazy:
        # only when the layers actually lack a table, and only when the checkpoint dir is known --
        # otherwise the layer raises its own, louder error when it is reached.
        if self._engram_tier is None:
            layers = self.engram_layers()
            model_dir = getattr(self._config, "checkpoint_path", None)
            if layers and model_dir and any(b.engram.table is None for b in layers):
                self.bind_engram_tier(device=torch.device(device), model_dir=model_dir)

    def mark_for_rebind(self) -> None:
        """Force a re-bind on the next forward. The model holds NO pool reference -- buffers are
        read off ctx.kv_cache via @property -- so a runtime rebuild needs no unbind; the old pool
        frees when the engine drops ctx.kv_cache. But the per-bind scratch (the compressor's rope
        table, the runtime's fixed-shape publish buffers) depends on the new pool's geometry, so
        re-derive it via _ensure_bound."""
        self._bound = False

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
        """One engine step: ragged prefill or a batched decode, logits per row.

        The batch is the scheduler's: ``input_ids`` [T] concatenated for a prefill (one segment per
        request, each starting at its own ``cached_len``) and [padded_size, 1] for a decode. Every
        stateful piece (window ring, compressor carry, indexer key cache, the published picks) is
        addressed through the attention metadata, so requests never read each other's KV.
        """
        self._ensure_bound()
        batch = get_global_ctx().batch
        input_ids = batch.input_ids.long()
        md = batch.attn_metadata
        device = input_ids.device
        if batch.is_prefill:
            segments = md.segments
            # full_logits so the head can pick each REQUEST's final token: a ragged prefill packs
            # several requests into the token axis and only their last row carries a next token.
            logits = self.model.forward_paged(
                input_ids.view(1, -1),
                segments=segments,
                flat_positions=batch.positions.long(),
                full_logits=True,
            )
            last = torch.tensor(
                [off + n - 1 for off, n, _ti, _sp in segments], dtype=torch.long, device=device
            )
            return logits[0].index_select(0, last)
        # DECODE (bs >= 1): per-row position (GPU int tensor -> no host syncs / graph safe). The
        # compressed staging cap is the max position any row reaches (eager); a static max_seq-1
        # under graph capture, so the captured static-shape graph serves any replay position.
        B = batch.padded_size
        pos = batch.positions.long().view(-1)[:B]
        if torch.cuda.is_current_stream_capturing():
            cmp_stage_cap = md.stage_width - 1
        else:
            cmp_stage_cap = int(pos.max().item())
        return self.model.forward_paged(
            input_ids.view(B, 1),
            pos=pos,
            rows=torch.arange(B, device=device),
            engram_rows=md.table_rows,
            cmp_stage_cap=cmp_stage_cap,
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
