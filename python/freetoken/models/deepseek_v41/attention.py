"""DeepSeek-V4.1 (CSA2) attention -- the eager layer-0 path.

V4.1 attention is ONE sparse pass per layer over

    [ this layer's own 128-slot sliding window | up to ``index_topk`` (512) selected compressed slots ]

The compressed pool is *published* by a kv-source layer (2/8/14/20) and *consumed* by every layer in
its band (3-7 <- 2, 9-13 <- 8, 15-19 <- 14, 21-39 <- 20), which is why the engine keeps its
compressed KV in the shared paged pool rather than per layer.

Ratio 0 (layers 0, 1 and the MTP layers) has no compressor and no indexer: attention is the window
half alone. That is what the layer-0 oracle dump (``block0_ref.pt``) pins down numerically. Ratios 2
and 1 add the compressor (a kv-source layer publishes its latent into the band's shared pool) and
the indexer (which compressed positions a query may see), and ``attn2_ref.pt`` pins those down per
stage.

The caches here are the reference's own eager buffers, kept per layer and indexed by absolute
position, which is what makes a step-by-step comparison against the reference possible at all. The
paged backend (pool slots, shared runtime across the band, CUDA-graph-safe state) replaces them in
M5 -- and the *order* of the two methods below is load-bearing in the meantime: the indexer reads
the compressor latent before it is rotated or quantized, so ``_compress_kv`` must run it first.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from freetoken.core import get_global_ctx
from freetoken.kernel.triton.dsv4.fp8_linear import act_quant_fp8_inplace
from freetoken.kernel.triton.fp4_e4m3_act import fp4_act_quant_e4m3_inplace
from freetoken.layers import (
    BaseOP,
    LinearColParallelMerged,
    LinearReplicated,
    LinearRowParallel,
    RMSNorm,
)

from ..deepseek_v4.ops import apply_rotary_emb_decode
from .args import DeepseekV41Args
from .compress import Compressor
from .indexer import Indexer, SharedAttentionRuntime
from .ops import apply_rotary_emb, get_freqs_cis, get_window_topk_idxs, sparse_attn_torch


class Attention(BaseOP):
    """CSA2 attention over the layer's own window and the band's shared compressed pool."""

    def __init__(
        self,
        layer_id: int,
        args: DeepseekV41Args,
        *,
        quant_config=None,
        prefix: str = "",
        runtime: SharedAttentionRuntime | None = None,
    ):
        self.layer_id = layer_id
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.q_lora_rank = args.q_lora_rank
        self.o_lora_rank = args.o_lora_rank
        self.head_dim = args.head_dim
        self.rope_head_dim = args.rope_head_dim
        self.n_groups = args.o_groups
        self.window_size = args.window_size
        self.eps = args.norm_eps
        self.max_batch_size = args.max_batch_size
        self.max_seq_len = args.max_seq_len
        self.compress_ratio = args.layer_ratio(layer_id)
        self.is_kv_source = args.is_kv_source(layer_id)
        self.is_index_source = args.is_index_source(layer_id)
        self.kv_source_layer = args.kv_source_for(layer_id)
        # The band shares one compressed pool: the kv-source layer writes it, every layer in the
        # band (the source included) reads it. Underscore-prefixed: it is state, not a parameter.
        self._runtime = runtime if runtime is not None else SharedAttentionRuntime()

        self.attn_sink = torch.empty(self.n_heads, dtype=torch.float32)
        # The latent projections are replicated; wq_b shards over heads, wo_b over the output groups.
        self.wq_a = LinearReplicated(
            self.dim, self.q_lora_rank, has_bias=False, quant_config=quant_config,
            prefix=f"{prefix}.wq_a",
        )
        self.q_norm = RMSNorm(self.q_lora_rank, self.eps)
        self.wq_b = LinearColParallelMerged(
            self.q_lora_rank, [self.n_heads * self.head_dim], has_bias=False,
            quant_config=quant_config, prefix=f"{prefix}.wq_b",
        )
        self.wkv = LinearReplicated(
            self.dim, self.head_dim, has_bias=False, quant_config=quant_config,
            prefix=f"{prefix}.wkv",
        )
        self.kv_norm = RMSNorm(self.head_dim, self.eps)
        # wo_a is one [o_lora_rank, K] matrix per output group, stacked on N and applied as an
        # einsum; the reference dequantizes it to bf16 and so does the reader.
        wo_a_rows = self.n_groups * args.o_lora_rank
        wo_a_k = self.n_heads * self.head_dim // self.n_groups
        self.wo_a = torch.empty(wo_a_rows, wo_a_k, dtype=torch.bfloat16)
        self.wo_b = LinearRowParallel(
            self.n_groups * args.o_lora_rank, self.dim, has_bias=False, quant_config=quant_config,
            prefix=f"{prefix}.wo_b",
        )
        self.softmax_scale = self.head_dim ** -0.5

        self.compressor = None
        self.indexer = None
        if self.compress_ratio:
            # A kv-source layer pools its own tokens (ratio 2 gates pairs, ratio 1 projects) and
            # publishes the latent; an index-source layer selects which of those positions each
            # query may see. Layers 2/8/14/20 are both, 24/28/32/36 select only.
            if self.is_kv_source:
                self.compressor = Compressor(
                    layer_id, args, quant_config=quant_config, prefix=f"{prefix}.compressor"
                )
            if self.is_index_source:
                self.indexer = Indexer(
                    layer_id, args, quant_config=quant_config, prefix=f"{prefix}.indexer",
                    runtime=self._runtime,
                )
        self._window_cache: torch.Tensor | None = None
        self._compress_cache: torch.Tensor | None = None
        # Paged addressing (M5). The pool itself is never stored -- ``attn`` and ``pool`` are read
        # off the live global context -- so a runtime rebuild needs no unbind.
        self._paged = False
        self.band_ratio = self.compress_ratio
        self._has_compression = False
        self.index_topk = args.index_topk

        # Ratio-0 layers turn YaRN off (the window never exceeds the trained length); compressed
        # layers rope their latents with ``compress_rope_theta`` over the 16x stretched context.
        if self.compress_ratio:
            original_seq_len, rope_theta = args.original_seq_len, args.compress_rope_theta
        else:
            original_seq_len, rope_theta = 0, args.rope_theta
        self._freqs_params = (
            self.rope_head_dim, args.max_seq_len, original_seq_len,
            rope_theta, args.rope_factor, args.beta_fast, args.beta_slow,
        )
        # Bound on first forward. Underscore-prefixed so it stays out of state_dict.
        self._freqs_cis: torch.Tensor | None = None

    def bind(self, device: torch.device, pool=None) -> None:
        """Bind the rope table and pick the path: ``pool is None`` keeps the eager per-layer
        caches (the oracle-comparison path), a pool switches to the paged backend.

        The pool is used for its geometry only, never stored: a rebuild is followed by
        ``mark_for_rebind`` on the model, which re-derives the fixed-shape scratch.
        """
        self._freqs_cis = get_freqs_cis(*self._freqs_params, device)
        self._paged = pool is not None
        if pool is None:
            # The reference allocates these as non-persistent buffers; here they are plain state, so
            # they must stay out of ``state_dict`` (underscore) and out of the checkpoint's way.
            self._window_cache = torch.zeros(
                self.max_batch_size, self.window_size, self.head_dim,
                dtype=torch.bfloat16, device=device,
            )
            if self.compressor is not None:
                self._compress_cache = torch.zeros(
                    self.max_batch_size, self.max_seq_len // self.compress_ratio, self.head_dim,
                    dtype=torch.bfloat16, device=device,
                )
            return
        self._window_cache = None
        self._compress_cache = None
        # The band's ratio, not this layer's: a consumer shares the pool its source addressed.
        self.band_ratio = pool.cmp_ratio_of(self.layer_id) or self.compress_ratio
        self._has_compression = pool.cmp_source_of(self.layer_id) is not None
        if self.compressor is not None:
            self.compressor.bind_paged(self.layer_id, self._freqs_cis, tier="attn")
        if self.indexer is not None:
            self.indexer.bind_paged(self.layer_id, self._freqs_cis, device)

    def reset(self) -> None:
        """Drop the per-sequence state: window/compressed rings, the compressor carry and the
        indexer's key cache. The rope table and the shared runtime slots are re-derived per step."""
        if self._paged:
            # the pool owns the paged state and the scheduler recycles it per request, so there is
            # nothing here to drop -- and the runtime's buffers are fixed-shape, not per-sequence
            return
        if self._window_cache is not None:
            self._window_cache.zero_()
        if self._compress_cache is not None:
            self._compress_cache.zero_()
        if self.compressor is not None:
            self.compressor.reset()
        if self.indexer is not None:
            self.indexer.reset()
        self._runtime.reset()

    def _window_kv(
        self, x: torch.Tensor, freqs: torch.Tensor, start_pos: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project, rope and *quantize round-trip* this layer's window KV, and write the ring cache.

        The reference quantizes the FULL head vector (rope tail included) with
        ``act_quant(kv, 32, "ue8m0", e8m0, inplace=True)`` -- unlike V4, which quantizes only the
        non-rope part at block 64. Any deviation here shows up immediately in the ``kv`` probe.

        A cold prefill returns the whole (unclipped) KV and indexes it directly; the cache only
        exists for the decode steps that follow, where the window is the ring buffer in slot order.
        """
        bsz, seqlen, _ = x.shape
        win = self.window_size
        kv = self.kv_norm.forward(self.wkv.forward(x))
        apply_rotary_emb(kv[..., -self.rope_head_dim:], freqs)
        act_quant_fp8_inplace(kv, 32)
        if start_pos == 0:
            if seqlen <= win:
                self._window_cache[:bsz, :seqlen] = kv
            else:
                cutoff = seqlen % win
                tail = kv[:, -win:]
                self._window_cache[:bsz, cutoff:win], self._window_cache[:bsz, :cutoff] = (
                    tail.split([win - cutoff, cutoff], dim=1)
                )
            window_kv = kv
        else:
            self._window_cache[:bsz, start_pos % win] = kv.squeeze(1)
            window_kv = self._window_cache[:bsz]
        idxs = get_window_topk_idxs(win, bsz, seqlen, start_pos).to(kv.device)
        return window_kv, idxs

    def _compress_topk_idxs(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        latent: torch.Tensor | None,
        start_pos: int,
        offset: int,
        compress_len: int,
    ) -> torch.Tensor:
        """The compressed half of the index list (a band inherits its source's list verbatim)."""
        if self.indexer is None:
            return self._runtime.topk_idxs
        bsz, seqlen, _ = x.shape
        if compress_len == 0:
            return torch.empty(bsz, seqlen, 0, dtype=torch.int32, device=x.device)
        idxs = self.indexer.forward(x, qr, latent, start_pos, offset)
        self._runtime.topk_idxs = idxs
        return idxs

    def _compress_kv(
        self, x: torch.Tensor, qr: torch.Tensor, start_pos: int, offset: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pool, select, rope, fp4-quantize and cache this layer's compressed KV.

        Order is the whole point: the indexer scores the *unrotated* latent, so it runs before the
        rope/fp4 round trip below overwrites it -- and ``offset`` is the window length, because the
        selected slots address the concatenated ``[window | compressed]`` KV the attention reads.
        """
        ratio = self.compress_ratio
        bsz, seqlen, _ = x.shape
        compress_len = (start_pos + seqlen) // ratio
        latent = None
        if self.compressor is not None:
            latent = self.compressor.forward(x, start_pos)
            self._runtime.compress_kv = self._compress_cache
        idxs = self._compress_topk_idxs(x, qr, latent, start_pos, offset, compress_len)
        if latent is not None:
            # A latent stands for the FIRST token of its group: group j is roped at position j*ratio.
            if start_pos == 0:
                freqs = self._freqs_cis[: seqlen - seqlen % ratio : ratio]
            else:
                freqs = self._freqs_cis[start_pos + 1 - ratio].unsqueeze(0)
            apply_rotary_emb(latent[..., -self.rope_head_dim:], freqs)
            fp4_act_quant_e4m3_inplace(latent, 16)
            slot = start_pos // ratio
            self._compress_cache[:bsz, slot:slot + latent.size(1)] = latent
        return self._runtime.compress_kv[:bsz, :compress_len], idxs

    def _wo(self, o: torch.Tensor, bsz: int, seqlen: int) -> torch.Tensor:
        o = o.reshape(bsz, seqlen, self.n_groups, -1)
        wo_a = self.wo_a.view(self.n_groups, self.o_lora_rank, -1)
        o = torch.einsum("bsgd,grd->bsgr", o, wo_a).flatten(2)
        return self.wo_b.forward(o)

    # ----- paged path (M5) ----------------------------------------------------------------
    @property
    def attn(self):
        return get_global_ctx().attn_backend

    @property
    def pool(self):
        return get_global_ctx().kv_cache

    def _segment_window(self, kv_seg: torch.Tensor, ti: int, start_pos: int, n: int):
        """Store this segment's window KV and return its window half as GLOBAL slot lists."""
        win = self.window_size
        device = kv_seg.device
        end = start_pos + n
        slots = self.attn.window_slots_of(ti, start_pos, end)
        self.attn.store_window(kv_seg[0], self.layer_id, slots)
        if start_pos == 0:
            cols = get_window_topk_idxs(win, 1, n, 0).to(device)
            return self.attn.win_cols_to_global(cols, slots)
        # an extend segment sees the whole ring: slot j holds the newest position p <= q with
        # p % win == j, i.e. p = q - ((q - j) % win)
        w_lo = max(0, start_pos - win + 1)
        ring = self.attn.window_slots_of(ti, w_lo, end)
        q = start_pos + torch.arange(n, device=device).unsqueeze(1)
        cols = (q - win + 1).clamp(min=w_lo) + torch.arange(win, device=device)
        cols = torch.where(cols > q, -1, cols - w_lo).unsqueeze(0)
        return self.attn.win_cols_to_global(cols, ring)

    def _segment_blocks(self, x_seg, qr_seg, ti: int, start_pos: int, off: int) -> torch.Tensor:
        """The compressed half of this segment's index list, as BLOCK indices.

        A layer without a compressor of its own reads the band source's published picks verbatim;
        its pool is the source's pool, so those blocks resolve to the same rows.
        """
        n = x_seg.size(1)
        assert self._runtime.topk_idxs is not None, (
            f"layer {self.layer_id} reads its band's picks but source "
            f"{self.kv_source_layer} has no indexer"
        )
        topk = min(self.index_topk, (start_pos + n) // self.band_ratio)
        return self._runtime.topk_idxs[: x_seg.size(0), off : off + n, :topk]

    def _segment(self, x_seg, qr_seg, kv_seg, ti: int, start_pos: int, off: int):
        """``(window half, compressed half)`` for one request's segment, both as global slots."""
        n = x_seg.size(1)
        win_global = self._segment_window(kv_seg, ti, start_pos, n)
        if not self._has_compression:
            return win_global, None
        latent = starts = None
        if self.compressor is not None:
            latent, starts = self.compressor.compress_paged(x_seg, start_pos, ti)
        if self.indexer is not None:
            blocks = self.indexer.forward_paged(
                x_seg, qr_seg, latent, starts, start_pos, ti, off
            )
        else:
            blocks = self._segment_blocks(x_seg, qr_seg, ti, start_pos, off)
        if latent is not None and starts is not None and starts.numel():
            # scatter only now: the indexer above had to score the latent UNROTATED
            self.compressor.store_paged(latent, starts, ti)
        return win_global, self.attn.blocks_to_global(blocks, self.band_ratio, ti=ti)

    @torch.no_grad()
    def prefill_ragged(
        self, x: torch.Tensor, segments: list, flat_positions: torch.Tensor
    ) -> torch.Tensor:
        """Paged ragged prefill: ``x`` is [1, T, dim] and ``segments`` carries one
        ``(offset, n, table_idx, start_pos)`` per request, tiling [0, T). Returns [1, T, dim]."""
        assert self._paged, "prefill_ragged is the paged path; bind the layer with a pool"
        total = x.size(1)
        if len(segments) == 1:
            freqs = self._freqs_cis[segments[0][3] : segments[0][3] + total]
        else:
            freqs = self._freqs_cis.index_select(0, flat_positions)
        rd = self.rope_head_dim

        qr = self.q_norm.forward(self.wq_a.forward(x))
        q = self.wq_b.forward(qr).unflatten(-1, (self.n_heads, self.head_dim))
        apply_rotary_emb(q[..., -rd:], freqs)

        kv = self.kv_norm.forward(self.wkv.forward(x))
        apply_rotary_emb(kv[..., -rd:], freqs)
        act_quant_fp8_inplace(kv, 32)

        win_parts, cmp_parts = [], []
        for off, n, ti, start_pos in segments:
            win = self._segment(
                x[:, off : off + n], qr[:, off : off + n], kv[:, off : off + n], ti, start_pos,
                off,
            )
            win_parts.append(win[0])
            if win[1] is not None:
                cmp_parts.append(win[1])
        # one kernel launch over all segments, so both halves share a width across the batch
        n_window = self.window_size if len(segments) > 1 else win_parts[0].shape[-1]
        if cmp_parts:
            max_c = max(part.shape[-1] for part in cmp_parts)
            flat = [
                torch.cat(
                    [
                        F.pad(w, (0, n_window - w.shape[-1]), value=-1),
                        F.pad(c, (0, max_c - c.shape[-1]), value=-1),
                    ],
                    dim=-1,
                )
                for w, c in zip(win_parts, cmp_parts)
            ]
        else:
            flat = [F.pad(w, (0, n_window - w.shape[-1]), value=-1) for w in win_parts]
        topk_idxs = (flat[0] if len(flat) == 1 else torch.cat(flat, dim=1)).int()
        o = self.attn.attend(
            q, self.layer_id, topk_idxs, n_window, self.attn_sink, self.softmax_scale,
            has_compression=bool(cmp_parts),
        )
        apply_rotary_emb(o[..., -rd:], freqs, True)
        return self._wo(o, 1, total)

    @torch.no_grad()
    def decode_step(
        self, x: torch.Tensor, pos: torch.Tensor, rows: torch.Tensor, cmp_stage_cap: int,
        wctx=None,
    ) -> torch.Tensor:
        """One decode token per row against the window ring and the band's compressed pool."""
        bsz = x.size(0)
        if wctx is None:
            wctx = get_global_ctx().batch.attn_metadata.window_ctx(pos, rows)
        window_slots, prev_window_slots, window_slots_topk = wctx
        rd = self.rope_head_dim
        freqs_t = self._freqs_cis.index_select(0, pos)

        qr = self.q_norm.forward(self.wq_a.forward(x))
        q = self.wq_b.forward(qr).unflatten(-1, (self.n_heads, self.head_dim))
        apply_rotary_emb_decode(q[..., -rd:], freqs_t)

        kv = self.kv_norm.forward(self.wkv.forward(x))
        apply_rotary_emb_decode(kv[..., -rd:], freqs_t)
        act_quant_fp8_inplace(kv, 32)
        self.attn.store_window(kv.view(bsz, -1), self.layer_id, window_slots)

        cmp_counts = None
        if self._has_compression:
            ratio = self.band_ratio
            n_stage = (cmp_stage_cap + 1) // ratio
            latent = completed = None
            if self.compressor is not None:
                latent, completed = self.compressor.decode_paged(
                    x, pos, prev_window_slots, window_slots
                )
            if self.indexer is not None:
                blocks = self.indexer.decode_paged(x, qr, latent, pos, n_stage, rows)
            else:
                assert self._runtime.topk_idxs is not None, (
                    f"layer {self.layer_id} reads its band's picks but source "
                    f"{self.kv_source_layer} has no indexer"
                )
                blocks = self._runtime.topk_idxs[:bsz, :1, : min(self.index_topk, n_stage)]
            if latent is not None:
                self.compressor.store_decode(latent, rows, pos, completed)
            compress_idxs = self.attn.blocks_to_global(blocks, ratio, rows=rows)
            topk_idxs = torch.cat([window_slots_topk, compress_idxs], dim=-1)
            cmp_counts = ((pos + 1) // ratio).clamp(max=compress_idxs.shape[-1])
            cmp_counts = cmp_counts.to(torch.int32).view(bsz, 1)
        else:
            topk_idxs = window_slots_topk
        o = self.attn.attend(
            q, self.layer_id, topk_idxs.int(), self.window_size, self.attn_sink,
            self.softmax_scale, cmp_counts=cmp_counts, has_compression=self._has_compression,
        )
        apply_rotary_emb_decode(o[..., -rd:], freqs_t, True)
        return self._wo(o, bsz, 1)

    @torch.no_grad()
    def forward(
        self, x: torch.Tensor, start_pos: int = 0, *, trace: dict | None = None
    ) -> torch.Tensor:
        """One attention step. ``x`` is [B, S, dim]; returns [B, S, dim].

        ``S == 1`` with ``start_pos > 0`` is a decode step against the ring window and the shared
        compressed pool; ``start_pos == 0`` is a cold prefill. ``trace`` (optional dict) is filled
        with the intermediate tensors the oracle dumps carry (``q`` / ``kv`` / ``idxs`` / ``o`` /
        ``window_kv`` / ``compress_kv``) so a numeric test compares stage by stage instead of only
        at the block output.
        """
        bsz, seqlen, _ = x.shape
        if self._freqs_cis is None:
            self.bind(x.device)
        freqs = self._freqs_cis[start_pos:start_pos + seqlen]

        rd = self.rope_head_dim
        # NB: no extra norm on q after wq_b -- V4 adds an rms_norm here, V4.1 does not. ``qr`` is
        # the query *latent* the indexer scores with; it is this layer's own, not a compressed one.
        qr = self.q_norm.forward(self.wq_a.forward(x))
        q = self.wq_b.forward(qr).unflatten(-1, (self.n_heads, self.head_dim))
        apply_rotary_emb(q[..., -rd:], freqs)

        window_kv, idxs = self._window_kv(x, freqs, start_pos)
        kv, compress_kv = window_kv, None
        if self.compress_ratio:
            compress_kv, compress_idxs = self._compress_kv(x, qr, start_pos, window_kv.size(1))
            kv = torch.cat([window_kv, compress_kv], dim=1)
            idxs = torch.cat([idxs, compress_idxs], dim=-1)
        o = sparse_attn_torch(q, kv, self.attn_sink, idxs, self.softmax_scale)
        if trace is not None:
            # Snapshot: `o` is rotated in place just below, so hand out copies, not views --
            # the reference dump records the tail *before* the inverse rotation.
            trace.update(
                q=q.clone(), kv=kv.clone(), idxs=idxs.clone(), o=o.clone(), freqs_cis=freqs,
                window_kv=window_kv.clone(),
                compress_kv=None if compress_kv is None else compress_kv.clone(),
            )

        apply_rotary_emb(o[..., -rd:], freqs, True)
        out = self._wo(o, bsz, seqlen)
        if trace is not None:
            trace["attn_out"] = out
        return out


__all__ = ["Attention"]
