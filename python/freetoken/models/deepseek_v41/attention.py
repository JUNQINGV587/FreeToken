"""DeepSeek-V4.1 (CSA2) attention -- the eager layer-0 path.

V4.1 attention is ONE sparse pass per layer over

    [ this layer's own 128-slot sliding window | up to ``index_topk`` (512) selected compressed slots ]

The compressed pool is *published* by a kv-source layer (2/8/14/20) and *consumed* by every layer in
its band (3-7 <- 2, 9-13 <- 8, 15-19 <- 14, 21-39 <- 20), which is why the engine keeps its
compressed KV in the shared paged pool rather than per layer.

Ratio 0 (layers 0, 1 and the MTP layers) has no compressor and no indexer: attention is the window
half alone. That is what this module implements eagerly and what the layer-0 oracle dump
(``block0_ref.pt``) pins down numerically. Two pieces are deliberately deferred:

  * ratio 2/1 compressors and the two-level indexer (M3),
  * the paged KV backend + shared compressed-KV runtime (M2) -- so ``forward`` currently accepts a
    cold prefill (``start_pos == 0``) only, where no cache state has to be carried.
"""

from __future__ import annotations

import torch

from freetoken.kernel.triton.dsv4.fp8_linear import act_quant_fp8_inplace
from freetoken.layers import (
    BaseOP,
    LinearColParallelMerged,
    LinearReplicated,
    LinearRowParallel,
    RMSNorm,
)

from .args import DeepseekV41Args
from .ops import apply_rotary_emb, get_freqs_cis, get_window_topk_idxs, sparse_attn_torch


class Attention(BaseOP):
    """CSA2 attention over the layer's own window (M1) and, later, the shared compressed pool."""

    def __init__(
        self, layer_id: int, args: DeepseekV41Args, *, quant_config=None, prefix: str = ""
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
        self.compress_ratio = args.layer_ratio(layer_id)
        self.is_kv_source = args.is_kv_source(layer_id)
        self.is_index_source = args.is_index_source(layer_id)
        self.kv_source_layer = args.kv_source_for(layer_id)

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

        if self.compress_ratio:
            # Ratio 2 pools KV pairs with a gated softmax, ratio 1 is a plain projection; both live
            # in the shared compressed pool, and the indexer that selects from that pool is the
            # two-level (publisher L20 + per-layer selector) structure. M3.
            raise NotImplementedError(
                f"DeepSeek-V4.1 layer {layer_id}: compress_ratio={self.compress_ratio} needs the "
                "compressor + two-level indexer (M3); only ratio-0 layers run in M1."
            )

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

    def bind(self, device: torch.device) -> None:
        self._freqs_cis = get_freqs_cis(*self._freqs_params, device)

    def _window_kv(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        """Project, rope and *quantize round-trip* this layer's whole window KV.

        The reference quantizes the FULL head vector (rope tail included) with
        ``act_quant(kv, 32, "ue8m0", e8m0, inplace=True)`` -- unlike V4, which quantizes only the
        non-rope part at block 64. Any deviation here shows up immediately in the ``kv`` probe.
        """
        kv = self.kv_norm.forward(self.wkv.forward(x))
        apply_rotary_emb(kv[..., -self.rope_head_dim:], freqs)
        act_quant_fp8_inplace(kv, 32)
        return kv

    def _wo(self, o: torch.Tensor, bsz: int, seqlen: int) -> torch.Tensor:
        o = o.reshape(bsz, seqlen, self.n_groups, -1)
        wo_a = self.wo_a.view(self.n_groups, self.o_lora_rank, -1)
        o = torch.einsum("bsgd,grd->bsgr", o, wo_a).flatten(2)
        return self.wo_b.forward(o)

    @torch.no_grad()
    def forward(
        self, x: torch.Tensor, start_pos: int = 0, *, trace: dict | None = None
    ) -> torch.Tensor:
        """Cold-prefill attention. ``x`` is [B, S, dim]; returns [B, S, dim].

        ``trace`` (optional dict) is filled with the intermediate tensors the layer-0 oracle dump
        carries (``q`` / ``kv`` / ``idxs`` / ``o``) so a numeric test can compare stage by stage
        instead of only at the output.
        """
        if start_pos != 0:
            raise NotImplementedError(
                "DeepSeek-V4.1 attention: start_pos > 0 needs the paged KV backend "
                "(window ring + shared compressed pool); M2."
            )
        bsz, seqlen, _ = x.shape
        if self._freqs_cis is None:
            self.bind(x.device)
        freqs = self._freqs_cis[start_pos:start_pos + seqlen]

        rd = self.rope_head_dim
        # NB: no extra norm on q after wq_b -- V4 adds an rms_norm here, V4.1 does not.
        q = self.wq_b.forward(self.q_norm.forward(self.wq_a.forward(x))).unflatten(-1, (self.n_heads, self.head_dim))
        apply_rotary_emb(q[..., -rd:], freqs)

        kv = self._window_kv(x, freqs)
        idxs = get_window_topk_idxs(self.window_size, bsz, seqlen, start_pos).to(x.device)
        o = sparse_attn_torch(q, kv, self.attn_sink, idxs, self.softmax_scale)
        if trace is not None:
            # Snapshot: `o` is rotated in place just below, so hand out copies, not views --
            # the reference dump records the tail *before* the inverse rotation.
            trace.update(q=q.clone(), kv=kv.clone(), idxs=idxs.clone(), o=o.clone(), freqs_cis=freqs)

        apply_rotary_emb(o[..., -rd:], freqs, True)
        out = self._wo(o, bsz, seqlen)
        if trace is not None:
            trace["attn_out"] = out
        return out


__all__ = ["Attention"]
