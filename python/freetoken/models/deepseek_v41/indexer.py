"""CSA2 two-level indexer: which compressed positions each query may attend to.

Faithful port of the reference ``Indexer`` (``inference/model.py:512-612``). One indexer runs per
index-source layer (2/8/14/20/24/28/32/36); only the four kv-source layers (2/8/14/20) own the K
projection -- they turn their own compressor latent into index keys, and every layer in the band
scores against that shared cache.

Two levels (the reason ``candidate_source_layer`` exists):

  * layer 20 publishes a *candidate mask*: the best ``candidate_topk_blocks`` blocks of
    ``candidate_block_size`` compressed positions, per query,
  * layers 24/28/32/36 score with their own ``wq_b`` + ``weights_proj`` but only inside that mask,
    then keep their own top ``index_topk``.

``SharedAttentionRuntime`` stands in for the reference's module-global ``shared_attn``: the
compressed-KV cache, the index-K cache, the candidate mask and the previous layer's ``topk_idxs``.
Bands run in layer order, so one instance per forward pass is enough; the paged backend replaces it
in M5.
"""

from __future__ import annotations

import torch

from freetoken.kernel.triton.dsv4.fp8_linear import fp4_act_quant_inplace
from freetoken.layers import BaseOP, LinearColParallelMerged, LinearReplicated, RMSNorm

from .args import DeepseekV41Args
from .compress import select_candidate_blocks
from .ops import apply_rotary_emb, get_freqs_cis


class SharedAttentionRuntime:
    """What a kv-source layer publishes for the rest of its band (reference ``shared_attn``).

    Deliberately *not* a ``BaseOP``: it holds no parameters, and the caches it carries appear
    mid-forward, so letting ``state_dict`` walk it would invent checkpoint keys.
    """

    def __init__(self):
        self.compress_kv: torch.Tensor | None = None
        self.index_k: torch.Tensor | None = None
        self.candidates: torch.Tensor | None = None
        self.topk_idxs: torch.Tensor | None = None

    def reset(self) -> None:
        self.compress_kv = None
        self.index_k = None
        self.candidates = None
        self.topk_idxs = None


class Indexer(BaseOP):
    """Scores compressed positions and returns the ``index_topk`` indices each query attends to."""

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
        self.owns_k = args.is_kv_source(layer_id)
        self.ratio = args.layer_ratio(layer_id)
        self.is_candidate_source = layer_id == args.candidate_source_layer
        self.uses_candidates = 0 <= args.candidate_source_layer < layer_id
        self.candidate_topk_blocks = args.candidate_topk_blocks
        self.candidate_block_size = args.candidate_block_size
        self.dim = args.dim
        self.n_heads = args.index_n_heads
        self.index_head_dim = args.index_head_dim
        self.rope_head_dim = args.rope_head_dim
        self.index_topk = args.index_topk
        self.softmax_scale = self.index_head_dim**-0.5
        self.max_batch_size = args.max_batch_size
        self.max_seq_len = args.max_seq_len
        self.runtime = runtime if runtime is not None else SharedAttentionRuntime()

        self.wq_b = LinearColParallelMerged(
            args.q_lora_rank, [self.n_heads * self.index_head_dim], has_bias=False,
            quant_config=quant_config, prefix=f"{prefix}.wq_b",
        )
        self.weights_proj = LinearReplicated(
            self.dim, self.n_heads, False, quant_config=quant_config, prefix=f"{prefix}.weights_proj"
        )
        self.wk = None
        self.k_norm = None
        if self.owns_k:
            self.wk = LinearReplicated(
                args.head_dim, self.index_head_dim, False,
                quant_config=quant_config, prefix=f"{prefix}.wk",
            )
            self.k_norm = RMSNorm(self.index_head_dim, args.norm_eps)
        self._k_cache: torch.Tensor | None = None

        # the index keys are roped with the *compressed* rope (every indexer sits on a compressed
        # layer), i.e. the same table its attention layer builds
        self._freqs_params = (
            self.rope_head_dim, args.max_seq_len, args.original_seq_len,
            args.compress_rope_theta, args.rope_factor, args.beta_fast, args.beta_slow,
        )
        self._freqs_cis: torch.Tensor | None = None

    def bind(self, device: torch.device) -> None:
        if self._freqs_cis is None:
            self._freqs_cis = get_freqs_cis(*self._freqs_params, device)

    def _cache(self, x: torch.Tensor) -> torch.Tensor:
        """Index-K cache, filled group by group by the band's kv-source layer."""
        if self._k_cache is None or self._k_cache.device != x.device:
            self._k_cache = torch.zeros(
                self.max_batch_size, self.max_seq_len // self.ratio, self.index_head_dim,
                dtype=x.dtype, device=x.device,
            )
        return self._k_cache

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        latent: torch.Tensor | None,
        start_pos: int,
        offset: int,
    ) -> torch.Tensor:
        """``x`` [b, s, dim], ``qr`` [b, s, q_lora_rank], ``latent`` the compressor's output (or
        None). Returns int32 ``[b, s, topk]`` indices into the compressed pool, ``-1`` = unreachable.
        """
        if self._freqs_cis is None:
            self.bind(x.device)
        bsz, seqlen, _ = x.shape
        ratio, rd, end_pos = self.ratio, self.rope_head_dim, start_pos + seqlen
        runtime = self.runtime

        if self.owns_k and latent is not None:
            # a latent stands for the first token of its group, so group j takes position j * ratio
            freqs = (
                self._freqs_cis[: seqlen - seqlen % ratio : ratio]
                if start_pos == 0
                else self._freqs_cis[start_pos + 1 - ratio].unsqueeze(0)
            )
            k = self.k_norm.forward(self.wk.forward(latent))
            apply_rotary_emb(k[..., -rd:], freqs)
            fp4_act_quant_inplace(k, 32)
            cache = self._cache(x)
            cache[:bsz, start_pos // ratio : start_pos // ratio + k.size(1)] = k
            runtime.index_k = cache

        q = self.wq_b.forward(qr).unflatten(-1, (self.n_heads, self.index_head_dim))
        apply_rotary_emb(q[..., -rd:], self._freqs_cis[start_pos:end_pos])
        fp4_act_quant_inplace(q, 32)

        index_k = runtime.index_k[:bsz, : end_pos // ratio]
        weights = self.weights_proj.forward(x) * (self.softmax_scale * self.n_heads**-0.5)
        index_score = torch.einsum("bshd,btd->bsht", q, index_k)
        index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)

        # a block becomes visible once the query has passed its last token
        if start_pos == 0:
            compress_lens = (torch.arange(1, seqlen + 1, device=x.device) // ratio).unsqueeze(-1)
            index_score.masked_fill_(
                torch.arange(seqlen // ratio, device=x.device) >= compress_lens, float("-inf")
            )
        else:
            compress_lens = end_pos // ratio

        if self.is_candidate_source:
            runtime.candidates = select_candidate_blocks(
                index_score, compress_lens, self.candidate_topk_blocks, self.candidate_block_size
            )
        elif self.uses_candidates:
            index_score = index_score.masked_fill(~runtime.candidates, float("-inf"))

        # top-k by score, re-sorted into position order; unreachable -> -1, rest shifted by offset
        topk = min(self.index_topk, end_pos // ratio)
        idxs = index_score.topk(topk, dim=-1, sorted=False).indices.sort(dim=-1).values
        return torch.where(idxs < compress_lens, idxs + offset, -1).int()


__all__ = ["Indexer", "SharedAttentionRuntime"]
