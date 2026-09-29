"""CSA2 compressor: pools ``compress_ratio`` tokens into one KV latent.

Verbatim port of the reference ``Compressor`` (``inference/model.py:429-485``): ratio 1 is a plain
bf16 projection (layer 20), ratio 2 is an fp32 softmax-gated pool over token pairs (layers 2/8/14).
An incomplete trailing group is *carried* in ``kv_state``/``score_state`` across decode steps, so
the same buffers must survive between forwards -- hence the lazy ``_state`` allocation rather than
a module parameter (the reference registers them ``persistent=False``; the engine's ``state_dict``
walks tensors, and these must never be loaded from the checkpoint).

The output is the **pre-RoPE** latent: the attention layer rotates and fp4-quantizes it on its way
into the shared compressed-KV cache.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from freetoken.layers import BaseOP, LinearReplicated, RMSNorm

from .args import DeepseekV41Args


class Compressor(BaseOP):
    """``ratio`` consecutive tokens -> one compressed KV latent."""

    def __init__(
        self,
        layer_id: int,
        args: DeepseekV41Args,
        *,
        quant_config=None,
        prefix: str = "",
    ):
        self.ratio = args.layer_ratio(layer_id)
        self.head_dim = args.head_dim
        self.max_batch_size = args.max_batch_size
        self.norm = RMSNorm(self.head_dim, args.norm_eps)
        # the reference keeps these fp32 (ratio 2) / bf16 (ratio 1) as *parameters*; the
        # checkpoint stores bf16 and the unquantized kernel upcasts on an fp32 activation stream,
        # which is the same arithmetic.
        self.wkv = LinearReplicated(
            args.dim, self.head_dim, False, quant_config=quant_config, prefix=f"{prefix}.wkv"
        )
        self.wgate = None
        if self.ratio > 1:
            self.wgate = LinearReplicated(
                args.dim, self.head_dim, False, quant_config=quant_config, prefix=f"{prefix}.wgate"
            )
        # [max_batch, ratio, head_dim] fp32 carry of an incomplete group (score: nothing yet)
        self._kv_state: torch.Tensor | None = None
        self._score_state: torch.Tensor | None = None

    def _state(self, x: torch.Tensor, bsz: int) -> tuple[torch.Tensor, torch.Tensor]:
        need = max(bsz, self.max_batch_size)
        if self._kv_state is None or self._kv_state.shape[0] < need:
            self._kv_state = torch.zeros(
                need, self.ratio, self.head_dim, dtype=torch.float32, device=x.device
            )
            self._score_state = torch.full(
                (need, self.ratio, self.head_dim),
                float("-inf"),
                dtype=torch.float32,
                device=x.device,
            )
        return self._kv_state, self._score_state

    def reset(self) -> None:
        """Drop the carried group (a new sequence starts)."""
        self._kv_state = None
        self._score_state = None

    def forward(self, x: torch.Tensor, start_pos: int) -> torch.Tensor | None:
        """``x`` [b, s, dim] -> latent [b, s // ratio, head_dim], or None if no group completes."""
        if self.ratio == 1:
            return self.norm.forward(self.wkv.forward(x))

        bsz, seqlen, _ = x.shape
        xf = x.float()
        kv = self.wkv.forward(xf)
        score = self.wgate.forward(xf)
        kv_state, score_state = self._state(x, bsz)

        if start_pos == 0:
            should_compress = seqlen >= self.ratio
            remainder = seqlen % self.ratio
            cutoff = seqlen - remainder
            if remainder:
                kv, tail = kv.split([cutoff, remainder], dim=1)
                score, score_tail = score.split([cutoff, remainder], dim=1)
                kv_state[:bsz, :remainder] = tail
                score_state[:bsz, :remainder] = score_tail
            # the reference pools unconditionally: with seqlen < ratio this is an empty group
            kv = kv.unflatten(1, (-1, self.ratio))
            score = score.unflatten(1, (-1, self.ratio))
            kv = (kv * score.softmax(dim=2)).sum(dim=2)
        else:
            should_compress = (start_pos + 1) % self.ratio == 0
            slot = start_pos % self.ratio
            kv_state[:bsz, slot] = kv.squeeze(1)
            score_state[:bsz, slot] = score.squeeze(1)
            if should_compress:
                kv = (kv_state[:bsz] * score_state[:bsz].softmax(dim=1)).sum(dim=1, keepdim=True)

        if not should_compress:
            return None
        return self.norm.forward(kv.to(x.dtype))


def select_candidate_blocks(
    logits: torch.Tensor,
    compress_lens: torch.Tensor | int,
    topk_blocks: int,
    block_size: int,
) -> torch.Tensor:
    """Two-level indexer: the candidate *mask* published by the ``candidate_source_layer``.

    Verbatim ``inference/model.py:487-510``. A block scores as the max of its members, the newest
    (partial) block is pinned ``+inf`` so it can never be dropped, and the top ``topk_blocks``
    survive. Returns a bool mask shaped like ``logits``.
    """
    width = logits.size(-1)
    scores = F.pad(logits, (0, -width % block_size), value=float("-inf"))
    scores = scores.unflatten(-1, (-1, block_size)).amax(dim=-1)
    num_blocks = scores.size(-1)
    last = (torch.as_tensor(compress_lens, device=logits.device) - 1) // block_size
    scores = scores.masked_fill(
        torch.arange(num_blocks, device=logits.device) == last, torch.inf
    )
    top = scores.topk(min(topk_blocks, num_blocks), dim=-1)
    keep = torch.zeros_like(scores, dtype=torch.bool)
    keep = keep.scatter_(-1, top.indices, top.values > float("-inf"))
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]


__all__ = ["Compressor", "select_candidate_blocks"]
