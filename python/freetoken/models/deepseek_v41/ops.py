"""Small math ops for DeepSeek-V4.1 (pure torch, faithful to ``inference/model.py``).

V4.1 shares the rotary / hyper-connection math with V4 (same reference lineage), so those are
re-exported from :mod:`freetoken.models.deepseek_v4.ops` rather than duplicated: the formulas are
``precompute_freqs_cis`` (YaRN; the ratio-0 layers disable it), ``apply_rotary_emb`` (in-place
interleaved complex) and ``hc_split_sinkhorn``.

What is new here is the *sparse* attention math. The reference kernel (``kernel.py:311``
``sparse_attn``) needs 138 KiB of dynamic shared memory per block, past SM89's 99 KiB opt-in
maximum, so an Ada/Hopper-class card cannot launch it; the torch form below is the exact
semantics (index-gather + fp32 online softmax with the ``attn_sink`` term in the denominator) and
doubles as the reference the Triton port is checked against.
"""

from __future__ import annotations

import torch

# Rotary + hyper-connection math: identical to the reference in both V4 and V4.1.
from ..deepseek_v4.ops import (  # noqa: F401  (re-exported)
    apply_rotary_emb,
    get_freqs_cis,
    hc_split_sinkhorn,
    precompute_freqs_cis,
)


def get_window_topk_idxs(
    window_size: int, bsz: int, seqlen: int, start_pos: int
) -> torch.Tensor:
    """Which sliding-window cache slots each query attends to; ``-1`` marks an empty slot.

    Verbatim ``inference/model.py:410-426``. The window is a ring of ``window_size`` slots:
    prefill needs one row per query (each seeing its own causal window), a decode step has a
    single query that sees the whole ring oldest-first. Row order is immaterial to
    ``sparse_attn``, which treats every slot independently.
    """
    if start_pos == 0:
        end = torch.arange(seqlen).unsqueeze(1)
        idxs = (end - window_size + 1).clamp(0) + torch.arange(min(seqlen, window_size))
        idxs = torch.where(idxs > end, -1, idxs)  # before the sequence started
    else:
        oldest = start_pos % window_size + 1
        idxs = torch.cat([torch.arange(oldest, window_size), torch.arange(oldest)])
        idxs = torch.where(idxs > start_pos, -1, idxs)  # ring still filling
    # sparse_attn needs real [b, m, topk] int32 memory, hence the materializing expand
    return idxs.int().unsqueeze(0).expand(bsz, -1, -1).contiguous()


def sparse_attn_torch(
    q: torch.Tensor,
    kv: torch.Tensor,
    sink: torch.Tensor,
    idxs: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Reference ``sparse_attn`` in torch.

    ``q`` [b, m, h, d], ``kv`` [b, n, d] (single-head latent KV, shared by every head), ``sink``
    [h] fp32 attention sink logits, ``idxs`` [b, m, topk] int (``-1`` = empty slot, contributes
    nothing), ``scale`` = ``head_dim**-0.5``. Returns ``o`` [b, m, h, d] in ``q``'s dtype.
    """
    b, m, h, d = q.shape
    topk = idxs.shape[-1]
    valid = (idxs >= 0).unsqueeze(2)  # [b, m, 1, topk]
    g = idxs.clamp(min=0).reshape(b, -1).unsqueeze(-1).expand(-1, -1, d).reshape(b, m, topk, d)
    k = kv.gather(1, g.reshape(b, -1, d)).reshape(b, m, topk, d)  # [b, m, topk, d]
    logits = torch.einsum("bmhd,bmtd->bmht", q.float(), k.float()) * scale
    logits = logits.masked_fill(~valid, float("-inf"))
    sink_b = sink.float().view(1, 1, h, 1)
    mx = torch.maximum(logits.amax(-1, keepdim=True), sink_b)
    p = torch.exp(logits - mx)
    denom = p.sum(-1, keepdim=True) + torch.exp(sink_b - mx)
    return (torch.einsum("bmht,bmtd->bmhd", p, k.float()) / denom).to(q.dtype)


__all__ = [
    "apply_rotary_emb",
    "get_freqs_cis",
    "get_window_topk_idxs",
    "hc_split_sinkhorn",
    "precompute_freqs_cis",
    "sparse_attn_torch",
]
