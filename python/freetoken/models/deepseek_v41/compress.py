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

from freetoken.core import get_global_ctx
from freetoken.kernel.triton.fp4_e4m3_act import fp4_act_quant_e4m3_inplace
from freetoken.layers import BaseOP, LinearReplicated, RMSNorm

from ..deepseek_v4.ops import apply_rotary_emb, apply_rotary_emb_decode
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
        self.rope_head_dim = args.rope_head_dim
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
        # Paged addressing (M5): bound by ``bind_paged``. The pool itself is never stored -- the
        # backend and the live pool are read through ``attn`` per access.
        self.layer_id: int | None = None
        self.tier: str = "attn"
        self._freqs_cis: torch.Tensor | None = None

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

    # ----- paged path (M5) ----------------------------------------------------------------
    @property
    def attn(self):
        """The paged backend. Read live, like the pool: a runtime rebuild needs no unbind."""
        return get_global_ctx().attn_backend

    @property
    def paged(self) -> bool:
        return self.layer_id is not None

    def bind_paged(self, layer_id: int, freqs_cis: torch.Tensor, tier: str = "attn") -> None:
        self.layer_id = layer_id
        self.tier = tier
        self._freqs_cis = freqs_cis

    @staticmethod
    def _gated_pool(kv: torch.Tensor, score: torch.Tensor) -> torch.Tensor:
        """``(kv * softmax(score)).sum`` over the group axis (the reference's gated pool)."""
        return (kv * score.softmax(dim=-2)).sum(dim=-2)

    def _empty_group(self, device) -> tuple[torch.Tensor, torch.Tensor]:
        """A group with no tokens yet: zero kv and a ``-inf`` score, so it contributes nothing."""
        kv = torch.zeros(self.ratio, self.head_dim, dtype=torch.float32, device=device)
        score = torch.full(
            (self.ratio, self.head_dim), float("-inf"), dtype=torch.float32, device=device
        )
        return kv, score

    def _ring_read(self, ti: int, group_start: int):
        """The partial group persisted at ``group_start``: ``(kv, score, its window slot)``."""
        slots = self.attn.window_slots_of(ti, group_start, group_start + 1)
        block = self.attn.read_state(self.layer_id, self.tier, slots, self.ratio)[0]
        return block[..., : self.head_dim].clone(), block[..., self.head_dim :].clone(), slots

    def _ring_write(self, slots: torch.Tensor, kv: torch.Tensor, score: torch.Tensor) -> None:
        self.attn.write_state(
            self.layer_id, self.tier, slots, self.ratio,
            torch.cat([kv, score], dim=-1).unsqueeze(0),
        )

    @torch.no_grad()
    def compress_paged(self, x: torch.Tensor, start_pos: int, ti: int):
        """Pool this segment's token groups into the band's pool.

        Returns ``(latent [1, n_groups, head_dim], group_starts [n_groups])`` with the latent still
        UNROTATED - the indexer scores it in that state, and :meth:`store_paged` finishes it. The
        group that straddles the segment's start is completed from the ring the previous segment
        persisted; a trailing partial group is written back. ``None`` means no group completed.
        """
        ratio = self.ratio
        assert x.size(0) == 1, "the paged prefill path walks single-request segments"
        n = x.size(1)
        if ratio == 1:
            # No group state at all: every token is its own group, roped at its own position.
            latent = self.norm.forward(self.wkv.forward(x))
            return latent, torch.arange(start_pos, start_pos + n, device=x.device)

        end = start_pos + n
        device = x.device
        xf = x.float()
        kv = self.wkv.forward(xf)[0]
        score = self.wgate.forward(xf)[0]
        r = start_pos % ratio
        g0 = start_pos - r
        pooled, starts = [], []
        kv_s, sc_s = self._empty_group(device)
        phase = r
        if r:
            kv_s, sc_s, slots = self._ring_read(ti, g0)
            take = min(ratio - r, n)
            kv_s[phase : phase + take] = kv[:take]
            sc_s[phase : phase + take] = score[:take]
            if phase + take < ratio:
                # the segment ended inside the carried group: persist the longer partial and stop
                self._ring_write(slots, kv_s, sc_s)
                return None, None
            pooled.append(self._gated_pool(kv_s, sc_s).unsqueeze(0))
            starts.append(torch.tensor([g0], device=device))
            g0 = start_pos + take
        whole_end = (end // ratio) * ratio
        if g0 < whole_end:
            m = whole_end - g0
            lo = g0 - start_pos
            pooled.append(
                self._gated_pool(
                    kv[lo : lo + m].view(-1, ratio, self.head_dim),
                    score[lo : lo + m].view(-1, ratio, self.head_dim),
                )
            )
            starts.append(torch.arange(g0, whole_end, ratio, device=device))
            g0 = whole_end
        if g0 < end:
            # ratio divides the page size, so this partial group stays inside one window page and
            # the next segment reads it back off the same ring row.
            tail = end - g0
            finish = end - start_pos
            kv_t, sc_t = self._empty_group(device)
            kv_t[:tail] = kv[finish - tail : finish]
            sc_t[:tail] = score[finish - tail : finish]
            slots = self.attn.window_slots_of(ti, g0, g0 + 1)
            self._ring_write(slots, kv_t, sc_t)
        if not pooled:
            return None, None
        blocks = torch.cat(pooled, dim=0)
        return self.norm.forward(blocks.to(x.dtype)).unsqueeze(0), torch.cat(starts)

    @torch.no_grad()
    def store_paged(self, latent: torch.Tensor, starts: torch.Tensor, ti: int) -> None:
        """Rope, fp4-round-trip and scatter the pooled latents into the band's compressed pool."""
        rd = self.rope_head_dim
        apply_rotary_emb(latent[..., -rd:], self._freqs_cis.index_select(0, starts))
        fp4_act_quant_e4m3_inplace(latent, 16)
        rows = self.attn.compress_rows_of(ti, starts, self.ratio)
        self.attn.scatter_compressed(self.layer_id, self.tier, rows, latent[0])

    @torch.no_grad()
    def decode_paged(
        self, x: torch.Tensor, pos: torch.Tensor, prev_window_slots: torch.Tensor,
        window_slots: torch.Tensor,
    ):
        """One decode token per row: ``(latent [B, 1, head_dim], completed [B])``.

        The group is keyed by its START's window slot, which is why the carried half is read off
        the PREVIOUS token's page: a group never straddles a window page (P % ratio == 0).
        """
        ratio = self.ratio
        if ratio == 1:
            return (
                self.norm.forward(self.wkv.forward(x)),
                torch.ones(x.size(0), dtype=torch.bool, device=x.device),
            )
        bsz = x.size(0)
        xf = x.float()
        kv = self.wkv.forward(xf).squeeze(1)
        score = self.wgate.forward(xf).squeeze(1)
        phase = pos % ratio
        # The ring block belongs to the whole group, so it is addressed by the GROUP's slot, not
        # the token's: page groups start at multiples of the ratio, i.e. at in-page phase 0, so
        # the base is the current slot on a group's first token and the previous one otherwise.
        group_slots = torch.where(phase == 0, window_slots, prev_window_slots)
        block = self.attn.read_state(self.layer_id, self.tier, group_slots, ratio)
        fresh = (phase == 0)[:, None, None]
        kv_s = torch.where(
            fresh, torch.zeros_like(block[..., : self.head_dim]), block[..., : self.head_dim]
        )
        sc_s = torch.where(
            fresh,
            torch.full_like(block[..., self.head_dim :], float("-inf")),
            block[..., self.head_dim :],
        )
        rows = torch.arange(bsz, device=x.device)
        kv_s[rows, phase] = kv
        sc_s[rows, phase] = score
        self.attn.write_state(
            self.layer_id, self.tier, group_slots, ratio, torch.cat([kv_s, sc_s], dim=-1)
        )
        latent = self.norm.forward(self._gated_pool(kv_s, sc_s).to(x.dtype))
        return latent.unsqueeze(1), (pos + 1) % ratio == 0

    @torch.no_grad()
    def store_decode(
        self, latent: torch.Tensor, rows: torch.Tensor, pos: torch.Tensor, completed: torch.Tensor
    ) -> None:
        """Rope/fp4 the decode latent and scatter it, or to this row's scratch when the group is
        still open (a discarded write that keeps the masked store graph-safe)."""
        rd = self.rope_head_dim
        freqs = self._freqs_cis.index_select(0, (pos + 1 - self.ratio).clamp_min(0))
        apply_rotary_emb_decode(latent[..., -rd:], freqs)
        fp4_act_quant_e4m3_inplace(latent, 16)
        dst = self.attn.decode_compress_rows(
            rows, pos, self.ratio, self.layer_id, self.tier, completed
        )
        self.attn.scatter_compressed(self.layer_id, self.tier, dst, latent.squeeze(1))


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
