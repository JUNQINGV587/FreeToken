"""Paged addressing for the DeepSeek-V4 compressors, mixed into the sparse-attention backend.

DSV4 runs two compressors per compressed layer -- the attention one (writes ``cmp_pool``) and
the Lightning Indexer's own (writes ``idx_pool``) -- over three tiers of paged state:

* the compressed-KV pool row for a block, which is arithmetic off the block's full loc
  (``full_loc(b * ratio) // ratio``), so it needs no slot map;
* the compress-state RING, which carries the rolling reduction across page and request
  boundaries so a radix hit can resume a prefix by value. Overlap (ratio-4) rings are keyed by
  window page -- that page's carry IS read back (a radix match seeds from it); non-overlap
  (ratio-128) rings are keyed by the resident request's page-table row, because their per-page
  copy is provably dead (the extend zeroes the register before reducing, and a radix match
  re-derives the block from its own tokens);
* a per-row scratch row, the destination for a decode step whose block did not complete (a
  discarded write that keeps the masked store graph-safe and collision-free).

All of that is addressing, so it lives here rather than in the model: the module keeps its
weights and the reduction math and passes them in, mirroring sglang's ``CompressorBackendMixin``
(``forward_compress(*, ape, norm, freqs_cis_cache, ...)``).
"""

from __future__ import annotations

import os

import torch
from freetoken.core import get_global_ctx


class CompressorBackendMixin:
    # DSV4_RING_CHECK=1 -> assert the ring read-back equals what was written. Off by default.
    _RING_CHECK = os.environ.get("DSV4_RING_CHECK", "0") == "1"

    # tier -> (KV-pool, state-ring, scratch-base, seed-store) attribute names on the pool.
    _TIER_ATTRS = {
        "attn": ("cmp_pool", "state_ring", "cmp_scratch_base", "state_seed"),
        "idx": ("idx_pool", "indexer_state_ring", "idx_scratch_base", "indexer_state_seed"),
    }

    # ----- pool views ------------------------------------------------------------------
    def compress_pool(self, layer_id: int, tier: str) -> torch.Tensor:
        return getattr(self.pool, self._TIER_ATTRS[tier][0])[layer_id]

    def compress_state_ring(self, layer_id: int, tier: str):
        return getattr(self.pool, self._TIER_ATTRS[tier][1])[layer_id]

    def compress_state_seed(self, layer_id: int, tier: str):
        """The tier's overlap seed store (``None`` on tiers that keep no page seed)."""
        return getattr(self.pool, self._TIER_ATTRS[tier][3])[layer_id]

    def compress_scratch_base(self, layer_id: int, tier: str) -> int:
        return getattr(self.pool, self._TIER_ATTRS[tier][2])[layer_id]

    # ----- compressed-row addressing ---------------------------------------------------
    def compress_rows_of(self, ti: int, block_starts: torch.Tensor, ratio: int) -> torch.Tensor:
        """Rows for blocks whose ABSOLUTE start positions are ``block_starts``, off the request's
        LIVE full locs (prefill/extend). A full page is ratio-divisible, so every position in a
        block shares one row."""
        return self.pool.cmp_rows(self.pool.full_loc_map[ti, block_starts], ratio)

    def decode_compress_rows(
        self, rows: torch.Tensor, pos: torch.Tensor, ratio: int, layer_id: int, tier: str,
        completed: torch.Tensor,
    ) -> torch.Tensor:
        """Per-row decode store destination: the completed block's arithmetic row, or the row's
        OWN scratch row when this step did not finish a block.

        Reads the decode SNAPSHOT (not the live map) so a concurrent allocate_paged cannot
        redirect the write; scratch keeps the masked store free of negative indices (which
        ``index_copy_`` would treat as out of bounds) and collision-free across rows.
        """
        row_of_block = self.pool.cmp_rows(self.snapshot()[rows, pos], ratio)
        scratch = rows + self.compress_scratch_base(layer_id, tier)
        return torch.where(completed, row_of_block, scratch)

    def scatter_compressed(
        self, layer_id: int, tier: str, rows: torch.Tensor, kv: torch.Tensor
    ) -> None:
        if tier == "idx":
            # The indexer tier stores fp4 rows there, so route the write through the pool's
            # packer (bf16 is what the reference cache would hold, not what this pool holds).
            self.pool.store_indexer(kv, layer_id, rows)
            return
        pool = self.compress_pool(layer_id, tier)
        pool.index_copy_(0, rows, kv.to(pool.dtype))

    # ----- compress-state ring ---------------------------------------------------------
    def carry_state_loc(self, window_slot: int, ring_size: int) -> torch.Tensor:
        """The window page's ring block: ``(window_slot // P) * ring_size + arange(ring_size)``."""
        base = (window_slot // self.window_size) * ring_size
        return torch.arange(base, base + ring_size, device=self.device, dtype=torch.int64)

    def ring_page_base(self, window_slots: torch.Tensor, ring_size: int) -> torch.Tensor:
        """Per-row ring block base. Distinct window pages map to disjoint blocks, which is what
        keeps co-tenant decode rows from contaminating each other's carry."""
        return torch.div(window_slots, self.window_size, rounding_mode="floor") * ring_size

    def request_state_loc(self, row: int, ring_size: int) -> torch.Tensor:
        """A request-keyed ring's block: ``row * ring_size + arange(ring_size)``. ``row`` is the
        resident request's page-table row, NOT a batch position -- batch positions move when a
        request finishes, so a carry keyed by one would follow the wrong request."""
        base = row * ring_size
        return torch.arange(base, base + ring_size, device=self.device, dtype=torch.int64)

    def ring_block_base(
        self, ring, window_slots: torch.Tensor, ring_size: int, rows: torch.Tensor | None
    ) -> torch.Tensor:
        """Block bases for a decode row set: page-keyed (overlap) rings take the window slots,
        request-keyed (non-overlap) rings take the batch's page-table rows."""
        if ring.request_keyed:
            assert rows is not None, "request-keyed ring needs the batch's page-table rows"
            return rows * ring_size
        return self.ring_page_base(window_slots, ring_size)

    def request_rows(self) -> torch.Tensor:
        """The decode batch's stable per-row key: its page-table row (see ``ring_block_base``)."""
        md = get_global_ctx().batch.attn_metadata
        rows = getattr(md, "table_rows", None)
        assert rows is not None, "decode metadata is missing its page-table rows"
        return rows if rows.dtype == torch.int64 else rows.to(torch.int64)

    def read_carry(
        self, layer_id: int, tier: str, window_slot: int, ring_size: int
    ) -> torch.Tensor | None:
        """The compact overlap seed at ``window_slot``'s page -- ``[2 * ratio * head_dim]``, kv
        quadrant then score quadrant. Used to resume a prefix by value on a radix hit; ``None``
        where the tier keeps no page seed (the non-overlap tiers' per-page carry is dead)."""
        seed = self.compress_state_seed(layer_id, tier)
        if seed is None:
            return None
        return seed.get(self.seed_page_loc(window_slot))

    def seed_page_loc(self, window_slot: int) -> torch.Tensor:
        """Seed slot of a window slot: its page. A radix match is page-aligned, so only a whole
        page's quadrant is ever read back."""
        return torch.tensor(window_slot // self.window_size, device=self.device)

    def write_seed(
        self, layer_id: int, tier: str, window_slot: int, kv_rows: torch.Tensor,
        score_rows: torch.Tensor,
    ) -> None:
        """Persist a page's overlap seed: ``ratio`` rows of the kv half then of the score half,
        each row ``item`` wide (the whole row is load-bearing -- the extend's overlap_transform
        reads its second half)."""
        seed = self.compress_state_seed(layer_id, tier)
        if seed is None:
            return
        flat = torch.cat([kv_rows.reshape(-1), score_rows.reshape(-1)])
        page = self.seed_page_loc(window_slot)
        seed.set(page, flat)
        if self._RING_CHECK:
            assert torch.equal(seed.get(page), flat), (
                f"seed read-back != written rows (layer={layer_id} tier={tier} slot="
                f"{window_slot})"
            )

    def write_seed_blocks(
        self, layer_id: int, tier: str, slots: torch.Tensor, kv_rows: torch.Tensor,
        score_rows: torch.Tensor,
    ) -> None:
        """Batched per-row seed write at ``slots`` ``[B]`` (``-1`` -> discarded scratch row)."""
        seed = self.compress_state_seed(layer_id, tier)
        if seed is None:
            return
        B = slots.size(0)
        flat = torch.cat([kv_rows.reshape(B, -1), score_rows.reshape(B, -1)], dim=-1)
        seed.set_blocks(slots, flat)
        if self._RING_CHECK:
            assert torch.equal(seed.get(slots), flat), "seed read-back != written rows"

    def write_carry(
        self, layer_id: int, tier: str, window_slot: int, ring_size: int, kv_score: torch.Tensor,
        row: int | None = None,
    ) -> None:
        ring = self.compress_state_ring(layer_id, tier)
        if ring is None:
            return
        if ring.request_keyed:
            assert row is not None, "request-keyed ring needs the request's page-table row"
            state_loc = self.request_state_loc(int(row), ring_size)
        else:
            state_loc = self.carry_state_loc(window_slot, ring_size)
        ring.set(state_loc, kv_score)
        if self._RING_CHECK:
            assert torch.equal(ring.get(state_loc), kv_score), (
                f"ring read-back != written block (layer={layer_id} tier={tier} "
                f"window_slot={window_slot})"
            )

    def read_carry_blocks(
        self, layer_id: int, tier: str, window_slots: torch.Tensor, ring_size: int,
        rows: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Per-row carry blocks ``[B, ring_size, 2 * item]`` for a decode step."""
        ring = self.compress_state_ring(layer_id, tier)
        return ring.get_blocks(self.ring_block_base(ring, window_slots, ring_size, rows))

    def write_carry_blocks(
        self, layer_id: int, tier: str, window_slots: torch.Tensor, ring_size: int,
        blocks: torch.Tensor, rows: torch.Tensor | None = None,
    ) -> None:
        ring = self.compress_state_ring(layer_id, tier)
        ring.set_blocks(self.ring_block_base(ring, window_slots, ring_size, rows), blocks)

    def write_boundary_carries(
        self, *, layer_id: int, tier: str, ratio: int, overlap: bool, ring_size: int,
        ape: torch.Tensor, kv: torch.Tensor, score: torch.Tensor, lo: int, hi: int,
        window_slots: torch.Tensor, row: int | None = None,
    ) -> None:
        """Persist the carry at EVERY window-page boundary in ``(lo, hi]``, so any future radix
        match (always page-aligned) can read its carry by value.

        The carry at a page-aligned boundary B is position-local: with overlap (ratio 4) it is
        ``kv[B-ratio:B]`` in the overlap-seed slots, without it (ratio 128) it is the reset
        state. ``kv``/``score`` cover ``[lo, hi)``; ``window_slots`` is indexed by ``pos - lo``.
        The first boundary written is ``lo + P`` -- the one AT ``lo`` is the seed source and is
        already in the ring.

        An overlap tier writes two places: the page's compact seed (the only part a later request
        reads back) and the requesting row's live carry block, which the next decode step rolls
        from. A non-overlap tier writes nothing: its boundary block is the reset constant and
        nothing reads a page's block back (the model's extend zeroes the register before
        reducing).
        """
        seed = self.compress_state_seed(layer_id, tier)
        if seed is None:
            return
        P = self.window_size
        item = self.compress_state_ring(layer_id, tier).item_size
        for B in range((lo // P + 1) * P, hi + 1, P):
            ws = int(window_slots[B - 1 - lo].item())
            if overlap:
                blk_kv = kv.new_zeros(ring_size, item, dtype=torch.float32)
                blk_ss = kv.new_full((ring_size, item), float("-inf"), dtype=torch.float32)
                blk_kv[:ratio] = kv[0, B - ratio - lo: B - lo]
                blk_ss[:ratio] = score[0, B - ratio - lo: B - lo] + ape
                assert row is not None, "overlap boundary write needs the request's row"
                self.write_carry(
                    layer_id, tier, ws, ring_size, torch.cat([blk_kv, blk_ss], dim=-1), row=row,
                )
                self.write_seed(layer_id, tier, ws, blk_kv[:ratio], blk_ss[:ratio])


__all__ = ["CompressorBackendMixin"]
