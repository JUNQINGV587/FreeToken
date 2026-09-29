"""Paged addressing for the DeepSeek-V4.1 compressor, mixed into the sparse-attention backend.

V4.1 runs ONE compressor per KV source layer and shares its paged tiers with every consumer
layer of the band, so this mixin never indexes a pool list: it routes through the pool's band
accessors (``cmp_pool_of`` / ``idx_pool_of`` / ``state_ring_of`` / ...), which resolve a layer
to its band owner and therefore always return that owner's single tensor object
(``pool.cmp_pool[25] is pool.cmp_pool[20]``). Indexing ``pool.cmp_pool[layer_id]`` directly
would work only by accident of how the pool fills its per-layer lists.

Three tiers of paged state, all of them addressing:

* the compressed-KV pool row for a block, arithmetic off the block's full loc
  (``full_loc(b * ratio) // ratio``), so it needs no slot map;
* the per-window-page compress-state RING, which carries the rolling reduction across page and
  request boundaries so a radix hit can resume a prefix by value;
* a per-row scratch row, the destination for a decode step whose block did not complete (a
  discarded write that keeps the masked store graph-safe and collision-free).

V4.1's ring differs from DSV4's: ``ring_size == ratio`` (1 or 2) and the item is ``2 *
head_dim`` fp32 with no overlap half. ``P % ratio == 0``, so a token group never straddles a
window page and there is no boundary-carry machinery to port.

All of that is addressing, so it lives here rather than in the model: the module keeps its
weights and the reduction math and passes them in, mirroring sglang's ``CompressorBackendMixin``
(``forward_compress(*, ape, norm, freqs_cis_cache, ...)``).
"""

from __future__ import annotations

import torch


class DSV41CompressorBackendMixin:
    # tier -> (KV-pool, state-ring, scratch-base) ACCESSOR names on the pool. The tuple shape is
    # kept from DSV4 so the two mixins diff cleanly; only the indirection changed.
    _TIER_ATTRS = {
        "attn": ("cmp_pool_of", "state_ring_of", "cmp_scratch_base_of"),
        "idx": ("idx_pool_of", "indexer_state_ring_of", "idx_scratch_base_of"),
    }

    # ----- pool views (band-routed) ------------------------------------------------------
    def compress_pool(self, layer_id: int, tier: str) -> torch.Tensor:
        return getattr(self.pool, self._TIER_ATTRS[tier][0])(layer_id)

    def compress_state_ring(self, layer_id: int, tier: str):
        """The band's compress-state ring for ``tier``.

        Always None for ``"idx"``: V4.1's Indexer has no compressor of its own, so an index
        band owns no ring to route to.
        """
        return getattr(self.pool, self._TIER_ATTRS[tier][1])(layer_id)

    def compress_scratch_base(self, layer_id: int, tier: str) -> int:
        return getattr(self.pool, self._TIER_ATTRS[tier][2])(layer_id)

    # ----- compressed-row addressing -----------------------------------------------------
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
        pool = self.compress_pool(layer_id, tier)
        pool.index_copy_(0, rows, kv.to(pool.dtype))

    # ----- compress-state ring -----------------------------------------------------------
    def state_loc(self, window_slots: torch.Tensor, ring_size: int) -> torch.Tensor:
        """Batch/vector form of ``pool.state_loc``: the ring row holding the window page's
        partial token group, with the ``-1`` gather sentinel preserved."""
        return self.pool.state_loc(window_slots, ring_size, self.pool.P)

    def read_state(
        self, layer_id: int, tier: str, window_slots: torch.Tensor, ring_size: int
    ) -> torch.Tensor:
        """The ``[..., ring_size, 2 * head_dim]`` compress-state block of each window slot's
        PAGE -- kv half then score half. Used to resume a prefix by value on a radix hit."""
        self._assert_state_ring(layer_id, tier)
        return self.pool.get_state(layer_id, self._state_rows(window_slots, ring_size))

    def write_state(
        self, layer_id: int, tier: str, window_slots: torch.Tensor, ring_size: int,
        blocks: torch.Tensor,
    ) -> None:
        """Persist the compress-state block of each window slot's page.

        The model writes exactly one trailing partial group per (row, window page): ``P % ratio
        == 0`` means a group cannot straddle a page, so a page's block is complete on its own
        and no cross-page carry has to be written.
        """
        self._assert_state_ring(layer_id, tier)
        self.pool.set_state(layer_id, self._state_rows(window_slots, ring_size), blocks)

    # ----- internals --------------------------------------------------------------------
    def _assert_state_ring(self, layer_id: int, tier: str) -> None:
        # The pool routes a ring read/write to ``state_ring[layer_id]`` (the attention band),
        # so an "idx" call would silently hit the wrong ring; fail loudly instead.
        assert self.compress_state_ring(layer_id, tier) is not None, (
            f"layer {layer_id} tier {tier!r} owns no compress-state ring "
            f"(V4.1's Indexer has no compressor of its own)"
        )

    def _state_rows(self, window_slots: torch.Tensor, ring_size: int) -> torch.Tensor:
        """``[..., ring_size]`` ring rows of each window slot's PAGE block: the ring is
        page-local, so a negative slot addresses only the ring's scratch row.

        This is the page BASE plus ``arange``, not ``state_loc`` plus ``arange``: ``state_loc``
        picks the one cell a slot's phase owns, and consecutive phases of the same page are
        adjacent cells, so adding an arange to it would overlap neighbouring blocks.
        """
        pages = torch.div(window_slots, self.pool.P, rounding_mode="floor")
        rows = (pages * ring_size)[..., None] + torch.arange(
            ring_size, device=window_slots.device, dtype=pages.dtype
        )
        return torch.where(window_slots[..., None] < 0, torch.full_like(rows, -1), rows)


__all__ = ["DSV41CompressorBackendMixin"]
