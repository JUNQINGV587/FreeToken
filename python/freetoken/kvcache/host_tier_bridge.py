"""Bridge between a paged KV pool and the host tier: one radix node <-> one tier entry.

The radix caches store page indices and nothing else; the host tier stores slabs and knows
nothing about pages. The KV buffers themselves live on the engine (``engine.kv_cache``), not on
the ``CacheManager`` that decides what to evict, so neither side can do the copy alone. This
module is that seam.

Key discipline: a tier key names ONE resident copy. A pool page id is not usable as a key (the
pool hands an id out again as soon as the pages come back), so keys derive from the node's uuid.

Slot ids, not page numbers: ``node.value`` holds one pool SLOT id per token, laid out as
``page * page_size + offset``, so a page's first slot is ``value[::page_size]`` and its page
index is that value divided by the page size. Reading a slot id as a page number silently
addresses another page, and because ids are recycled the mistake stays quiet until the wrong KV
is read.

What a page carries on a QSA pool: the K/V slabs, one compressed index row per ``index_ratio``
tokens per index layer, and (under mrope) the 3-axis rope position of every token. All three are
addressed by the token's own slot, so all three move with the page and are remapped to the fresh
page on the way back -- a missing index or rope bank is invisible in the copied bytes and only
shows up as a different QSA block selection or a shifted rope position, which is why
``check_tier_geometry`` refuses to build a bridge whose banks do not cover the pool's.

The pending ring is deliberately NOT carried. It is per-request scratch (``[num_req_slots, ...]``
indexed by ``table_idx``), holding the un-compressed tail of the forward in flight, and it is
irrelevant to a moved page because: a spilled node is a whole page span, ``page_size %
index_ratio == 0`` is enforced at pool construction, so the span ends on a group boundary and
every token in it already has its compressed row; a matched prefix is never re-forwarded (only
the extension is), so nothing reads the ring for restored tokens; and ``cmp_scratch_base`` rows
past the paged region are per-forward scratch, not per-page state.
"""

from __future__ import annotations

from typing import Callable, Hashable

import torch

from .host_tier import HostKVTier, TierGeometryMismatch, check_tier_geometry


class PoolHostBridge:
    """Copies one radix node's paged KV between the engine pool and the host tier.

    ``alloc_pages(n) -> Tensor`` hands out ``n`` fresh pool page indices for a restore; the
    caller (the cache manager, which owns the free list) injects it, so the tier never guesses
    which pages are free. Construction raises :class:`TierGeometryMismatch` when the tier's banks
    do not describe the pool, which is also the only "unsupported" signal there is.
    """

    def __init__(
        self,
        pool,
        tier: HostKVTier,
        page_size: int,
        *,
        alloc_pages: Callable[[int], torch.Tensor],
        key_prefix: str = "kv",
    ) -> None:
        self.pool = pool
        self.tier = tier
        self.page_size = page_size
        self.alloc_pages = alloc_pages
        self.key_prefix = key_prefix
        check_tier_geometry(tier.geometry, pool, page_size=page_size)
        self.index_layers = tier.geometry.index_layers
        self.rows_per_page = tier.geometry.index_rows_per_page
        self.rope_pos = tier.geometry.rope_pos
        if self.index_layers and not callable(getattr(pool, "cmp_k_cache", None)):
            raise TierGeometryMismatch(
                f"tier geometry index_layers={self.index_layers} but "
                f"{type(pool).__name__} keeps no compressed index slab"
            )
        if self.rope_pos and getattr(pool, "rope_positions", None) is None:
            raise TierGeometryMismatch(
                f"tier geometry rope_pos={self.rope_pos} but {type(pool).__name__} keeps "
                "no per-token rope positions"
            )
        self.num_pages = self._pool_num_pages(pool)
        if self.num_pages <= 0:
            raise TierGeometryMismatch(
                f"cannot read a page count from {type(pool).__name__}: the host tier addresses "
                "pools that keep their K/V in a paged buffer"
            )

    @staticmethod
    def _pool_num_pages(pool) -> int:
        buf = getattr(pool, "_kv_buffer", None)
        if buf is not None and buf.dim() == 6:
            return int(buf.shape[2])
        buf = getattr(pool, "_k_buffer", None)
        if buf is not None and buf.dim() == 5:
            return int(buf.shape[1])
        return 0

    @property
    def supported(self) -> bool:
        """True once constructed: every per-slot slab the pool keeps is carried or refused."""
        return True

    # -- helpers ---------------------------------------------------------------------------

    def _key(self, node) -> str:
        return f"{self.key_prefix}:{node.uuid}"

    def _pages(self, node) -> list[int] | None:
        """Page indices of a node's stored span, or None when it cannot be moved as whole pages.

        None is a refusal, never a partial copy: the caller keeps the pages and evicts as it
        always did.
        """
        if node.length == 0 or node.length % self.page_size:
            return None
        n_pages = node.length // self.page_size
        heads = [int(v) for v in node.value[:: self.page_size]]
        if len(heads) != n_pages or any(v % self.page_size for v in heads):
            return None
        pages = [v // self.page_size for v in heads]
        if len(set(pages)) != n_pages or min(pages) < 0 or max(pages) >= self.num_pages:
            return None
        return pages

    def _index_rows(self, page: int) -> slice:
        r0 = page * self.rows_per_page
        return slice(r0, r0 + self.rows_per_page)

    def _index_slab(self, page: int) -> torch.Tensor:
        rows = self._index_rows(page)
        return torch.stack([self.pool.cmp_k_cache(layer)[rows] for layer in range(self.index_layers)])

    def _scatter_index(self, page: int, slab: torch.Tensor) -> None:
        rows = self._index_rows(page)
        for layer in range(self.index_layers):
            self.pool.cmp_k_cache(layer)[rows].copy_(slab[layer])

    def _rope_rows(self, page: int) -> torch.Tensor:
        return self.pool.rope_positions[page * self.page_size : (page + 1) * self.page_size]

    # -- the cache-side contract -----------------------------------------------------------

    def spill(self, node) -> str | None:
        """Copy ``node``'s pages into the tier and return the key, or None to refuse."""
        pages = self._pages(node)
        if pages is None:
            return None
        key = self._key(node)
        if self.tier.entry_slots(key) is not None:
            # The node is already spilled. Overwriting would leave the first owner reading the
            # second one's bytes, and the cache counts one host-resident span per key.
            return None
        slots = self.tier.alloc_entry(key, len(pages))
        try:
            for i, page in enumerate(pages):
                kv = self.pool.page_kv_view(page)
                self.tier.k_page(slots[i]).copy_(kv[0])
                self.tier.v_page(slots[i]).copy_(kv[1])
                if self.index_layers:
                    self.tier.index_page(slots[i]).copy_(self._index_slab(page))
                if self.rope_pos:
                    self.tier.rope_page(slots[i]).copy_(self._rope_rows(page))
        except BaseException:
            # A half-written entry is one a later restore would trust.
            self.tier.drop(key)
            raise
        return key

    def materialize(self, node, key: Hashable) -> torch.Tensor | None:
        """Copy the entry back into freshly allocated pages, one page index per token."""
        slots = self.tier.entry_slots(key)
        if slots is None:
            return None  # the tier's LRU dropped it; the prefix is gone, recompute
        if node.length % self.page_size:
            self.tier.drop(key)
            return None
        n_pages = node.length // self.page_size
        if len(slots) != n_pages:
            self.tier.drop(key)
            return None
        fresh = self.alloc_pages(n_pages)
        if fresh is None or int(fresh.numel()) != n_pages:
            self.tier.drop(key)
            return None
        pages = [int(p) for p in fresh]
        if len(set(pages)) != n_pages or min(pages) < 0 or max(pages) >= self.num_pages:
            self.tier.drop(key)
            return None
        for i, page in enumerate(pages):
            kv = self.pool.page_kv_view(page)
            kv[0].copy_(self.tier.k_page(slots[i]))
            kv[1].copy_(self.tier.v_page(slots[i]))
            if self.index_layers:
                self._scatter_index(page, self.tier.index_page(slots[i]))
            if self.rope_pos:
                self._rope_rows(page).copy_(self.tier.rope_page(slots[i]))
        self.tier.drop(key)
        # Same layout as node.value: each token names its page by the page's first slot id.
        return (fresh.to(torch.int32) * self.page_size).repeat_interleave(self.page_size)

    def forget(self, key: Hashable) -> bool:
        """Release an entry nobody will restore (the cache freed its node)."""
        return self.tier.drop(key)


__all__ = ["PoolHostBridge"]
