"""Host-side KV page tier: spill and restore whole KV pages in host memory.

The radix prefix caches evict by returning page ids to the pool: the page's K/V is dropped, so
no prefix longer than VRAM allows can ever be reused. This module adds the missing lower tier
behind that call -- a bounded LRU store of evicted pages held in a host bank, written and read
with plain device<->host copies.

Scope of v0 (callbacks are injected into ``HybridRadixCache``; no engine constructs them yet):
- Pages are addressed by a caller-supplied opaque key; the tier never interprets it (a pool
  page id is NOT a safe key: the pool hands the id out again as soon as the pages come back).
- ``spill`` is called BEFORE the pool frees a page, ``restore`` fills a page the caller has
  already allocated. A page the LRU dropped is gone for good -- ``on_drop`` tells the caller so.
- An *entry* groups the pages of one unit (one radix node's KV span) under a single key and is
  evicted as a whole: a node whose pages are half in host memory is not a usable prefix, so the
  LRU never has to reason about partial entries.
- Both directions copy on the caller's current stream and leave the ordering to the caller: a
  spill is ordered before the pages return to the pool, a restore before the page is read. Two
  streams have no ordering between them, so a copy must not move to a side stream without a
  matching ``wait_stream``; ``non_blocking=True`` additionally requires ``pin()``.
- This tier carries K/V, the QSA index shadow and mrope positions, and NOTHING else. It has no
  GDN/PLE state and must not grow one: ``LinearStatePool`` is a GPU-only COW pool, and a hybrid
  prefix is resumable only through a LIVE snapshot on its node. The caller therefore spills only
  nodes that still own their snapshot, keeps that snapshot alive for the whole host residency,
  and resumes a restored prefix FROM the snapshot -- never by recomputing it.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass

import torch

from freetoken.moe.host_banks import HostBank
from freetoken.utils import init_logger, mem_GB

from .base import HostTierKeyCollision

logger = init_logger(__name__)


@dataclass(frozen=True)
class TierGeometry:
    """The slab shapes one page of the pool occupies, taken from the pool's own buffers.

    ``index_layers``/``index_head_dim`` cover QSA's compressed index shadow (one row per
    ``index_ratio`` tokens); ``rope_pos`` covers mrope's per-token 3-axis position.
    """

    num_layers: int
    page_size: int
    num_kv_heads: int
    head_dim: int
    dtype: torch.dtype
    index_layers: int = 0
    index_head_dim: int = 0
    index_ratio: int = 1
    rope_pos: bool = False

    @property
    def kv_page_elems(self) -> int:
        return self.num_layers * self.page_size * self.num_kv_heads * self.head_dim

    @property
    def index_rows_per_page(self) -> int:
        return self.page_size // self.index_ratio if self.index_layers else 0

    @property
    def index_page_elems(self) -> int:
        return self.index_rows_per_page * self.index_layers * self.index_head_dim

    @property
    def rope_page_elems(self) -> int:
        return self.page_size * 3 if self.rope_pos else 0


class HostTierUnpinned(RuntimeError):
    """A non-blocking copy was asked of a bank that is not page-locked.

    torch falls back to a synchronous copy for pageable host memory, so the flag would become a
    silent performance lie instead of an error. ``pin()`` first."""


@dataclass
class TierStats:
    spills: int = 0
    restores: int = 0
    hits: int = 0
    misses: int = 0
    dropped: int = 0
    dropped_bytes: int = 0


class HostKVTier:
    """Bounded LRU store of spilled KV pages in host memory.

    Capacity is a page count; the banks are allocated once and reused, so a spill never
    allocates. ``restore`` returns False for a page that was never spilled or was already
    dropped, which the caller reports as a cold prefix (recompute) rather than an error.
    """

    def __init__(
        self,
        geometry: TierGeometry,
        num_pages: int,
        *,
        backing: str = "mmap",
        on_drop: Callable[[Hashable], None] | None = None,
    ) -> None:
        assert num_pages > 0, "a host tier needs at least one page of capacity"
        if geometry.index_layers:
            # the shadow slab rides the compute dtype, same 2-byte constraint as QSAKVCache
            assert geometry.dtype.itemsize == 2, "the QSA index shadow is a 2-byte slab"
        self.geometry = geometry
        self.num_pages = num_pages
        self.on_drop = on_drop
        self.stats = TierStats()

        shape = (num_pages, geometry.num_layers, geometry.page_size,
                 geometry.num_kv_heads, geometry.head_dim)
        self._k = HostBank(shape, geometry.dtype, backing=backing)
        self._v = HostBank(shape, geometry.dtype, backing=backing)
        self._index = (
            HostBank(
                (num_pages, geometry.index_layers, geometry.index_rows_per_page,
                 geometry.index_head_dim),
                geometry.dtype,
                backing=backing,
            )
            if geometry.index_layers
            else None
        )
        self._rope = (
            HostBank((num_pages, geometry.page_size, 3), torch.int32, backing=backing)
            if geometry.rope_pos
            else None
        )

        self._slots: dict[Hashable, list[int]] = {}  # key -> its page slots, in key order
        self._free: list[int] = list(range(num_pages - 1, -1, -1))
        self._lru: OrderedDict[Hashable, None] = OrderedDict()

    # -- sizing ---------------------------------------------------------------------------

    @property
    def bytes_per_page(self) -> int:
        g = self.geometry
        kv = 2 * g.kv_page_elems * g.dtype.itemsize
        return kv + g.index_page_elems * 2 + g.rope_page_elems * 4

    @property
    def capacity_bytes(self) -> int:
        return self.bytes_per_page * self.num_pages

    @property
    def resident_pages(self) -> int:
        return sum(len(slots) for slots in self._slots.values())

    @property
    def resident_bytes(self) -> int:
        return self.bytes_per_page * self.resident_pages

    @property
    def resident_entries(self) -> int:
        return len(self._slots)

    def __contains__(self, key: Hashable) -> bool:
        return key in self._slots

    def pin(self) -> None:
        """Page-lock the banks so device copies can be non-blocking (needs CUDA)."""
        for bank in self._banks():
            bank.pin()

    def _banks(self) -> Sequence[HostBank]:
        banks = [self._k, self._v]
        if self._index is not None:
            banks.append(self._index)
        if self._rope is not None:
            banks.append(self._rope)
        return banks

    # -- entries --------------------------------------------------------------------------

    def alloc_entry(self, key: Hashable, num_pages: int) -> list[int]:
        """Reserve ``num_pages`` slots for one entry (one radix node's KV span) and return them.

        The entry is visible to :meth:`entry_slots` as soon as it is allocated, so the caller
        must fill every slot through :meth:`k_page`/:meth:`v_page` and call :meth:`drop` if it
        cannot -- a half-written entry is one a later restore would trust.

        Making room happens here: victims are reported to ``on_drop`` as whole entries.
        """
        return self._alloc(key, num_pages)

    def entry_slots(self, key: Hashable) -> list[int] | None:
        return self._slots.get(key)

    def k_page(self, slot: int) -> torch.Tensor:
        """[num_layers, page_size, num_kv_heads, head_dim] view of one stored page.

        The bridge fills and drains whole pages through these: one page of the pool is
        ``[2, num_layers, page_size, num_kv_heads, head_dim]``, so going layer by layer would
        cost 2*num_layers copies per page instead of two.
        """
        return self._k.tensor[slot]

    def v_page(self, slot: int) -> torch.Tensor:
        return self._v.tensor[slot]

    def index_page(self, slot: int) -> torch.Tensor | None:
        return None if self._index is None else self._index.tensor[slot]

    def rope_page(self, slot: int) -> torch.Tensor | None:
        return None if self._rope is None else self._rope.tensor[slot]

    # -- movement -------------------------------------------------------------------------

    def spill(
        self,
        key: Hashable,
        k_slab: torch.Tensor,
        v_slab: torch.Tensor,
        *,
        index_slab: torch.Tensor | None = None,
        rope_slab: torch.Tensor | None = None,
        non_blocking: bool = False,
    ) -> None:
        """Copy one page's slabs into the tier. Must be called before the pool frees the page.

        Slabs are the pool's own page views: ``k_slab``/``v_slab`` are ``[num_layers,
        page_size, num_kv_heads, head_dim]`` and may be strided (``pool._k_buffer[:, page]``).

        ``key`` names one resident copy, so a key that is already resident is an error rather
        than an overwrite: the pool recycles page ids, and a silent overwrite would leave the
        first owner reading the second owner's bytes. Callers key by something that cannot be
        reused while the copy is resident.
        """
        self._check_kv(k_slab, v_slab)
        self._check_async(non_blocking)
        if self._index is not None:
            assert index_slab is not None, "geometry carries a QSA index shadow; pass index_slab"
            self._check_index(index_slab)
        if self._rope is not None:
            assert rope_slab is not None, "geometry carries rope positions; pass rope_slab"

        slot = self._alloc(key, 1)[0]
        self._k.tensor[slot].copy_(k_slab, non_blocking=non_blocking)
        self._v.tensor[slot].copy_(v_slab, non_blocking=non_blocking)
        if self._index is not None:
            self._index.tensor[slot].copy_(index_slab, non_blocking=non_blocking)
        if self._rope is not None:
            self._rope.tensor[slot].copy_(rope_slab, non_blocking=non_blocking)
        self.stats.spills += 1

    def restore(
        self,
        key: Hashable,
        k_slab: torch.Tensor,
        v_slab: torch.Tensor,
        *,
        index_slab: torch.Tensor | None = None,
        rope_slab: torch.Tensor | None = None,
        non_blocking: bool = False,
    ) -> bool:
        """Copy a spilled page back into the caller's freshly allocated page views.

        Returns False when the page is not resident (never spilled, or dropped by the LRU);
        the buffers are left untouched in that case.
        """
        self._check_kv(k_slab, v_slab)
        self._check_async(non_blocking)
        slots = self._slots.get(key)
        if slots is None:
            self.stats.misses += 1
            return False
        assert len(slots) == 1, "restore() takes whole-page slabs; a multi-page entry has none"
        slot = slots[0]
        if self._index is not None:
            assert index_slab is not None, "geometry carries a QSA index shadow; pass index_slab"
            self._check_index(index_slab)
        if self._rope is not None:
            assert rope_slab is not None, "geometry carries rope positions; pass rope_slab"

        k_slab.copy_(self._k.tensor[slot], non_blocking=non_blocking)
        v_slab.copy_(self._v.tensor[slot], non_blocking=non_blocking)
        if self._index is not None:
            index_slab.copy_(self._index.tensor[slot], non_blocking=non_blocking)
        if self._rope is not None:
            rope_slab.copy_(self._rope.tensor[slot], non_blocking=non_blocking)
        self._lru.move_to_end(key)
        self.stats.restores += 1
        self.stats.hits += 1
        return True

    def drop(self, key: Hashable) -> bool:
        """Release an entry's slots without restoring it (the caller freed the prefix)."""
        slots = self._slots.pop(key, None)
        if slots is None:
            return False
        self._lru.pop(key, None)
        self._free.extend(slots)
        return True

    def clear(self) -> None:
        """Drop every resident page, reporting each to ``on_drop``.

        The callback is what unlinks the cache's host-resident node; a silent clear would leave
        nodes pointing at a tier that no longer holds their bytes.
        """
        for key in self._slots:
            if self.on_drop is not None:
                self.on_drop(key)
        self._slots.clear()
        self._lru.clear()
        self._free = list(range(self.num_pages - 1, -1, -1))

    # -- internals ------------------------------------------------------------------------

    def _alloc(self, key: Hashable, num_pages: int) -> list[int]:
        if key in self._slots:
            raise HostTierKeyCollision(
                f"host tier key {key!r} is already resident (page ids are recycled by the "
                "pool, so key by a value unique per resident copy)"
            )
        assert 0 < num_pages <= self.num_pages, (
            f"{num_pages} pages for a {self.num_pages}-page tier"
        )
        while len(self._free) < num_pages:
            self._evict_lru()
        slots = [self._free.pop() for _ in range(num_pages)]
        self._slots[key] = slots
        self._lru[key] = None
        return slots

    def _evict_lru(self) -> None:
        """Free one whole entry for room.

        The victim is out of ``_slots`` and its slots are back on the free list BEFORE
        ``on_drop`` runs, so a callback that calls back in (drop, or a re-entrant spill) sees a
        consistent tier instead of a half-freed one.
        """
        victim, _ = self._lru.popitem(last=False)
        slots = self._slots.pop(victim)
        self._free.extend(slots)
        self.stats.dropped += len(slots)
        self.stats.dropped_bytes += self.bytes_per_page * len(slots)
        if self.on_drop is not None:
            self.on_drop(victim)
        logger.debug(
            "host KV tier full (%s): dropped entry %s (%d pages)",
            mem_GB(self.capacity_bytes), victim, len(slots),
        )

    def _check_async(self, non_blocking: bool) -> None:
        """A non-blocking copy is only real on page-locked banks (see HostTierUnpinned)."""
        if not non_blocking:
            return
        unpinned = [i for i, bank in enumerate(self._banks()) if not bank.tensor.is_pinned()]
        if unpinned:
            raise HostTierUnpinned(
                f"non_blocking=True needs page-locked host banks (unpinned banks: {unpinned}); "
                "call pin() first -- torch copies pageable memory synchronously without saying so"
            )

    def _check_kv(self, k_slab: torch.Tensor, v_slab: torch.Tensor) -> None:
        g = self.geometry
        want = (g.num_layers, g.page_size, g.num_kv_heads, g.head_dim)
        for name, slab in (("k", k_slab), ("v", v_slab)):
            assert tuple(slab.shape) == want, f"{name}_slab shape {tuple(slab.shape)} != {want}"
            assert slab.dtype == g.dtype, f"{name}_slab dtype {slab.dtype} != {g.dtype}"
        assert k_slab.device == v_slab.device, "k_slab and v_slab live on different devices"

    def _check_index(self, index_slab: torch.Tensor) -> None:
        g = self.geometry
        want = (g.index_layers, g.index_rows_per_page, g.index_head_dim)
        assert tuple(index_slab.shape) == want, (
            f"index_slab shape {tuple(index_slab.shape)} != {want}"
        )


__all__ = ["HostKVTier", "HostTierUnpinned", "TierGeometry", "TierStats"]
