"""Host-side KV page tier: spill and restore whole KV pages in host memory.

The radix prefix caches evict by returning page ids to the pool: the page's K/V is dropped, so
no prefix longer than VRAM allows can ever be reused. This module adds the missing lower tier
behind that call -- a bounded LRU store of evicted pages held in a host bank, written and read
with plain device<->host copies.

Scope of v0 (deliberately not wired into the engine yet):
- Pages are addressed by the pool's own page id; the tier keeps no radix state and no free list.
- ``spill`` is called BEFORE the pool frees a page, ``restore`` fills a page the caller has
  already allocated. A page the LRU dropped is gone for good -- ``on_drop`` tells the caller so.
- GDN state has no host format (``LinearStatePool`` snapshots are a GPU-only COW pool), so a
  restored prefix is attention-only as far as this tier is concerned: the caller treats a
  restored prefix as cold for the recurrent state and recomputes it.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from freetoken.moe.host_banks import HostBank
from freetoken.utils import init_logger, mem_GB

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
        on_drop: Callable[[int], None] | None = None,
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

        self._slots: dict[int, int] = {}  # page id -> slot
        self._free: list[int] = list(range(num_pages - 1, -1, -1))
        self._lru: OrderedDict[int, None] = OrderedDict()

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
        return len(self._slots)

    @property
    def resident_bytes(self) -> int:
        return self.bytes_per_page * len(self._slots)

    def __contains__(self, page_id: int) -> bool:
        return page_id in self._slots

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

    # -- movement -------------------------------------------------------------------------

    def spill(
        self,
        page_id: int,
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
        """
        self._check_kv(k_slab, v_slab)
        if self._index is not None:
            assert index_slab is not None, "geometry carries a QSA index shadow; pass index_slab"
            self._check_index(index_slab)
        if self._rope is not None:
            assert rope_slab is not None, "geometry carries rope positions; pass rope_slab"

        slot = self._slots.get(page_id)
        if slot is None:
            slot = self._take_slot(page_id)
            self._slots[page_id] = slot
        else:
            self._lru.move_to_end(page_id)

        self._k.tensor[slot].copy_(k_slab, non_blocking=non_blocking)
        self._v.tensor[slot].copy_(v_slab, non_blocking=non_blocking)
        if self._index is not None:
            self._index.tensor[slot].copy_(index_slab, non_blocking=non_blocking)
        if self._rope is not None:
            self._rope.tensor[slot].copy_(rope_slab, non_blocking=non_blocking)
        self.stats.spills += 1

    def restore(
        self,
        page_id: int,
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
        slot = self._slots.get(page_id)
        if slot is None:
            self.stats.misses += 1
            return False
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
        self._lru.move_to_end(page_id)
        self.stats.restores += 1
        self.stats.hits += 1
        return True

    def drop(self, page_id: int) -> bool:
        """Release a page's slot without restoring it (the caller freed the prefix)."""
        slot = self._slots.pop(page_id, None)
        if slot is None:
            return False
        self._lru.pop(page_id, None)
        self._free.append(slot)
        return True

    def clear(self) -> None:
        self._slots.clear()
        self._lru.clear()
        self._free = list(range(self.num_pages - 1, -1, -1))

    # -- internals ------------------------------------------------------------------------

    def _take_slot(self, page_id: int) -> int:
        if self._free:
            slot = self._free.pop()
        else:
            victim, slot = self._lru.popitem(last=False)
            del self._slots[victim]
            self.stats.dropped += 1
            self.stats.dropped_bytes += self.bytes_per_page
            if self.on_drop is not None:
                self.on_drop(victim)
            logger.debug(
                "host KV tier full (%s): dropped page %s", mem_GB(self.capacity_bytes), victim
            )
        self._lru[page_id] = None
        return slot

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


__all__ = ["HostKVTier", "TierGeometry", "TierStats"]
