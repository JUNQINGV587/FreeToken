"""DSV4.1 paged KV pools (sglang-style management, FreeToken byte layout).

V4.1 (CSA2) is *shared-cache* sparse attention: one compressed KV per BAND of backbone
layers, plus a Lightning-Indexer key tier on the (larger) index-source set, plus a
per-layer 128-sliding window. The tiers:

* ``window_pool[L]``   -- all ``n_layers``; the 128-sliding KV ring, page-granular.
* ``cmp_pool[S]``      -- one per KV SOURCE layer, shared by S's whole band.
* ``idx_pool[I]``      -- one per INDEX source layer, shared by I's whole index band.
* ``state_ring[S]``    -- one per KV SOURCE layer: the per-window-page compress-state
  ring (fp32, ``kv|score`` split), index ``-1`` a permanent scratch slot.

Sharing is by IDENTITY: ``pool.cmp_pool[25] is pool.cmp_pool[20]`` (and
``cmp_source_of(25) == 20``), so two consumers of one band provably write and read the
same tensor. A tier list is None on every layer that owns no such tier -- a ratio-0
layer has no compressed tier, and an index source that is not a KV source
(24/28/32/36) owns an indexer tier but no compressed tier of its own: its indexer rows
come from its band's compressed rows (`full_token // band_ratio`, the same row count).

Ring/state policy (the one deliberate divergence from DSV4's ratio-4 machinery):
``models/deepseek_v41/compress.py`` shows the V4.1 compressor carries at most one
INCOMPLETE trailing group, in ``_kv_state``/``_score_state`` indexed by ``pos % ratio``
(ratio 2 = gated pooling of token pairs; ratio 1 = a plain projection with no state at
all). There are no overlapping carry blocks, so ``coff == 1`` and the ring needs exactly
``ratio`` slots per window page. Because ``P % ratio == 0``, a group never straddles a
window page: at every page boundary the pending partial is empty, so a page-aligned
radix resume never needs a carry-by-value, and DSV4's boundary-carry machinery
(``write_boundary_carries``, overlap halves in the ring row) has no analogue here.

``indexer_state_ring`` is kept as part of the plugin surface but is None everywhere:
V4.1's Indexer has no compressor of its own (``models/deepseek_v41/indexer.py`` builds
``wk`` only on ``owns_k`` layers and consumes the *attention* compressor's latent), so
there is no second ring to allocate -- the band's ``state_ring`` already is the indexer's
compressor state. DSV4 keeps a distinct indexer ring because its indexer compresses with
its own overlap ring.

The KV/compressed/indexer pools are bf16 (the fp8/fp4 quant is an in-place round-trip
already baked into the bf16 value, so ``index_select`` staging is byte-exact); only the
compress-state ring is fp32.

MTP is out of scope: ``compress_ratios`` ships 43 entries for 40 backbone layers + 3
MTP/DSpark layers, and the pool covers ``[:args.n_layers]`` (the MTP layers never own a
paged tier here).

``state_loc`` is DERIVED from a window slot, never stored:
    state_loc = where(ws < 0, -1, (ws // P) * ring_size + ws % ring_size)
``ring_size | P`` so distinct pages map to disjoint ring blocks.
"""

from __future__ import annotations

import torch

from freetoken.utils import init_logger

from .base import BaseKVCachePool
from .dsv4_paged_pool import CompressStateRing, FreeListAllocator
from .dsv41_cost_model import (
    DSV41PoolSizes,
    dsv41_cmp_source_of,
    dsv41_index_source_of,
    dsv41_kv_sources,
    dsv41_kv_unit_bytes,
    dsv41_ring_size_for_ratio,
    dsv41_window_unit_bytes,
)


logger = init_logger(__name__)


class DSV41PagedKVCache(BaseKVCachePool):
    def __init__(
        self,
        sizes: DSV41PoolSizes,
        args,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
        P: int = 128,
        n_scratch: int = 1,
    ) -> None:
        assert dtype == torch.bfloat16, "KV pools are bf16 (fp4/fp8 is an in-place round-trip)"
        self.args = args
        self.sizes = sizes
        self._device = device
        self._dtype = dtype
        self.P = P
        self._n_layers = args.n_layers
        self.head_dim = args.head_dim
        self.index_head_dim = args.index_head_dim
        # The checkpoint ships 43 ratios for 40 backbone layers + 3 MTP; the pool covers the
        # backbone only (MTP layers own no paged tier -- see the module docstring).
        self.compress_ratios = tuple(args.compress_ratios)[: self._n_layers]
        assert len(self.compress_ratios) == self._n_layers
        # Owners. The compressed, state AND index-K tiers are all owned by the KV source, so
        # one list drives every allocation below; ``_idx_source`` is only the PUBLICATION
        # axis (which indexer publishes topk/candidates), not a tier owner.
        self._cmp_owners = dsv41_kv_sources(args)
        self._cmp_source = [dsv41_cmp_source_of(args, L) for L in range(self._n_layers)]
        self._idx_source = [dsv41_index_source_of(args, L) for L in range(self._n_layers)]
        # Scratch rows appended to each cmp/idx pool tensor BEYOND the allocator's capacity
        # (never handed out): batched decode routes each row whose compressed block did NOT
        # complete this step to its own scratch row ``cmp_scratch_base + row`` (a discarded
        # write), so the masked per-row scatter is graph-safe (no host sync, no -1 index, no
        # cross-row collision). One per running request row.
        self.n_scratch = int(n_scratch)

        for ratio in set(self.compress_ratios):
            assert ratio == 0 or P % ratio == 0, f"P={P} must be divisible by ratio {ratio}"
        self._paged_params: tuple[int, bool] | None = None  # (_init_paged_state args, for rebuild)
        self._alloc_buffers()

    # ----- shared-band routing (the pool's ONE source of tier ownership) -----
    def cmp_source_of(self, layer_id: int) -> int | None:
        """KV-source layer owning this layer's compressed/indexer/state tiers (None for
        window-only layers)."""
        return self._cmp_source[layer_id]

    def idx_source_of(self, layer_id: int) -> int | None:
        """Index-source layer whose indexer *publishes* the topk/candidate lists this layer
        reads (most recent at or before it), or None outside every index source's band.

        This is the PUBLICATION axis only. It does NOT own a key tier: the index-K cache
        belongs to the band's KV source (only a kv-source indexer has ``wk``), which is what
        :meth:`idx_pool_of` resolves. A caller that wants the key-cache owner wants
        :meth:`cmp_source_of`.
        """
        return self._idx_source[layer_id]

    def state_source_of(self, layer_id: int) -> int | None:
        """Compress-state owner -- the band's KV source, which is where the one Compressor
        of the band lives."""
        return self._cmp_source[layer_id]

    def cmp_ratio_of(self, layer_id: int) -> int:
        """The band's compressor ratio (rows are ``full_loc // ratio``); 0 = no compressed
        tier for this layer, so consumers gate on it."""
        source = self._cmp_source[layer_id]
        return 0 if source is None else self.compress_ratios[source]

    def cmp_pool_of(self, layer_id: int) -> torch.Tensor | None:
        return self.cmp_pool[layer_id]

    def idx_pool_of(self, layer_id: int) -> torch.Tensor | None:
        """The band's index-K cache -- the KV source's tensor, shared by every layer of the
        band (identity: ``idx_pool_of(25) is idx_pool_of(20)``). Index sources that own no
        ``wk`` (24/28/32/36) read this one; see :meth:`idx_source_of` for why the routing
        follows the KV source and not the nearest index source."""
        return self.idx_pool[layer_id]

    def state_ring_of(self, layer_id: int) -> CompressStateRing | None:
        return self.state_ring[layer_id]

    def indexer_state_ring_of(self, layer_id: int) -> CompressStateRing | None:
        return self.indexer_state_ring[layer_id]

    def cmp_scratch_base_of(self, layer_id: int) -> int | None:
        return self.cmp_scratch_base[layer_id]

    def idx_scratch_base_of(self, layer_id: int) -> int | None:
        return self.idx_scratch_base[layer_id]

    def _alloc_buffers(self) -> None:
        """(Re)allocate every physical buffer for the CURRENT ``self.sizes``. Shared by
        __init__ and the in-place ``rebuild`` (identity-preserving -- the
        CacheManager/engine/ctx all hold THIS object).

        One buffer per OWNER, then the SAME object is placed at every layer of its band, so
        band sharing is identity rather than a lookup the backend could bypass.
        """
        sizes, device, dtype = self.sizes, self._device, self._dtype
        self.cmp_scratch_base: list[int | None] = [None] * self._n_layers
        self.idx_scratch_base: list[int | None] = [None] * self._n_layers
        self.full_to_window = torch.full(
            (sizes.full_token + 1,), -1, dtype=torch.int64, device=device
        )

        # The ONE slot map (table_idx, pos) -> full loc: the shared page_table, attached by the
        # engine policy. Window slots come from ``full_to_window``; cmp/idx rows are arithmetic.
        if not hasattr(self, "full_loc_map"):
            self.full_loc_map: torch.Tensor | None = None

        # Window KV: every layer, one ring each.
        self.window_pool: list[torch.Tensor] = [
            torch.zeros(sizes.n_win_slots, self.head_dim, device=device, dtype=dtype)
            for _ in range(self._n_layers)
        ]

        self.cmp_pool: list[torch.Tensor | None] = [None] * self._n_layers
        self.idx_pool: list[torch.Tensor | None] = [None] * self._n_layers
        self.state_ring: list[CompressStateRing | None] = [None] * self._n_layers
        # No V4.1 analogue of DSV4's indexer compressor; kept for the plugin surface, always
        # None (see the module docstring).
        self.indexer_state_ring: list[CompressStateRing | None] = [None] * self._n_layers

        for src in self._cmp_owners:
            blocks = sizes.cmp_blocks[src]
            slots = sizes.state_slots[src]
            assert blocks is not None and slots is not None, (
                f"kv source {src} owns no compressed/state tier in sizes"
            )
            pool = torch.zeros(blocks + self.n_scratch, self.head_dim, device=device, dtype=dtype)
            ring = CompressStateRing(
                n_slots=slots,
                ring_size=sizes.ring_sizes[src],
                overlap=False,  # coff == 1: plain token groups, no carry blocks
                head_dim=self.head_dim,
                device=device,
            )
            for L in range(self._n_layers):
                if self._cmp_source[L] == src:
                    self.cmp_pool[L] = pool
                    self.state_ring[L] = ring
                    self.cmp_scratch_base[L] = blocks

        for src in self._cmp_owners:
            blocks = sizes.idx_blocks[src]
            assert blocks is not None, f"kv source {src} owns no indexer tier in sizes"
            pool = torch.zeros(
                blocks + self.n_scratch, self.index_head_dim, device=device, dtype=dtype
            )
            for L in range(self._n_layers):
                if self._cmp_source[L] == src:
                    self.idx_pool[L] = pool
                    self.idx_scratch_base[L] = blocks

    # ----- full-loc translation (gather-only -1 safety; see field comment) -----
    def translate_full_to_window(self, full_locs: torch.Tensor) -> torch.Tensor:
        # int64 gather indices: the shared page_table stores full locs as int32.
        return self.full_to_window[full_locs.to(dtype=torch.int64)]

    @staticmethod
    def cmp_rows(full_locs: torch.Tensor, ratio: int) -> torch.Tensor:
        return torch.div(full_locs.to(dtype=torch.int64), ratio, rounding_mode="floor")

    def bind_window_pages(self, full_page_base: int, window_page_base: int) -> None:
        assert full_page_base % self.P == 0 and window_page_base % self.P == 0
        self.full_to_window[full_page_base : full_page_base + self.P] = torch.arange(
            window_page_base, window_page_base + self.P, dtype=torch.int64, device=self._device
        )

    def unbind_window_pages(self, full_locs: torch.Tensor) -> None:
        self.full_to_window[full_locs[full_locs >= 0]] = -1

    # ----- generic swa_pool duck-type (the CacheManager plug-in surface) -----
    # ShadowRadix layering: the shared page_table is the virtual full-token coordinate; this pool
    # projects it into the physical tiers. The window tier is the managed "second currency" --
    # token-face signatures (what the generic CacheManager speaks), PAGE-ATOMIC internals (window
    # pages are 1:1 page-bound to full pages; the per-page state ring requires it). alloc_swa
    # receives whole ascending pages (allocate_paged's _page_to_token expansion) and free paths
    # are page-complete by construction (padded finish tails, align_down frontiers, page-aligned
    # tree nodes) -- asserted here, not assumed.
    swa_paged = True

    @property
    def sliding_window_size(self) -> int:
        return self.P

    @property
    def prefill_chunk_budget(self) -> int:
        return self._chunk_budget

    # ----- engine-facing rebuild surface -----
    # The tier buffers are bound into per-forward model scratch, invalid after a realloc.
    needs_rebind_on_rebuild = True

    @classmethod
    def kv_cost(cls, config) -> tuple[int, int, int, int]:
        from .dsv41_cost_model import _dsv41_swa_ratio, _dsv41_window_floor_pages
        from .dsv41_cost_model import dsv41_auto_cost_model

        dsv41_args = config.model_config.dsv41_args
        P = dsv41_args.window_size
        floor = _dsv41_window_floor_pages(config, P)
        per_page, fixed, min_reserve_tokens = dsv41_auto_cost_model(
            dsv41_args, _dsv41_swa_ratio(config), floor, P=P, n_scratch=config.max_running_req + 1
        )
        return per_page, fixed, config.page_size, min_reserve_tokens

    @classmethod
    def solve_num_pages(cls, config, available_memory: int) -> int:
        # Solve the largest budget-respecting anchor with the exact per-tier byte model.
        # num_pages is in P (window) units and anchors full_token = num_pages*P (the SHARED
        # cmp/idx tiers). The window working-set floor is honored in PAGES and the total is
        # byte-checked here. A budget too small for even the minimal working set raises a
        # graceful config error, not a late OOM.
        from freetoken.utils import mem_GB

        from .dsv41_cost_model import _dsv41_pool_sizes, _dsv41_swa_ratio, _dsv41_window_floor_pages
        from .dsv41_cost_model import dsv41_pool_bytes, dsv41_solve_num_pages

        dsv41_args = config.model_config.dsv41_args
        P = dsv41_args.window_size
        num_pages = config.num_page_override
        if num_pages is None:
            sizes = dsv41_solve_num_pages(
                available_memory, dsv41_args, _dsv41_swa_ratio(config),
                floor_win_pages=_dsv41_window_floor_pages(config, P), P=P,
                n_scratch=config.max_running_req + 1,
            )
            # The solver fits PHYSICAL pages to memory; one is the dummy page, so the
            # usable (advertised) count is one less.
            num_pages = sizes.full_token // P - 1
        else:
            # Fail at config time with guidance: a below-floor pool would otherwise boot
            # (dsv41_pool_sizes caps the window at num_pages) and die at runtime alloc.
            floor = _dsv41_window_floor_pages(config, P)
            if num_pages < floor:
                raise ValueError(
                    f"--num-pages {num_pages} ({num_pages * P} tokens) is below the DSV4.1 "
                    f"window working-set floor {floor} pages ({floor * P} tokens); raise "
                    f"--num-pages or lower max_running_req/max_seq_len"
                )
            sizes = _dsv41_pool_sizes(config, num_pages + 1)  # +1 for dummy page
        assert num_pages > 1, "Not enough memory for KV cache, try reducing --num-pages"
        real = dsv41_pool_bytes(sizes, dsv41_args, config.max_running_req + 1)
        logger.info(
            f"Allocating {num_pages * P} tokens for DSV4.1 KV cache "
            f"({sizes.n_win_pages} window pages, {len(dsv41_kv_sources(dsv41_args))} shared "
            f"bands), total = {mem_GB(real)}"
        )
        return num_pages

    @classmethod
    def min_kv_tokens(cls, config) -> int:
        # The full anchor must cover the window working-set floor (full >= window always), so
        # that floor -- the value validate_rebuild enforces -- is the pool's floor in tokens.
        from .dsv41_cost_model import _dsv41_window_floor_pages

        P = config.model_config.dsv41_args.window_size
        return _dsv41_window_floor_pages(config, P) * P

    def validate_rebuild(
        self, config, *, num_pages: int | None, target_moe: int, per_expert_bytes: int,
        baseline_free: int, weights_bytes: int, current_num_pages: int,
        extra_fixed_bytes: int = 0, extra_note: str = "",
        num_swa_pages: int | None = None, **targets,
    ) -> None:
        from freetoken.engine.cache_budget import net_cache_budget_bytes
        from freetoken.utils import mem_GB

        from .base import CacheRebuildRejected
        from .dsv41_cost_model import _dsv41_pool_sizes, _dsv41_window_floor_pages
        from .dsv41_cost_model import dsv41_pool_bytes

        dsv41_args = config.model_config.dsv41_args
        if num_pages is not None:
            floor = _dsv41_window_floor_pages(config, dsv41_args.window_size)
            if num_pages < floor:
                raise CacheRebuildRejected(
                    f"num_pages {num_pages} is below the DSV4.1 window working-set floor {floor} "
                    f"(max_running_req={config.max_running_req}); admission would deadlock"
                )
        if num_pages is not None or num_swa_pages is not None:
            # Size the pool a KV/window rebuild would build: the target anchor (or current) with
            # the target window (or current), computed BEFORE the config is mutated.
            target_pages = num_pages if num_pages is not None else current_num_pages
            kv_sizes = _dsv41_pool_sizes(
                config, target_pages + 1, num_swa_pages=num_swa_pages
            )  # +1 for dummy page
        else:
            # MoE-only rebuild keeps the CURRENT pool: budget-check against its live sizes
            # (reflects DSV41_FORCE_SMALL_POOL and the physical dummy page).
            kv_sizes = self.sizes
        # The rebuilds are free-before-alloc, so the whole budget is available (no fixed
        # cache term); an unfit request must still reject BEFORE the teardown.
        budget = net_cache_budget_bytes(config.memory_ratio, baseline_free, weights_bytes, 0)
        need = target_moe * per_expert_bytes + dsv41_pool_bytes(
            kv_sizes, dsv41_args, config.max_running_req + 1
        )
        if need > budget:
            kv_part = f"kv={num_pages} P-pages" if num_pages is not None else "kv=current pool"
            raise CacheRebuildRejected(
                f"requested cache (moe={target_moe} slots, {kv_part}) needs "
                f"{mem_GB(need)} > budget {mem_GB(budget)}; old cache kept, still serving"
            )

    def rebuild_from_config(
        self, config, num_pages: int, *, num_swa_pages: int | None = None
    ) -> None:
        from .dsv41_cost_model import _dsv41_pool_sizes

        # +1 for the dummy page
        self.rebuild(_dsv41_pool_sizes(config, num_pages + 1, num_swa_pages=num_swa_pages))

    def attach_page_table(self, page_table: torch.Tensor) -> None:
        # The model reads full locs through full_loc_map; under the shared route that IS the
        # page table. The attention backend re-allocates its decode snapshot on re-capture.
        self.full_loc_map = page_table

    def unit_bytes(self) -> tuple[int, int]:
        # No measurable flat buffer (owned paged pool): the full (shared cmp/idx + mapping) and
        # window (sliding KV + state rings) per-token costs come from the per-tier cost model.
        return dsv41_kv_unit_bytes(self.args, self.P), dsv41_window_unit_bytes(self.args, self.P)

    def rebuild(self, sizes) -> None:
        """In-place resize to ``sizes`` (identity-preserving; free-before-alloc). The manager's
        tree/page bookkeeping reset is the scheduler's generic cache_manager.rebuild; the engine
        re-attaches the page table via attach_page_table afterwards."""
        import gc

        assert self._paged_params is not None, "rebuild before _init_paged_state"
        self.sizes = sizes
        self.window_pool = self.cmp_pool = self.idx_pool = None
        self.state_ring = self.indexer_state_ring = None
        self.full_to_window = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._alloc_buffers()
        self._init_paged_state(*self._paged_params)

    def _init_paged_state(self, max_running_req: int, radix: bool) -> None:
        """Build the pool-owned window free-list + the tail dummy binding. The LAST full page and
        LAST window page are the reserved dummy region: page_table's dummy row points at
        ``full_token - P`` (== the generic ``fill_(num_tokens)`` convention with num_tokens = the
        allocatable token count), permanently bound so graph-padded rows scatter to a real slot."""
        from .dsv41_cost_model import dsv41_reserved_window_pages
        P = self.P
        self._paged_params = (int(max_running_req), bool(radix))
        self.full_to_window.fill_(-1)
        self._win_alloc = FreeListAllocator(self.sizes.n_win_slots - P, self._device, page_unit=P)
        self.bind_window_pages(self.sizes.full_token - P, self.sizes.n_win_slots - P)
        # Chunk cap: a batched prefill holds the whole chunk's window live at once (sliding frees
        # only between chunks; peak ~2x the chunk), so reserve the concurrent working set and
        # halve the rest -- the same formula the bespoke manager used.
        n_win_pages = (self.sizes.n_win_slots // P) - 1
        reserved = dsv41_reserved_window_pages(max_running_req, radix)
        self._chunk_budget = max(P, (n_win_pages - reserved) // 2 * P)

    @property
    def swa_num_tokens(self) -> int:
        # Allocatable window slots + 1: the generic capacity convention reserves slot 0 as a
        # sentinel (cap == swa_num_tokens - 1); DSV4.1's reserved unit is the tail dummy page,
        # already excluded from the free-list, so +1 re-encodes the same cap.
        return (self.sizes.n_win_slots - self.P) + 1

    def swa_available_size(self) -> int:
        return int(self._win_alloc.available())

    def alloc_swa(self, full_indices: torch.Tensor) -> None:
        """Bind one window page per incoming FULL page. ``full_indices`` must be whole ascending
        pages (the ``_page_to_token`` expansion); the in-page offsets are preserved
        (``window_slot = wbase + pos % P``), which the state ring's page-block layout requires."""
        n = int(full_indices.numel())
        if n == 0:
            return
        P = self.P
        assert n % P == 0, f"alloc_swa needs whole pages, got {n} slots"
        fi = full_indices.to(device=self._device, dtype=torch.int64).view(-1, P)
        fbases = fi[:, 0]
        assert torch.equal(fi, fbases[:, None] + torch.arange(P, device=self._device)), (
            "alloc_swa pages must be contiguous ascending"
        )
        wbases = self._win_alloc.alloc(fbases.numel())  # raises when exhausted (caller gated)
        offsets = torch.arange(P, dtype=torch.int64, device=self._device)
        self.full_to_window[(fbases[:, None] + offsets).flatten()] = (
            wbases[:, None] + offsets
        ).flatten()

    def free_swa(self, full_indices: torch.Tensor) -> None:
        """Return the window pages backing these FULL locs and unbind the mapping. Page-atomic:
        the incoming locs must cover each touched page completely (guaranteed by the padded
        finish tails / aligned frontiers / page-aligned tree values). Idempotent over already
        unbound (slid/tombstoned) pages."""
        if full_indices.numel() == 0:
            return
        P = self.P
        fi = full_indices.to(device=self._device, dtype=torch.int64)
        fi = fi[fi >= 0]
        if fi.numel() == 0:
            return
        fbases, counts = torch.unique(
            torch.div(fi, P, rounding_mode="floor") * P, return_counts=True
        )
        assert bool((counts == P).all()), (
            f"free_swa got partial pages (counts {counts[counts != P].tolist()[:4]})"
        )
        ws = self.full_to_window[fbases]
        live = ws[ws >= 0]
        offsets = torch.arange(P, dtype=torch.int64, device=self._device)
        self.full_to_window[(fbases[:, None] + offsets).flatten()] = -1
        if live.numel():
            self._win_alloc.free(torch.div(live, P, rounding_mode="floor") * P)

    def translate_loc_from_full_to_swa(self, kv_indices: torch.Tensor) -> torch.Tensor:
        return self.full_to_window[kv_indices.to(dtype=torch.int64)]

    # ----- state_loc derivation (vectorized, LongTensor in/out) -----
    @staticmethod
    def state_loc(window_slot: torch.Tensor, ring_size: int, P: int) -> torch.Tensor:
        pages = torch.div(window_slot, P, rounding_mode="floor")
        loc = pages * ring_size + (window_slot % ring_size)
        return torch.where(window_slot < 0, torch.full_like(loc, -1), loc)

    def ring_size(self, layer_id: int) -> int:
        source = self.state_source_of(layer_id)
        assert source is not None, f"layer {layer_id} has no compress-state ring"
        return dsv41_ring_size_for_ratio(self.compress_ratios[source])

    # ----- compress-state ring accessors (band-routed) -----
    def get_state(self, layer_id: int, state_loc: torch.Tensor) -> torch.Tensor:
        ring = self.state_ring[layer_id]
        assert ring is not None, f"layer {layer_id} has no compress-state ring"
        return ring.get(state_loc)

    def set_state(self, layer_id: int, state_loc: torch.Tensor, kv_score: torch.Tensor) -> None:
        ring = self.state_ring[layer_id]
        assert ring is not None, f"layer {layer_id} has no compress-state ring"
        ring.set(state_loc, kv_score)

    # ----- specialized writes (band-routed) -----
    def store_window(self, k: torch.Tensor, layer_id: int, window_slot: torch.Tensor) -> None:
        self.window_pool[layer_id].index_copy_(0, window_slot, k.to(self._dtype))

    def store_compressed(self, kv: torch.Tensor, layer_id: int, cmp_slot: torch.Tensor) -> None:
        pool = self.cmp_pool[layer_id]
        assert pool is not None, (
            f"layer {layer_id} (no kv-source band) has no compressed pool"
        )
        pool.index_copy_(0, cmp_slot, kv.to(self._dtype))

    def store_indexer(self, k: torch.Tensor, layer_id: int, idx_slot: torch.Tensor) -> None:
        pool = self.idx_pool[layer_id]
        assert pool is not None, (
            f"layer {layer_id} has no indexer pool (not in an index-source band)"
        )
        pool.index_copy_(0, idx_slot, k.to(self._dtype))

    def total_bytes(self) -> int:
        n = self.full_to_window.numel() * self.full_to_window.element_size()
        n += sum(t.numel() * t.element_size() for t in self.window_pool)
        # Owners only: a band's tensor is referenced by every consumer layer (identity sharing),
        # so summing the per-layer lists would count a shared pool once per reader.
        n += sum(
            self.cmp_pool[src].numel() * self.cmp_pool[src].element_size()
            for src in self._cmp_owners
        )
        n += sum(
            self.idx_pool[src].numel() * self.idx_pool[src].element_size()
            for src in self._cmp_owners
        )
        n += sum(
            self.state_ring[src].buffer.numel() * self.state_ring[src].buffer.element_size()
            for src in self._cmp_owners
        )
        return int(n)

    # ----- BaseKVCachePool interface -----
    def k_cache(self, index: int) -> torch.Tensor:
        return self.window_pool[index]

    def v_cache(self, index: int) -> torch.Tensor:  # MLA: K == V (single latent)
        return self.window_pool[index]

    def store_kv(self, k, v, out_loc, layer_id) -> None:
        # Thin window-write shim for ABC compat; DSV4.1 writes via the specialized setters
        # above. K == V (single latent), so the window slot is out_loc.
        self.store_window(k, layer_id, out_loc)

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def num_layers(self) -> int:
        return self._n_layers


__all__ = ["CompressStateRing", "DSV41PagedKVCache", "FreeListAllocator"]
