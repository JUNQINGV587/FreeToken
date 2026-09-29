"""Composite cost model + independent per-tier sizing for DSV4.1 paged KV.

V4.1 (CSA2) shares ONE compressed cache per BAND of backbone layers: the tier is owned
by the band's KV-source layer (``args.kv_source_layers``) and every reader in the band
addresses the identical tensor. The indexer tier is owned by the band's index-source
layer (``args.index_source_layers``, a superset of the KV sources). The window tier is
per-layer on all ``n_layers``, as in DSV4.

Consequence for the per-layer lists below: a tier slot is non-None ONLY on the layer
that OWNS it, so summing a list can never double-count a shared band.
``DSV41PagedKVCache`` is what expands ownership to the whole band (each consumer layer
holds the owner's object).

Byte conventions:
  kv bytes    = head_dim       * 2  (bf16)
  index bytes = index_head_dim * 2  (bf16)
  state bytes = 2 * head_dim   * 4  (fp32)
  ring_size   = compress_ratio
"""

from __future__ import annotations

import os

from dataclasses import dataclass, field

_BF16_BYTES = 2
_FP32_BYTES = 4
_INT64_BYTES = 8


def dsv41_reserved_window_pages(max_running_req: int, radix: bool) -> int:
    """Window pages the sliding pool must always keep for the concurrent working set.

    Same formula as DSV4's: the reservation is a property of the generic
    CacheManager/SWARadixCache working set (each row's decode transients +, in radix
    mode, one locked live-tail page and one retained prompt-end window per concurrent
    request), not of the compressor, and V4.1 runs the same manager.
    """
    return 2 * (max_running_req + 1) + (3 * max_running_req if radix else 0) + 1


def dsv41_ring_size_for_ratio(ratio: int) -> int:
    """Compress-state ring slots per window page: one per in-page group phase.

    The V4.1 compressor pools plain token groups (``models/deepseek_v41/compress.py``:
    the incomplete trailing group is carried in ``_kv_state``/``_score_state``), with no
    DSV4-style overlapping carry blocks, so the ring needs exactly ``ratio`` slots per
    page. ``ring_size | P`` (128 % 1 == 0, 128 % 2 == 0), so distinct pages map to
    disjoint ring blocks.
    """
    if ratio in (1, 2):
        return ratio
    raise ValueError(f"no ring for ratio {ratio} (DSV4.1 compressor ratios are 1 and 2)")


def _kv_bytes(args) -> int:
    return args.head_dim * _BF16_BYTES


def _index_bytes(args) -> int:
    return args.index_head_dim * _BF16_BYTES


def _state_bytes(args) -> int:
    # coff == 1: the ring row is ``kv | score`` over one head_dim each (no overlap half).
    return 2 * args.head_dim * _FP32_BYTES


def dsv41_kv_sources(args) -> tuple[int, ...]:
    """KV-source layers the pool covers (sorted, in range, ratio > 0 -- a ratio-0 source
    owns no compressed tier, so it shares nothing)."""
    sources = sorted(s for s in args.kv_source_layers if 0 <= s < args.n_layers)
    dead = [s for s in sources if _ratio_of(args, s) == 0]
    assert not dead, f"kv_source_layers {dead} have ratio 0 and own no compressed tier"
    return tuple(sources)


def dsv41_index_sources(args) -> tuple[int, ...]:
    """Index-source layers the pool covers (sorted, in range)."""
    return tuple(sorted(s for s in args.index_source_layers if 0 <= s < args.n_layers))


def dsv41_cmp_source_of(args, layer_id: int) -> int | None:
    """The KV-source layer whose shared compressed/indexer/state tiers ``layer_id``
    addresses (the most recent source at or before it), or None for window-only layers."""
    if not 0 <= layer_id < args.n_layers:
        return None
    source = None
    for s in dsv41_kv_sources(args):
        if s <= layer_id:
            source = s
    return source


def dsv41_index_source_of(args, layer_id: int) -> int | None:
    """The index-source layer whose indexer tier ``layer_id`` addresses (most recent at
    or before it), or None outside every index source's band."""
    if not 0 <= layer_id < args.n_layers:
        return None
    source = None
    for s in dsv41_index_sources(args):
        if s <= layer_id:
            source = s
    return source


def _ratio_of(args, layer_id: int) -> int:
    return int(args.layer_ratio(layer_id))


def _band_ratio(args, layer_id: int) -> int:
    """Compressor ratio of ``layer_id``'s band -- the ratio of its KV source. The
    compressed AND indexer row counts of a band are both ``full_token // band_ratio``
    (an index source that is not a KV source has no ``wk`` of its own and derives its
    indexer rows from the band's compressed rows)."""
    source = dsv41_cmp_source_of(args, layer_id)
    assert source is not None, f"layer {layer_id} has no compressed band"
    ratio = _ratio_of(args, source)
    assert ratio > 0, f"layer {layer_id}'s band source {source} has ratio 0"
    return ratio


def dsv41_cache_per_page(args, swa_ratio: float, P: int = 128) -> int:
    """Bytes per P-token across ALL tiers, for the budget DIVISION only.

    The window term is summed over LAYERS (per-layer tier); the compressed/indexer/state
    terms are summed over OWNERS, since one band is one allocation. The state rings are
    sized off the WINDOW pages, so their per-FULL-page share is the fractional
    ``swa_ratio * ring_size``; it is accumulated exactly rather than rounded per layer
    (a V4.1 ring is only 1-2 slots, where rounding would drop the term entirely).
    """
    assert P % 1 == 0 and P > 0
    kv_b = _kv_bytes(args)
    idx_b = _index_bytes(args)

    total = 0
    total += args.n_layers * round(swa_ratio * P) * kv_b  # window tier: every layer
    for src in dsv41_kv_sources(args):
        ratio = _ratio_of(args, src)
        total += (P // ratio) * kv_b  # compressed KV: P//ratio blocks per page
    for src in dsv41_kv_sources(args):
        total += (P // _ratio_of(args, src)) * idx_b  # index keys: one row per band row
    state = 0.0
    for src in dsv41_kv_sources(args):
        state += swa_ratio * dsv41_ring_size_for_ratio(_ratio_of(args, src)) * _state_bytes(args)
    return int(total + state)


def dsv41_kv_unit_bytes(args, P: int = 128) -> int:
    """FULL-tier bytes per full-history token: compressed KV + indexer KV + the
    full->window mapping. These tiers scale with the full anchor
    (``cmp_blocks = full_token // ratio``), so the cost is independent of ``swa_ratio``.
    The window pool and its state rings are NOT here -- see
    :func:`dsv41_window_unit_bytes`. This is the ``kv_bytes_per_token`` the cache-status
    slider divides the VRAM budget by."""
    kv_b = _kv_bytes(args)
    idx_b = _index_bytes(args)
    per_page = P * _INT64_BYTES  # full_to_window map: one int64 slot per full token
    for src in dsv41_kv_sources(args):
        per_page += (P // _ratio_of(args, src)) * kv_b
    for src in dsv41_kv_sources(args):
        per_page += (P // _ratio_of(args, src)) * idx_b
    return -(-per_page // P)  # ceil to bytes/token (conservative slider max)


def dsv41_window_unit_bytes(args, P: int = 128) -> int:
    """WINDOW(swa)-tier bytes per window token: the sliding KV pool (every layer) plus
    the compress-state rings (sized off the window pages, ``state_slots = n_win_pages *
    ring_size``). Independent of ``swa_ratio`` -- the ratio only sets how many window
    tokens exist, not the per-token cost. This is ``swa_bytes_per_token``."""
    kv_b = _kv_bytes(args)
    per_page = args.n_layers * P * kv_b  # window KV: P slots per page, every layer
    for src in dsv41_kv_sources(args):
        per_page += dsv41_ring_size_for_ratio(_ratio_of(args, src)) * _state_bytes(args)
    return -(-per_page // P)  # ceil to bytes/window-token


@dataclass
class DSV41PoolSizes:
    """Per-tier slot counts derived from the budget anchor ``full_token``.

    Every list is indexed by layer and is non-None ONLY at the layer that owns the tier,
    which is the KV source for all four of them: ``cmp_blocks``/``state_slots``/``ring_sizes``
    are the band's compressed KV and its compressor state, and ``idx_blocks`` is the
    index-K cache the KV source's indexer fills (a non-owning index source has no ``wk``).
    Band consumers are None here and resolve to the owner through the pool's
    ``cmp_source_of``/``idx_source_of`` mapping -- for the tiers, ``idx_source_of`` is the
    same band KV source, while the pool's ``idx_source_of`` reports the *publication*
    axis (which indexer publishes topk/candidates).
    """

    P: int
    swa_ratio: float
    full_token: int  # num_pages * P  (the budget anchor)
    n_win_slots: int  # global window pool rows (bf16 kv)
    n_win_pages: int  # n_win_slots // P
    cmp_blocks: list[int | None] = field(default_factory=list)
    idx_blocks: list[int | None] = field(default_factory=list)
    state_slots: list[int | None] = field(default_factory=list)
    ring_sizes: list[int | None] = field(default_factory=list)


def dsv41_pool_sizes(
    num_pages: int, args, swa_ratio: float, P: int = 128, n_win_pages: int | None = None
) -> DSV41PoolSizes:
    n_layers = args.n_layers
    ratios = tuple(args.compress_ratios)[:n_layers]
    full_token = num_pages * P

    # Window tier: swa_ratio of full history rounded UP to a whole page count, OR the
    # caller's explicit page count (the working-set floor is applied exactly ONCE, by the
    # caller, in pages). Capped at the full history.
    if n_win_pages is None:
        raw = round(swa_ratio * full_token)
        n_win_pages = (raw + P - 1) // P
    n_win_pages = min(n_win_pages, num_pages)
    n_win_slots = n_win_pages * P

    kv_sources = set(dsv41_kv_sources(args))

    cmp_blocks: list[int | None] = []
    idx_blocks: list[int | None] = []
    state_slots: list[int | None] = []
    ring_sizes: list[int | None] = []
    for L, ratio in enumerate(ratios):
        if L in kv_sources:
            assert ratio > 0, f"kv source {L} has ratio 0"
            assert full_token % ratio == 0, f"full_token {full_token} not divisible by {ratio}"
            rs = dsv41_ring_size_for_ratio(ratio)
            cmp_blocks.append(full_token // ratio)
            state_slots.append(n_win_pages * rs)
            ring_sizes.append(rs)
        else:
            cmp_blocks.append(None)
            state_slots.append(None)
            ring_sizes.append(None)
        # The INDEX-K cache belongs to the KV source, not to the nearest index source: only a
        # kv-source layer's indexer has ``wk``/``k_norm`` to build keys, and the non-owning
        # index sources (24/28/32/36) read that same cache (vLLM: "non-owning index sources
        # share the K cache of the latest kv source below them"). ``index_source_layers``
        # governs which indexers exist and who publishes the topk/candidate lists -- model-side
        # state, not a pool tier -- so it must not size this one.
        idx_blocks.append(full_token // ratio if L in kv_sources else None)

    return DSV41PoolSizes(
        P=P,
        swa_ratio=swa_ratio,
        full_token=full_token,
        n_win_slots=n_win_slots,
        n_win_pages=n_win_pages,
        cmp_blocks=cmp_blocks,
        idx_blocks=idx_blocks,
        state_slots=state_slots,
        ring_sizes=ring_sizes,
    )


def dsv41_pool_bytes(sizes: DSV41PoolSizes, args, n_scratch: int = 1) -> int:
    """Exact bytes a ``DSV41PagedKVCache`` built from ``sizes`` allocates (mirror of
    ``total_bytes`` + the full_to_window mapping), including the scratch/sentinel rows
    (n_scratch per cmp/idx pool, +1 ring scratch row, +1 mapping sentinel row).

    Owners only: a band is one allocation however many layers read it. There is no
    indexer-state term -- V4.1's indexer has no compressor of its own (see the pool's
    ``indexer_state_ring`` note).
    """
    kv_b = _kv_bytes(args)
    idx_b = _index_bytes(args)

    total = args.n_layers * sizes.n_win_slots * kv_b  # window pool, every layer
    total += (sizes.full_token + 1) * _INT64_BYTES  # full_to_window (+ sentinel row)
    for src in dsv41_kv_sources(args):
        total += (sizes.cmp_blocks[src] + n_scratch) * kv_b
        total += (sizes.state_slots[src] + 1) * _state_bytes(args)
    for src in dsv41_kv_sources(args):
        total += (sizes.idx_blocks[src] + n_scratch) * idx_b
    return int(total)


def dsv41_solve_num_pages(
    available_bytes: int,
    args,
    swa_ratio: float,
    floor_win_pages: int,
    P: int = 128,
    n_scratch: int = 1,
) -> DSV41PoolSizes:
    """Largest budget-respecting pool: max ``num_pages`` with exact
    ``dsv41_pool_bytes(sizes(num_pages, win=max(floor, ceil(r*num)))) <= available_bytes``.

    The window floor is honored in PAGES (never by inflating ``swa_ratio``) and the total
    is byte-checked: at small budgets the window pins at ``floor_win_pages`` and the
    full/cmp/idx anchor SHRINKS to fit. Raises ``ValueError`` when even the minimal pool
    does not fit.
    """
    def _sizes(num: int) -> DSV41PoolSizes:
        win = max(floor_win_pages, (round(swa_ratio * num * P) + P - 1) // P)
        return dsv41_pool_sizes(num, args, swa_ratio, P=P, n_win_pages=win)

    lo = max(floor_win_pages, 2)  # full history must at least cover the window working set
    if dsv41_pool_bytes(_sizes(lo), args, n_scratch) > available_bytes:
        raise ValueError(
            f"DSV4.1 KV budget {available_bytes} bytes cannot fit the minimal pool "
            f"({lo} pages incl. the window working-set floor {floor_win_pages}); "
            "raise memory_ratio or lower max_running_req/max_seq_len"
        )
    hi = max(lo, available_bytes // max(1, dsv41_cache_per_page(args, 0.0, P)))
    while dsv41_pool_bytes(_sizes(hi), args, n_scratch) <= available_bytes:
        hi *= 2  # cheap upper bracket (cache_per_page(0.0) undercounts the window term)
    while lo < hi - 1:
        mid = (lo + hi) // 2
        if dsv41_pool_bytes(_sizes(mid), args, n_scratch) <= available_bytes:
            lo = mid
        else:
            hi = mid
    return _sizes(lo)


_AUTO_KV_SLACK_BYTES = 2 << 30  # absorbs plan-vs-measured drift and leaves a usable pool


def dsv41_auto_cost_model(args, swa_ratio, floor_win_pages, P=128, n_scratch=1):
    """Affine (cache_per_page, fixed_cache_size, min_reserve_tokens) for the MoE-first
    auto planner: exact marginal per-page cost across all tiers (+ the full_to_window
    mapping), a fixed intercept anchored at the minimal viable pool, and a reserve floor
    covering the window working set plus a slack absorbing plan-vs-measured drift.
    Conservative at the shipped swa_ratio; an extreme swa_ratio can dip slightly under
    exact (harmless -- num_pages is re-solved byte-exactly from measured memory)."""
    per_page = dsv41_cache_per_page(args, swa_ratio, P) + P * _INT64_BYTES
    n0 = max(floor_win_pages, 2)
    win0 = max(floor_win_pages, (round(swa_ratio * n0 * P) + P - 1) // P)
    base = dsv41_pool_bytes(
        dsv41_pool_sizes(n0, args, swa_ratio, P=P, n_win_pages=win0), args, n_scratch
    )
    slack_pages = -(-_AUTO_KV_SLACK_BYTES // per_page)
    min_reserve_tokens = (n0 + slack_pages) * P
    return per_page, max(0, base - n0 * per_page), min_reserve_tokens


__all__ = [
    "DSV41PoolSizes",
    "dsv41_auto_cost_model",
    "dsv41_cache_per_page",
    "dsv41_cmp_source_of",
    "dsv41_index_source_of",
    "dsv41_index_sources",
    "dsv41_kv_sources",
    "dsv41_kv_unit_bytes",
    "dsv41_pool_bytes",
    "dsv41_pool_sizes",
    "dsv41_reserved_window_pages",
    "dsv41_ring_size_for_ratio",
    "dsv41_solve_num_pages",
    "dsv41_window_unit_bytes",
]


# ---- config-facing sizing (the engine/pool speak EngineConfig; the functions above are
# geometry-only). Same window:full decoupling as DSV4: the window tier is all-sliding
# (only the last 128 positions are read), so it needs only ~swa_ratio of full history;
# the shared cmp/idx tiers stay sized to the FULL history. The ratio lives on the config
# (a runtime rebuild can change it).


def _dsv41_swa_ratio(config) -> float:
    """The live DSV4.1 window/full ratio from the serving config
    (config.swa_full_tokens_ratio; a runtime rebuild can change it)."""
    return float(config.swa_full_tokens_ratio)


def _dsv41_window_floor_pages(config, P: int) -> int:
    """Minimum window pages = the live sliding working set the window pool must always
    hold: one prefill chunk's reach (capped at 8 pages == 1024 tok; chunked prefill bounds
    the rest) + each running request's 128-tail + one radix-locked live tail page per
    concurrent cached prompt + the reserved dummy. This is the only HARD floor on DSV4.1
    KV sizing; the full-history capacity above it is purely memory-derived."""
    prefill_reach_pages = (config.max_seq_len + P - 1) // P
    # The generic manager materializes a windowed cache as "swa_radix", so the radix term
    # keys on "not naive" (same reading as DSV4's).
    radix = config.cache_type != "naive"
    return min(prefill_reach_pages, 8) + dsv41_reserved_window_pages(config.max_running_req, radix)


def _dsv41_pool_sizes(config, num_pages: int, num_swa_pages: int | None = None):
    # P (the window-page size) is the sliding window (128), the radix block key --
    # independent of the generic page_size. num_pages (P units, PHYSICAL incl the dummy) is
    # the budget anchor; full_token = num_pages * P. Window sizing precedence: an explicit
    # num_swa_pages (validate's target, usable pages) > config.swa_num_pages_override (a
    # pinned window) > swa_ratio x full.
    args = config.model_config.dsv41_args
    P = args.window_size
    swa_ratio = _dsv41_swa_ratio(config)
    # Test hook DSV41_FORCE_SMALL_POOL: shrink the budget anchor so a multi-prompt workload
    # OVERFLOWS the pool and exercises the bounded-cache (radix LRU) eviction path.
    # BYPASSES the working-set floor by design (the point is a too-small pool).
    override = os.environ.get("DSV41_FORCE_SMALL_POOL")
    if override:
        num_pages = max(2, int(override))  # in 128-pages; keep >=2 so the dummy page + 1 fit
        return dsv41_pool_sizes(
            num_pages=num_pages, args=args, swa_ratio=swa_ratio, P=P,
        )

    floor_pages = _dsv41_window_floor_pages(config, P)
    target = num_swa_pages if num_swa_pages is not None else config.swa_num_pages_override
    if target is not None:
        # Absolute window: `target` usable pages + 1 dummy, floored and capped at the anchor.
        win = min(num_pages, max(floor_pages, int(target) + 1))
    else:
        # Default: ratio x full, rounded UP to whole window pages (floor applied ONCE).
        win = max(floor_pages, (round(swa_ratio * num_pages * P) + P - 1) // P)
    return dsv41_pool_sizes(
        num_pages=num_pages,
        args=args,
        swa_ratio=swa_ratio,
        P=P,
        n_win_pages=win,
    )
