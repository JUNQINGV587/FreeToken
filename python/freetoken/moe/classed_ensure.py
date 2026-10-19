"""Classed LRU admission for the MoE slot cache (disk-tier mode).

Vendored fork of flashlib's ``_lru_ensure_kernel`` (the ``_seq`` strategy — the only
strategy reachable at our cache size, 1440 << 40,000) with one change: the cache is
split into two regions and a miss may only evict inside its own class's region.

* main region  ``[0, pin_start)``        — disk-tier rows (NVMe-fetched, working set
                                           ~1376–1720/rank, wants static residency)
* pin region   ``[pin_start, num_cached)`` — RAM-pin rows (PCIe-served, churn by
                                           design: ~4128 rows/rank cycle through)

Today both classes share one global LRU, so the permanently-churning pin rows keep
evicting the disk rows whose residency is the entire point of the cache (TUNING-262K
rounds 9/10: decode NVMe reads never converge to zero). With classed admission the
disk rows LRU only among themselves and converge; the pin rows get a small mini-LRU
scratch region, which is all their access pattern can exploit anyway.

Only slot *assignment* changes — the bytes installed into a slot are the same expert
rows, so numerics are identical to the vanilla kernel (modulo the scheduling
nondeterminism already present there).

Safety contract (enforced by the caller in ``offload_kernels.ensure_experts``): the
per-call distinct miss count of each class must not exceed that class's region size,
otherwise the region exhausts and the masked argmin would recycle a claimed slot.
The caller bounds it conservatively by ``query.numel() <= min(pin_start, num_cached -
pin_start)`` and falls back to flashlib's ``lru_ensure`` otherwise (prefill's wide
queries take that path; decode's ``bs * topk <= 32`` always fits a 128-slot region).

Default pin region: 1024 slots (``FREETOKEN_PIN_CACHE_SLOTS``). History on the V4-Flash
1440-slot cache: the original 2026-10-18 sweep {64,128,256} looked flat in speed, which
was misread as "routing is iid, capacity doesn't matter". Oracle replay of a real
production routing trace (2026-10-19, runs/202610-v4f-cache-oracle/RESULT.md) proved the
opposite: 128 slots thrash the pin class (49.2 fetch rows/token vs 10.7 at 1024), and a
production A/B confirmed it — single-stream decode 7.5 -> 9.1 t/s (+21%) with the disk
class still protected (0.97 NVMe rows/token vs 1.98 unified). 1024 is the sweet spot:
bigger regions erode disk-row residency without cutting total fetches further.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from flashlib.kernels.slot_cache.triton.lru_ensure import (
    _num_warps_for,
    _phase1,
    _stats,
    _validate,
)


@triton.jit(do_not_specialize=["K", "num_cached", "id_base", "pin_start", "ram_local"])
def _classed_lru_ensure_kernel(
    query_ptr, slot_of_id_ptr, id_of_slot_ptr, lru_usage_ptr, lru_step_ptr,
    out_ptr, src_ptr, dst_ptr, num_copy_ptr, stats_ptr,
    row_map_ptr, K, num_cached, id_base, pin_start, ram_local,
    BLOCK_K: tl.constexpr, BLOCK_C: tl.constexpr,
    USAGE_MAX: tl.constexpr, COLLECT_STATS: tl.constexpr,
):
    """flashlib ``_lru_ensure_kernel`` with class-restricted victim selection.

    ``row_map[local_id]`` is the disk tier's expert -> bank-row map for this layer
    (layer-major device tensor row, same source ``graph_fetch._gf_request_kernel``
    uses); ``row < ram_local`` classifies the query as a RAM-pin row. Pin misses pick
    victims from ``[pin_start, num_cached)``, disk misses from ``[0, pin_start)``.
    Hit bump, dedup, ranking, install order, and stats are byte-for-byte the vanilla
    algorithm — only the candidate mask per victim pick is new.
    """
    step = tl.load(lru_step_ptr) + 1
    tl.store(lru_step_ptr, step)
    q, kmask, miss, first, first_miss, rank, num_missing, out = _phase1(
        query_ptr, slot_of_id_ptr, lru_usage_ptr, num_copy_ptr, step, K, BLOCK_K,
        id_base)

    # Classify each query by its bank row. ``q`` is already global; the map is
    # indexed by the layer-local id. Masked lanes load row 0 and are ignored below.
    row = tl.load(row_map_ptr + (q - id_base), mask=kmask, other=0)
    is_pin = kmask & (row < ram_local)

    if num_missing > 0:
        # Same required CTA fence as vanilla: the phase-1 hit bump is a scatter store
        # and the usage reload below must observe it.
        tl.debug_barrier()
        c = tl.arange(0, BLOCK_C)
        cmask = c < num_cached
        u = tl.load(lru_usage_ptr + c, mask=cmask, other=USAGE_MAX)
        umax = tl.full([BLOCK_C], USAGE_MAX, u.dtype)
        u = tl.where((u == step) | (~cmask), umax, u)
        for i in tl.range(num_missing):
            # 1 when the rank-i miss is a pin-class row: restrict the candidate
            # usage vector to that class's region for this pick.
            p = tl.sum(tl.where((rank == i) & first_miss, is_pin.to(tl.int32), 0))
            u_c = tl.where((c >= pin_start) == (p == 1), u, umax)
            victim = tl.argmin(u_c, axis=0).to(tl.int32)
            # Scalar load, not a masked block reduction: victims are distinct, so no
            # earlier iteration of this loop wrote the slot being read.
            old = tl.load(id_of_slot_ptr + victim)
            if old >= 0:
                tl.store(slot_of_id_ptr + old, -1)
            e = tl.sum(tl.where((rank == i) & first_miss, q, 0))
            tl.store(id_of_slot_ptr + victim, e)
            tl.store(slot_of_id_ptr + e, victim)
            tl.store(lru_usage_ptr + victim, step)
            tl.store(dst_ptr + i, victim)
            tl.store(src_ptr + i, e - id_base)  # back to the caller's id space
            out = tl.where((rank == i) & miss, victim, out)
            u = tl.where(c == victim, umax, u)  # claim in-register

    # Written from registers, never re-read from slot_of_id, so `out_ptr` is allowed
    # to alias `query_ptr` (every read of q happened above).
    tl.store(out_ptr + tl.arange(0, BLOCK_K), out, mask=kmask)
    if COLLECT_STATS:
        _stats(stats_ptr, first, num_missing)


def classed_ensure(
    query: torch.Tensor,
    slot_of_id: torch.Tensor,
    id_of_slot: torch.Tensor,
    lru_usage: torch.Tensor,
    lru_step: torch.Tensor,
    out_indices: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    num_copy: torch.Tensor,
    row_map: torch.Tensor,
    ram_local: int,
    pin_start: int,
    stats: torch.Tensor | None = None,
    id_base: int = 0,
) -> None:
    """Classed admission, drop-in for ``flashlib.kernels.slot_cache.lru_ensure``.

    Same arguments as ``lru_ensure`` plus:

    Args:
        row_map: ``(num_experts,)`` int32/int64 on the query's device — this layer's
            expert -> bank-row map (``DiskTier._row_map_dev[layer_id]``).
        ram_local: pinned bank-row count (``DiskTier._ram``); rows below it are the
            pin class.
        pin_start: first slot of the pin region; ``[0, pin_start)`` is the disk
            class's region.

    The caller must guarantee per-class ``|distinct misses| <= region size`` (see
    module docstring). Everything stays on device with fixed shapes, so the call is
    CUDA-graph capturable like the vanilla kernel.
    """
    k = query.numel()
    num_cached = id_of_slot.numel()
    _validate(query, slot_of_id, id_of_slot, lru_usage, lru_step, out_indices,
              src_indices, dst_indices, k, num_cached)
    assert row_map.dtype in (torch.int32, torch.int64) and row_map.is_contiguous()
    assert row_map.is_cuda == query.is_cuda
    assert 0 < pin_start < num_cached, f"pin_start={pin_start} outside (0, {num_cached})"
    assert 0 <= ram_local <= row_map.numel()
    block_c = triton.next_power_of_2(num_cached)
    _classed_lru_ensure_kernel[(1,)](
        query, slot_of_id, id_of_slot, lru_usage, lru_step,
        out_indices, src_indices, dst_indices, num_copy, stats,
        row_map, k, num_cached, id_base, int(pin_start), int(ram_local),
        BLOCK_K=triton.next_power_of_2(k),
        BLOCK_C=block_c,
        USAGE_MAX=torch.iinfo(lru_usage.dtype).max,
        COLLECT_STATS=stats is not None,
        num_warps=_num_warps_for(block_c),
    )
