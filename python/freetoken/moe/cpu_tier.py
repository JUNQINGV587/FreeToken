# Copyright (c) 2026, FreeToken contributors.
"""RAM-resident miss tier: compute slot-cache misses on the CPU instead of fetching.

M2 of the dsv41 split-protocol port (see ct/ct_vllm.py in dsv41-flash-offload).
With the disk tier, every decode miss currently rides the PCIe/NVMe fetch path
(``copy_missing``): the expert weights come to the GPU. This tier flips the
cheapest subset of misses the other way -- the TOKEN goes to the CPU, the
pinned host bank row is GEMV'd in place, and only the H-sized partial result
crosses back. The RAM-resident pin set (--expert-ram-experts / --moe-ram-pin-file)
makes ~80% of misses CPU-serviceable, so decode stops being gated on weight
movement.

Three-way miss classification (graph_fetch.py's priority contract, in order):
  1. slot HIT              -> GPU GEMM (unchanged)
  2. miss on a RAM pin row -> CPU tier (this module), classified BEFORE any
     doorbell request is built so ``graph_stage_fetch`` never sees the row
  3. miss on a disk row    -> doorbell/PCIe fetch (unchanged)

Mechanics of a claim: ``ensure_experts`` (flashlib lru_ensure) rewrites
``topk_ids`` to slot ids in place -- misses included (their slots are reserved
through ``evict_slots``/``src_indices``/``num_indices``, the copy lands in
``copy_missing``). The split kernel walks that staged plan, picks the cheapest
CPU-serviceable misses (dsv41 cost model), and for each claimed expert:
  * zeroes the routed entries' weights and points their slot ids at slot 0
    (the GPU kernels address ``packed_ptr + slot*stride`` without a negative
    guard, so the hybrid path's clamp_min(0)+zero-weight sentinel convention
    is applied in-kernel -- no extra torch ops, no OOB reads);
  * undoes the slot reservation (slot_for_id[e]=-1, id_of_slot[slot]=-1; the
    usage stamp is left alone -- the next ensure's argmin recycling self-heals);
  * compacts the staged plan so ``copy_missing``/doorbell fetch the remainder
    only;
  * publishes one pick per (token, expert) pair -- (token, bank row, weight
    bits) -- plus the tokens' hidden states to the pinned request block.

Graph-safety / transport contract (measured in graph_fetch.py's header):
  * device kernels NEVER write host sysmem from a replayed graph (the write is
    lost) -- the split kernel therefore publishes into DEVICE staging
    (ctrl_dev/hx_dev/picks_dev) and the capture records cudaMemcpyAsync D2H
    nodes that pull it into the pinned host mirror (ctrl[7]=seq sits at the END
    of the block, so a host observer seeing a new seq knows the whole block
    landed; hx/picks are pulled by their own earlier copies).
  * host->device is the safe direction: the C++ service plain-stores the done
    word into a pinned tensor and the combine kernel reads it back through a
    sys-scope acquire atomic; hout is read over PCIe by the combine kernel.
  * the host side makes ZERO CUDA calls.

Determinism contract: split is single-block with scalar decision loops (no
cross-block order), the cost model selects by (count asc, staged ordinal asc),
and the C++ side accumulates per-pick partials in pick order -- no atomic-order
dependence anywhere, so a fixed routing gives bitwise-stable outputs across
replays.

Numerics contract: ``_cpu_moe.CpuTierService`` reuses the exact gate/up and down
GEMV kernels (nvfp4 / ds_fp4, ISA-dispatched) and the per-route rounding rules
of ``CpuMoeExecutor`` (ds_fp4: bf16-round each gate/up dot, FP8 round-trip the
input and the intermediate, bf16-round every route's weighted output; nvfp4:
fp32 nvdot path, weight at the input when apply_router_weight_on_input).
``selftest_layer`` compares against a pure-PyTorch reference over the same
rounding rules (the GPU fused comparison is deferred to the GPU battery).

Env knobs (all optional, all read at construction):
  FREETOKEN_CT_THREADS  worker threads (default 0 = one per physical core)
  FREETOKEN_CT_TZC / FREETOKEN_CT_THIT / FREETOKEN_CT_A / FREETOKEN_CT_B /
  FREETOKEN_CT_TOK / FREETOKEN_CT_MAXN
      dsv41 cost-model parameters (defaults 0.58/0.03/0.11/0.20/0.35/384,
      re-calibrated locally by ``bench_isa_tiers``)
  FREETOKEN_CT_FORCE_N  int >= 0: serve the N cheapest CPU-ok miss experts per
      layer regardless of the cost model (dsv41's calibration knob; -1 = off)
  FREETOKEN_CT_TIMEOUT_NS  combine spin budget per reporting interval
      (default 300ms; on expiry the kernel KEEPS SPINNING and bumps the
      timeouts counter -- a hang is loud, a wrong output is silent)

20261007 rescue pieces (design: notes/engines/20261007-cpu-tier-rescue-design.md;
all default off, off == pre-rescue behaviour):
  FREETOKEN_CT_PRE_ENSURE  1 = claim BEFORE ensure_experts (pieces A+B):
      claimed experts get a slot-0 sentinel so lru_ensure treats them as hits
      (no staged eviction/fetch -> no churn) and the CPU job overlaps
      ensure+fetch+GEMM instead of fetch+GEMM alone. The combine kernel
      restores slot_for_id[e] = -1 from the claimed list.
  FREETOKEN_CT_REUSE_MIN  piece C: claim only misses whose reuse score
      (this-step duplicates + XSTEP_W * cross-step routing frequency) is >=
      this threshold; -1 = off (claim every CPU-ok miss, the old policy).
      FORCE_N bypasses the filter (calibration comparability).
  FREETOKEN_CT_XSTEP_W  weight of the cross-step frequency term (default 1.0)
"""

from __future__ import annotations

import json
import math
import os
import threading
from collections.abc import Iterable

import torch
import triton
import triton.language as tl

from freetoken.kernel.pinned import alloc_pinned_tensor

_CTRL_LEN = 8
# Kernel-body references must be tl.constexpr instances: Triton rejects plain
# module globals inside @triton.jit (NameError at compile). Host-side uses keep
# working -- tl.constexpr supports int() and tensor indexing.
_SEQ_OFF = tl.constexpr(7)  # LAST field of the block: the D2H memcpy lands it last
# ctrl[3]: claimed-EXPERT count of the pre-ensure path (ctrl[2] counts PICKS;
# the combine needs the expert count to restore the slot-0 sentinels). The
# legacy kernel never writes it, so it stays 0 there (zeroed at attach).
_CTRL_NCLAIMED = tl.constexpr(3)
_DFLAG_WAIT_NS = tl.constexpr(1)
_DFLAG_TIMEOUTS = tl.constexpr(2)
_SPIN_NS_PER_ITER = tl.constexpr(5.0)  # rough per-iteration cost of the combine spin loop
_DEFAULT_TIMEOUT_NS = 300_000_000
_BIG = tl.constexpr(2**30)

# dsv41 cost-model semantics, verbatim (ct_vllm.py):
#   gpu(k) = THIT*(nh+nm-k) + TZC*(nm-k)         ms, k miss experts moved to CPU
#   cpu(k) = A + sum_i B*(1 + TOK*(cnt_i - 1))   ms, over the k cheapest
#   pick the k minimizing tt = max(gpu(k), cpu(k)); serve nothing when the best
#   tt <= 0 (no CPU-ok misses / model saturated)
_COST_ENV = {
    "tzc": ("FREETOKEN_CT_TZC", 0.58),
    "thit": ("FREETOKEN_CT_THIT", 0.03),
    "a": ("FREETOKEN_CT_A", 0.11),
    "b": ("FREETOKEN_CT_B", 0.20),
    "tok": ("FREETOKEN_CT_TOK", 0.35),
    "maxn": ("FREETOKEN_CT_MAXN", 384),
}


def _env_cost() -> dict[str, float]:
    out: dict[str, float] = {}
    for key, (env, default) in _COST_ENV.items():
        raw = os.environ.get(env)
        out[key] = float(raw) if raw is not None else float(default)
    return out


# do_not_specialize: Triton folds int args == 1 into constexpr at compile
# time, and the kernel body calls .to() on layer_id/bsz -- layer 1 and
# single-request decode (bsz=1) are the common production cases.
@triton.jit(do_not_specialize=["layer_id", "bsz"])
def _ct_split_kernel(
    slots_ptr,  # [bsz*K] int, in/out: claimed entries point at slot 0
    weights_ptr,  # [bsz*K] fp32, in/out: claimed entries become 0
    hidden_ptr,  # [bsz, H] input activations (bf16/fp16)
    slot_for_id_ptr,  # [E] int32 (this layer's row of the [L,E] table)
    id_of_slot_ptr,  # [cache_size] int32 (flat layer*E+expert ids)
    row_map_ptr,  # [E] int32 (this layer's pin row map; -1 = not pinned)
    src_indices_ptr,  # [PLAN] int32, staged miss LOCAL expert ids (in/out)
    evict_slots_ptr,  # [PLAN] int32, staged victim slots (in/out)
    num_indices_ptr,  # [1] int64, staged miss count (in/out)
    mark_ptr,  # [PLAN] int32 scratch: slot -> (staged ordinal+1), sign = cpu_ok
    cnt_ptr,  # [PLAN] int32 scratch: routed entries per staged ordinal
    order_ptr,  # [PLAN] int32 scratch: selection-sorted staged ordinals
    ctrl_dev_ptr,  # [8] int64 device staging: [0]=layer [1]=bsz [2]=npk [7]=seq
    seqc_ptr,  # [1] int64 device counter (graph replay bumps it -> new seq)
    hx_dev_ptr,  # [max_tokens, H] fp16 device staging
    picks_dev_ptr,  # [max_picks*3] int32 device staging (tok, row, w-bits)
    stats_ptr,  # [L*16] int64: [0] hits [1] misses [2] picks [3] calls
    route_counts_ptr,  # [L*E] int64: per-expert routed entries (item ⑤ admission signal)
    miss_counts_ptr,  # [L*E] int64: per-expert miss entries (item ⑤ admission signal)
    ram_rows,
    E,  # local experts per layer (route/miss counter row width)
    layer_id,
    bsz,
    K: tl.constexpr,
    H: tl.constexpr,
    PLAN,
    TZC,
    THIT,
    CA,
    CB,
    TOK,
    MAXN,
    FORCE_N,
    REUSE_MIN,
    XSTEP_W,
    BLOCK_H: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    # ---- lanes: stage hidden states to fp16 device staging + zero scratch ----
    h_off = tl.arange(0, BLOCK_H)
    for tok in range(0, bsz):
        src = hidden_ptr + tok.to(tl.int64) * H
        dst = hx_dev_ptr + tok.to(tl.int64) * H
        for h0 in range(0, H, BLOCK_H):
            h = h0 + h_off
            hm = h < H
            v = tl.load(src + h, mask=hm, other=0.0)
            tl.store(dst + h, v.to(tl.float16), mask=hm)
    p_off = tl.arange(0, BLOCK_P)
    for p0 in range(0, PLAN, BLOCK_P):
        p = p0 + p_off
        pm = p < PLAN
        tl.store(mark_ptr + p, tl.zeros([BLOCK_P], dtype=tl.int32), mask=pm)
        tl.store(cnt_ptr + p, tl.zeros([BLOCK_P], dtype=tl.int32), mask=pm)
    tl.debug_barrier()  # scratch visible to the scalar loops below (same block)

    # ---- scalar pass 1: mark staged slots, count routed entries per expert ----
    nm64 = tl.load(num_indices_ptr)
    nm = nm64.to(tl.int32)
    for r in range(0, nm):
        e = tl.load(src_indices_ptr + r)
        s = tl.load(evict_slots_ptr + r)
        row = tl.load(row_map_ptr + e)
        ok = (row >= 0) & (row < ram_rows)
        tl.store(mark_ptr + s, tl.where(ok, r + 1, -(r + 1)).to(tl.int32))
    nh = 0
    n_miss_entries = 0
    # Per-expert counters: scalar RMW inside this single-block kernel, same
    # determinism contract as cnt_ptr above -- one writer, replay-stable, no
    # atomics on the output. Every routed entry increments route[e]; misses
    # additionally increment miss[e]. Invariants: sum(route) == nh +
    # n_miss_entries == bsz*K, sum(miss) == n_miss_entries.
    rc_base = route_counts_ptr + layer_id.to(tl.int64) * E
    mc_base = miss_counts_ptr + layer_id.to(tl.int64) * E
    for i in range(0, bsz * K):
        slot = tl.load(slots_ptr + i)
        mk = tl.load(mark_ptr + slot)
        if mk == 0:
            nh += 1
            e_hit = tl.load(id_of_slot_ptr + slot) - layer_id * E
            if (e_hit >= 0) & (e_hit < E):  # defensive: never scribble OOB
                tl.store(rc_base + e_hit, tl.load(rc_base + e_hit) + 1)
        else:
            n_miss_entries += 1
            r = tl.where(mk > 0, mk - 1, -mk - 1)
            c = tl.load(cnt_ptr + r)
            tl.store(cnt_ptr + r, c + 1)
            e_miss = tl.load(src_indices_ptr + r)
            tl.store(rc_base + e_miss, tl.load(rc_base + e_miss) + 1)
            tl.store(mc_base + e_miss, tl.load(mc_base + e_miss) + 1)

    # ---- reuse filter (20261007 rescue piece C): demote low-reuse CPU-ok ----
    # misses to the fetch path. Flipping the mark sign keeps the entry staged
    # (pass 5 compacts mk != 0) but invisible to the sort below (mk > 0), so
    # the doorbell fetches it like any other GPU-bound miss. FORCE_N bypasses
    # the filter: it is the calibration knob and must stay comparable across
    # reuse settings. REUSE_MIN < 0 compiles to a no-op (off == old policy).
    # freq uses the counters BEFORE this step's increments (pass 1 already
    # added this step's cnt; stats calls is only bumped at the end below).
    if (REUSE_MIN >= 0.0) & (FORCE_N < 0):
        calls_pre = tl.load(stats_ptr + layer_id.to(tl.int64) * 16 + 3)
        denom = tl.maximum(calls_pre, 1).to(tl.float32)
        for r in range(0, nm):
            s = tl.load(evict_slots_ptr + r)
            mk = tl.load(mark_ptr + s)
            if mk > 0:
                c = tl.load(cnt_ptr + r)
                e = tl.load(src_indices_ptr + r)
                rc_pre = tl.load(rc_base + e) - c.to(tl.int64)
                score = c.to(tl.float32) + XSTEP_W * (rc_pre.to(tl.float32) / denom)
                if score < REUSE_MIN:
                    tl.store(mark_ptr + s, -mk)

    # ---- scalar pass 2: order CPU-ok experts by (count asc, ordinal asc) ----
    # selection via "smallest key strictly greater than the previous" (keys are
    # unique in r, so this is a total order -- no tie-tracking needed).
    ncpu = 0
    for r in range(0, nm):
        s = tl.load(evict_slots_ptr + r)
        mk = tl.load(mark_ptr + s)
        if mk > 0:
            ncpu += 1
    prev_c = -1
    prev_r = -1
    for i in range(0, ncpu):
        best_c = _BIG
        best_r = _BIG
        for r in range(0, nm):
            s = tl.load(evict_slots_ptr + r)
            mk = tl.load(mark_ptr + s)
            if mk > 0:
                c = tl.load(cnt_ptr + r)
                after_prev = (c > prev_c) | ((c == prev_c) & (r > prev_r))
                before_best = (c < best_c) | ((c == best_c) & (r < best_r))
                if after_prev & before_best:
                    best_c = c
                    best_r = r
        tl.store(order_ptr + i, best_r)
        prev_c = best_c
        prev_r = best_r

    # ---- scalar pass 3: cost-model scan over the sorted prefix (dsv41 tt) ----
    npk = 0
    if FORCE_N >= 0:
        npk = tl.minimum(FORCE_N, ncpu)
    else:
        tt = 0.0
        for k in range(0, MAXN + 1):
            if k <= ncpu:
                gpu_ms = THIT * (nh + n_miss_entries - k) + TZC * (n_miss_entries - k)
                cpu_ms = CA
                for i in range(0, k):
                    r = tl.load(order_ptr + i)
                    c = tl.load(cnt_ptr + r)
                    cpu_ms += CB * (1.0 + TOK * (c - 1))
                m = tl.maximum(gpu_ms, cpu_ms)
                if (k == 0) | (m < tt):
                    tt = m
                    npk = k
        if tt <= 0.0:
            npk = 0

    # ---- scalar pass 4: claim the chosen experts, emit per-pair picks --------
    # claimed expert: every routed entry at its slot gets slot 0 + weight 0
    # (GPU GEMM contributes exactly 0), the slot reservation is undone, and one
    # pick per (token, expert) pair is appended in routed order.
    # B9 (20261007 battery follow-up): the old expert-major sweep rewrote a
    # claimed entry's slot to 0 BEFORE later claims re-scanned the entries, so
    # any later-claimed expert whose VICTIM slot was 0 re-matched those entries
    # and emitted duplicate weight-0 picks (numerically inert, but each cost a
    # full 22MB DRAM expert read in the service). Two-phase now: flag the
    # claimed slots first, then ONE routed sweep emits each pick exactly once.
    # cnt_ptr is re-purposed as the slot -> (pin row + 1) map: passes 1-3 have
    # consumed the per-ordinal counts, and the lane zero-pass below resets the
    # slot domain (slots and ordinals share the index range < PLAN).
    for p0 in range(0, PLAN, BLOCK_P):
        p = p0 + p_off
        pm = p < PLAN
        tl.store(cnt_ptr + p, tl.zeros([BLOCK_P], dtype=tl.int32), mask=pm)
    tl.debug_barrier()
    for j in range(0, npk):
        r = tl.load(order_ptr + j)
        e = tl.load(src_indices_ptr + r)
        s = tl.load(evict_slots_ptr + r)
        row = tl.load(row_map_ptr + e)
        tl.store(cnt_ptr + s, row + 1)  # rows >= 0, so 0 means "not claimed"
    w_out = 0
    for i in range(0, bsz * K):
        slot = tl.load(slots_ptr + i)
        fl = tl.load(cnt_ptr + slot)
        if fl > 0:
            wgt = tl.load(weights_ptr + i).to(tl.float32)
            tl.store(picks_dev_ptr + w_out * 3 + 0, i // K)
            tl.store(picks_dev_ptr + w_out * 3 + 1, fl - 1)
            tl.store(picks_dev_ptr + w_out * 3 + 2, wgt.to(tl.int32, bitcast=True))
            w_out += 1
            tl.store(slots_ptr + i, slot * 0)  # slot 0 sentinel, dtype-preserving
            tl.store(weights_ptr + i, 0.0)
    for j in range(0, npk):
        r = tl.load(order_ptr + j)
        e = tl.load(src_indices_ptr + r)
        s = tl.load(evict_slots_ptr + r)
        tl.store(slot_for_id_ptr + e, -1)
        tl.store(id_of_slot_ptr + s, -1)
        tl.store(mark_ptr + s, 0)  # claimed: drop from the staged plan below

    # ---- scalar pass 5: compact the staged plan (fetch path sees the rest) ---
    w = 0
    for r in range(0, nm):
        s = tl.load(evict_slots_ptr + r)
        mk = tl.load(mark_ptr + s)
        if mk != 0:
            e = tl.load(src_indices_ptr + r)
            tl.store(src_indices_ptr + w, e)
            tl.store(evict_slots_ptr + w, s)
            w += 1
    tl.store(num_indices_ptr, w.to(tl.int64))

    # ---- publish: header fields; seq handled by the D2H memcpy ordering ------
    tl.store(ctrl_dev_ptr + 0, layer_id.to(tl.int64))
    tl.store(ctrl_dev_ptr + 1, bsz.to(tl.int64))
    tl.store(ctrl_dev_ptr + 2, w_out.to(tl.int64))
    seq = tl.load(seqc_ptr, volatile=True)
    if w_out > 0:
        seq = seq + 1
        tl.store(seqc_ptr, seq)
    tl.store(ctrl_dev_ptr + _SEQ_OFF, seq)

    # ---- per-layer routing aggregates (observability only; item ⑤'s -------
    # admission signal is the per-expert counters recorded in pass 1) --------
    sb = stats_ptr + layer_id.to(tl.int64) * 16
    tl.store(sb + 0, tl.load(sb + 0) + nh.to(tl.int64))
    tl.store(sb + 1, tl.load(sb + 1) + n_miss_entries.to(tl.int64))
    tl.store(sb + 2, tl.load(sb + 2) + w_out.to(tl.int64))
    tl.store(sb + 3, tl.load(sb + 3) + 1)


@triton.jit(do_not_specialize=["layer_id", "bsz"])
def _ct_split_pre_kernel(
    ids_ptr,  # [bsz*K] int32 raw expert ids (READ ONLY; ensure rewrites later)
    weights_ptr,  # [bsz*K] fp32, in/out: claimed entries become 0
    hidden_ptr,  # [bsz, H] input activations (bf16/fp16)
    slot_for_id_ptr,  # [E] int32 (this layer's row), in/out: claimed -> sentinel 0
    row_map_ptr,  # [E] int32 (this layer's pin row map; -1 = not pinned)
    mark_ptr,  # [PLAN] int32 scratch: expert -> 1 = CPU-ok candidate
    cnt_ptr,  # [PLAN] int32 scratch: routed entries per candidate
    order_ptr,  # [PLAN] int32 scratch: selection-sorted candidate expert ids
    gpum_ptr,  # [PLAN] int32 scratch: expert -> 1 = distinct GPU-bound miss
    claimed_ptr,  # [PLAN] int32 out: claimed expert ids (combine restores)
    ctrl_dev_ptr,  # [8] int64 device staging: [0]=layer [1]=bsz [2]=npk [3]=nclaimed [7]=seq
    seqc_ptr,  # [1] int64 device counter (graph replay bumps it -> new seq)
    hx_dev_ptr,  # [max_tokens, H] fp16 device staging
    picks_dev_ptr,  # [max_picks*3] int32 device staging (tok, row, w-bits)
    stats_ptr,  # [L*16] int64: [0] hits [1] misses [2] picks [3] calls
    route_counts_ptr,  # [L*E] int64: per-expert routed entries
    miss_counts_ptr,  # [L*E] int64: per-expert miss entries
    ram_rows,
    E,  # local experts per layer (route/miss counter row width)
    layer_id,
    bsz,
    K: tl.constexpr,
    H: tl.constexpr,
    PLAN,
    REUSE_MIN,
    XSTEP_W,
    TZC,
    THIT,
    CA,
    CB,
    TOK,
    MAXN,
    FORCE_N,
    BLOCK_H: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    """Claim-BEFORE-ensure split (20261007 rescue pieces A+B+C).

    Runs on RAW routing, before ``lru_ensure``: a claimed expert's
    ``slot_for_id[e]`` is set to the slot-0 sentinel, so ensure classifies it
    as a hit -- no victim eviction and no fetch is ever staged for it (piece
    B, the churn fix) -- and the CPU job is published before
    ensure+fetch+GEMM instead of fetch+GEMM alone, widening the overlap
    window (piece A). Claimed entries' weights are zeroed here (ensure never
    touches weights), so after ensure rewrites their ids to slot 0 the GPU
    GEMM contributes exactly 0 for them. The combine kernel restores
    ``slot_for_id[e] = -1`` from ``claimed_ptr``.

    Sentinels bump ``usage[0]`` once per claimed expert per step, so slot 0's
    resident is effectively pinned -- a deliberate 1/cache_size capacity tax,
    documented in the design doc. Determinism contract identical to the
    legacy kernel: one block, scalar ordered loops, no atomics.
    """
    # ---- lanes: stage hidden states to fp16 device staging + zero scratch ----
    h_off = tl.arange(0, BLOCK_H)
    for tok in range(0, bsz):
        src = hidden_ptr + tok.to(tl.int64) * H
        dst = hx_dev_ptr + tok.to(tl.int64) * H
        for h0 in range(0, H, BLOCK_H):
            h = h0 + h_off
            hm = h < H
            v = tl.load(src + h, mask=hm, other=0.0)
            tl.store(dst + h, v.to(tl.float16), mask=hm)
    p_off = tl.arange(0, BLOCK_P)
    for p0 in range(0, PLAN, BLOCK_P):
        p = p0 + p_off
        pm = p < PLAN
        tl.store(mark_ptr + p, tl.zeros([BLOCK_P], dtype=tl.int32), mask=pm)
        tl.store(cnt_ptr + p, tl.zeros([BLOCK_P], dtype=tl.int32), mask=pm)
        tl.store(gpum_ptr + p, tl.zeros([BLOCK_P], dtype=tl.int32), mask=pm)
    tl.debug_barrier()  # scratch visible to the scalar loops below (same block)

    # ---- scalar pass A: classify raw routing, count per-expert entries ------
    # Candidates are keyed by EXPERT ID (not staged ordinal): the legacy
    # kernel's staged order is ascending expert id (lru_ensure ranks misses by
    # id), so (count asc, expert asc) reproduces its total order exactly.
    rc_base = route_counts_ptr + layer_id.to(tl.int64) * E
    mc_base = miss_counts_ptr + layer_id.to(tl.int64) * E
    nh = 0
    n_miss_entries = 0
    for i in range(0, bsz * K):
        e = tl.load(ids_ptr + i)
        s = tl.load(slot_for_id_ptr + e)
        tl.store(rc_base + e, tl.load(rc_base + e) + 1)
        if s >= 0:
            nh += 1
        else:
            n_miss_entries += 1
            tl.store(mc_base + e, tl.load(mc_base + e) + 1)
            row = tl.load(row_map_ptr + e)
            ok = (row >= 0) & (row < ram_rows)
            if ok:
                tl.store(mark_ptr + e, 1)
                tl.store(cnt_ptr + e, tl.load(cnt_ptr + e) + 1)
            else:
                tl.store(gpum_ptr + e, 1)

    # ---- reuse filter (piece C): demote low-reuse candidates to GPU-bound ---
    # Same semantics as the legacy kernel's filter: FORCE_N bypasses it;
    # freq uses pre-step counters. Demoted experts are simply not candidates
    # -- ensure stages them as ordinary misses (the pre-rescue behaviour for
    # every miss).
    if (REUSE_MIN >= 0.0) & (FORCE_N < 0):
        calls_pre = tl.load(stats_ptr + layer_id.to(tl.int64) * 16 + 3)
        denom = tl.maximum(calls_pre, 1).to(tl.float32)
        for e in range(0, E):
            mk = tl.load(mark_ptr + e)
            if mk == 1:
                c = tl.load(cnt_ptr + e)
                rc_pre = tl.load(rc_base + e) - c.to(tl.int64)
                score = c.to(tl.float32) + XSTEP_W * (rc_pre.to(tl.float32) / denom)
                if score < REUSE_MIN:
                    tl.store(mark_ptr + e, 0)
                    tl.store(gpum_ptr + e, 1)

    # ---- scalar pass B: order candidates by (count asc, expert asc) ---------
    ncpu = 0
    for e in range(0, E):
        if tl.load(mark_ptr + e) == 1:
            ncpu += 1
    prev_c = -1
    prev_e = -1
    for i in range(0, ncpu):
        best_c = _BIG
        best_e = _BIG
        for e in range(0, E):
            mk = tl.load(mark_ptr + e)
            if mk == 1:
                c = tl.load(cnt_ptr + e)
                after_prev = (c > prev_c) | ((c == prev_c) & (e > prev_e))
                before_best = (c < best_c) | ((c == best_c) & (e < best_e))
                if after_prev & before_best:
                    best_c = c
                    best_e = e
        tl.store(order_ptr + i, best_e)
        prev_c = best_c
        prev_e = best_e

    # ---- scalar pass C: cost-model scan over the sorted prefix (dsv41 tt) ---
    npk = 0
    if FORCE_N >= 0:
        npk = tl.minimum(FORCE_N, ncpu)
    else:
        tt = 0.0
        for k in range(0, MAXN + 1):
            if k <= ncpu:
                gpu_ms = THIT * (nh + n_miss_entries - k) + TZC * (n_miss_entries - k)
                cpu_ms = CA
                for i in range(0, k):
                    e = tl.load(order_ptr + i)
                    c = tl.load(cnt_ptr + e)
                    cpu_ms += CB * (1.0 + TOK * (c - 1))
                m = tl.maximum(gpu_ms, cpu_ms)
                if (k == 0) | (m < tt):
                    tt = m
                    npk = k
        if tt <= 0.0:
            npk = 0

    # ---- scalar pass D: claim the chosen experts, emit per-pair picks -------
    # sentinel FIRST (per claimed expert, once), then its routed entries in
    # routed order: weight -> 0 (GEMM contributes exactly 0) and one pick per
    # (token, expert) pair. ids_ptr is never written.
    w_out = 0
    for j in range(0, npk):
        e = tl.load(order_ptr + j)
        row = tl.load(row_map_ptr + e)
        tl.store(slot_for_id_ptr + e, e * 0)  # slot-0 sentinel, dtype-preserving
        tl.store(claimed_ptr + j, e)
        for i in range(0, bsz * K):
            ei = tl.load(ids_ptr + i)
            if ei == e:
                wgt = tl.load(weights_ptr + i).to(tl.float32)
                tl.store(picks_dev_ptr + w_out * 3 + 0, i // K)
                tl.store(picks_dev_ptr + w_out * 3 + 1, row)
                tl.store(picks_dev_ptr + w_out * 3 + 2, wgt.to(tl.int32, bitcast=True))
                w_out += 1
                tl.store(weights_ptr + i, 0.0)

    # ---- publish: header fields; seq handled by the D2H memcpy ordering -----
    tl.store(ctrl_dev_ptr + 0, layer_id.to(tl.int64))
    tl.store(ctrl_dev_ptr + 1, bsz.to(tl.int64))
    tl.store(ctrl_dev_ptr + 2, w_out.to(tl.int64))
    tl.store(ctrl_dev_ptr + _CTRL_NCLAIMED, npk.to(tl.int64))
    seq = tl.load(seqc_ptr, volatile=True)
    if w_out > 0:
        seq = seq + 1
        tl.store(seqc_ptr, seq)
    tl.store(ctrl_dev_ptr + _SEQ_OFF, seq)

    # ---- per-layer routing aggregates (same [L,16] contract as legacy) ------
    sb = stats_ptr + layer_id.to(tl.int64) * 16
    tl.store(sb + 0, tl.load(sb + 0) + nh.to(tl.int64))
    tl.store(sb + 1, tl.load(sb + 1) + n_miss_entries.to(tl.int64))
    tl.store(sb + 2, tl.load(sb + 2) + w_out.to(tl.int64))
    tl.store(sb + 3, tl.load(sb + 3) + 1)


@triton.jit
def _ct_combine_kernel(
    out_ptr,  # [bsz, H] GPU GEMM output, in/out
    hout_ptr,  # [max_tokens, H] fp32 pinned host partial sums (PCIe reads)
    done_ptr,  # [1] int64 pinned host flag, CPU-written
    dflag_ptr,  # [8] int64 device: [0]=target [1]=wait_ns [2]=timeouts
    ctrl_dev_ptr,  # [8] int64 device staging: [2]=npk [3]=nclaimed [7]=target seq
    claimed_ptr,  # [PLAN] int32: claimed expert ids to restore (pre-ensure path)
    slot_for_id_ptr,  # [E] int32: this layer's row (sentinel restore target)
    bsz,
    H: tl.constexpr,
    TIMEOUT_ITERS,
    BLOCK_H: tl.constexpr,
):
    npk = tl.load(ctrl_dev_ptr + 2)
    if npk <= 0:
        return
    # Rescue pieces A+B: restore the pre-ensure slot-0 sentinels
    # (slot_for_id[e] = 0 made claimed experts pseudo-hits so lru_ensure
    # staged no eviction/fetch for them). Stream-ordered after this layer's
    # ensure and GEMM; the next reader is next step's ensure for this layer.
    # nclaimed == 0 on the legacy path (ctrl[3] zero-initialised, never
    # written by the legacy split kernel), so this loop is a no-op there.
    ncl = tl.load(ctrl_dev_ptr + _CTRL_NCLAIMED)
    for j in range(0, ncl):
        e = tl.load(claimed_ptr + j)
        tl.store(slot_for_id_ptr + e, -1)
    target = tl.load(ctrl_dev_ptr + _SEQ_OFF)
    # Publish the target FIRST (graph_fetch's ordering: a torn sequence can only
    # show up as a timeout, never as a silently accepted stale frame).
    tl.store(dflag_ptr, target)
    tl.debug_barrier()
    # Piece D: real wall-clock spin time via %globaltimer (sm_70+). The old
    # accounting assumed ~5ns/iter while a sys-scope host atomic costs ~1.5us,
    # a ~300x underestimate of the GPU wait.
    t0 = tl.sum(
        tl.inline_asm_elementwise(
            "mov.u64 $0, %globaltimer;", "=l,l",
            [tl.full([1], 0, tl.int64)], dtype=tl.int64, is_pure=False, pack=1
        )
    )
    iters = 0
    done = tl.atomic_add(done_ptr, 0, sem="acquire", scope="sys")
    while done < target:
        # Never give up: the weights were never fetched, so returning without
        # the CPU partials would be a SILENT wrong output. Keep spinning (the
        # FATAL contract -- a hang is loud) and count the budget expiries.
        iters += 1
        if (TIMEOUT_ITERS > 0) & (iters % TIMEOUT_ITERS == 0):
            tl.store(dflag_ptr + _DFLAG_TIMEOUTS, tl.load(dflag_ptr + _DFLAG_TIMEOUTS) + 1)
        done = tl.atomic_add(done_ptr, 0, sem="acquire", scope="sys")
    t1 = tl.sum(
        tl.inline_asm_elementwise(
            "mov.u64 $0, %globaltimer;", "=l,l",
            [tl.full([1], 0, tl.int64)], dtype=tl.int64, is_pure=False, pack=1
        )
    )
    tl.store(dflag_ptr + _DFLAG_WAIT_NS,
             tl.load(dflag_ptr + _DFLAG_WAIT_NS) + (t1 - t0))
    h_off = tl.arange(0, BLOCK_H)
    for tok in range(0, bsz):
        dst = out_ptr + tok.to(tl.int64) * H
        srcp = hout_ptr + tok.to(tl.int64) * H
        for h0 in range(0, H, BLOCK_H):
            h = h0 + h_off
            hm = h < H
            v = tl.load(dst + h, mask=hm, other=0.0)
            w = tl.load(srcp + h, mask=hm, other=0.0)
            tl.store(dst + h, v + w.to(v.dtype), mask=hm)


def cost_select(entries: list[tuple[int, int]], nh: int, nm: int,
                cost: dict[str, float], force_n: int = -1,
                reuse_scores: list[float] | None = None,
                reuse_min: float = -1.0) -> int:
    """Python reference of the split kernel's passes 2-3 (dsv41 cost selection).

    ``entries`` holds one ``(duplicate_count, staged_ordinal)`` per CPU-ok miss
    expert in staged order; the kernel orders them by ``(count asc, ordinal
    asc)`` and picks the prefix ``k`` minimizing ``tt = max(gpu(k), cpu(k))``.
    Returns the number of experts to serve on the CPU (0 = keep the fetch path).
    Production runs the Triton kernel; this mirror exists for the pure-CPU
    accounting tests and for calibration tooling.

    Piece C (20261007 rescue): ``reuse_scores`` carries one score per entry
    (see :func:`reuse_score`); with ``reuse_min >= 0`` entries scoring below
    the threshold are demoted to the fetch path BEFORE ordering. ``force_n``
    bypasses the filter (calibration knob comparability), mirroring the
    kernel's ``(REUSE_MIN >= 0) & (FORCE_N < 0)`` guard.
    """
    if force_n < 0 and reuse_scores is not None and reuse_min >= 0.0:
        entries = [e for i, e in enumerate(entries) if reuse_scores[i] >= reuse_min]
    order = sorted(range(len(entries)), key=lambda i: (entries[i][0], entries[i][1]))
    ncpu = len(order)
    if force_n >= 0:
        return min(force_n, ncpu)
    maxn = min(int(cost["maxn"]), ncpu)
    tt = 0.0
    npk = 0
    for k in range(0, maxn + 1):
        gpu_ms = cost["thit"] * (nh + nm - k) + cost["tzc"] * (nm - k)
        cpu_ms = cost["a"] + sum(
            cost["b"] * (1.0 + cost["tok"] * (entries[order[i]][0] - 1)) for i in range(k)
        )
        m = max(gpu_ms, cpu_ms)
        if k == 0 or m < tt:
            tt = m
            npk = k
    return npk if tt > 0 else 0


def reuse_score(cnt: int, route_count_pre: int, calls_pre: int,
                xstep_w: float = 1.0) -> float:
    """Piece C reuse estimate: this-step duplicates + w * cross-step frequency.

    ``route_count_pre`` is the expert's cumulative routed-entry count BEFORE
    this step and ``calls_pre`` the layer's split-call count before this step,
    so ``route_count_pre / max(calls_pre, 1)`` is the per-step routing rate the
    [L,E] counters encode. The split kernels derive both from their own
    counters (subtracting this step's contribution, which pass 1/A has
    already added); the value is identical on both the legacy and the
    pre-ensure path.
    """
    return cnt + xstep_w * (route_count_pre / max(calls_pre, 1))


def pre_ensure_claim_plan(ids: list[int], K: int, slot_for_id_row: dict[int, int],
                          row_map_row: dict[int, int], route_counts_pre: dict[int, int],
                          calls_pre: int, cost: dict[str, float], ram_rows: int,
                          reuse_min: float = -1.0, xstep_w: float = 1.0,
                          force_n: int = -1) -> dict:
    """Pure-python mirror of ``_ct_split_pre_kernel``'s decision + bookkeeping.

    Reproduces the kernel's passes on raw routing: classify (hit / CPU-ok
    candidate / GPU-bound), piece-C reuse demotion (skipped under force_n),
    (count asc, expert asc) ordering, dsv41 cost selection, then the claim
    side effects. Returns a dict with ``claimed`` (expert ids in selection
    order), ``sentinels`` (same set -- the kernel writes slot_for_id[e] = 0),
    ``weight_zeroed`` (entry indices whose weight becomes 0), ``picks``
    ((token, row) per (token, expert) pair in publish order), ``demoted``
    (reuse-filtered candidates, ascending), ``nh``/``n_miss``/``ncpu``/
    ``npk``. Exists so the A+B+C bookkeeping is testable on pure CPU.
    """
    nh = 0
    cnt: dict[int, int] = {}
    for e in ids:
        if slot_for_id_row[e] >= 0:
            nh += 1
        else:
            row = row_map_row[e]
            if 0 <= row < ram_rows:
                cnt[e] = cnt.get(e, 0) + 1
    n_miss = len(ids) - nh
    demoted: set[int] = set()
    if force_n < 0 and reuse_min >= 0.0:
        for e in list(cnt):
            if reuse_score(cnt[e], route_counts_pre.get(e, 0), calls_pre, xstep_w) < reuse_min:
                demoted.add(e)
                del cnt[e]
    order = sorted(cnt, key=lambda e: (cnt[e], e))
    npk = cost_select([(cnt[e], e) for e in order], nh, n_miss, cost, force_n=force_n)
    claimed = order[:npk]
    picks: list[tuple[int, int]] = []
    weight_zeroed: list[int] = []
    for e in claimed:
        row = row_map_row[e]
        for i, ei in enumerate(ids):
            if ei == e:
                picks.append((i // K, row))
                weight_zeroed.append(i)
    return {
        "claimed": claimed,
        "sentinels": list(claimed),
        "weight_zeroed": weight_zeroed,
        "picks": picks,
        "demoted": sorted(demoted),
        "nh": nh,
        "n_miss": n_miss,
        "ncpu": len(order),
        "npk": npk,
    }


def route_count_mirror(slots: list[int], evict_slots: list[int], src_indices: list[int],
                       row_map: list[int], ram_rows: int, id_of_slot: list[int],
                       cache_size: int, layer_id: int, num_experts: int
                       ) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Python mirror of the split kernel's per-expert counting (scalar pass 1).

    Reproduces the mark loop and the entry walk exactly: ``mark[s]`` holds
    ``staged ordinal + 1`` for a CPU-ok staged miss and ``-(ordinal + 1)`` for
    a GPU-bound one; an entry whose slot is unmarked is a hit whose expert is
    ``id_of_slot[slot]`` (flat ``layer*E + expert``). Returns
    ``(route_counts[E], miss_counts[E], nh, n_miss)``. The kernel keeps the
    invariants ``route.sum() == nh + n_miss == len(slots)`` and
    ``miss.sum() == n_miss`` by construction. Production counting runs in the
    Triton kernel (the GPU battery cross-checks it against this mirror); the
    mirror exists so the accounting is testable on pure CPU.
    """
    mark = [0] * cache_size
    for r in range(len(src_indices)):
        s = evict_slots[r]
        row = row_map[src_indices[r]]
        mark[s] = (r + 1) if 0 <= row < ram_rows else -(r + 1)
    route = torch.zeros(num_experts, dtype=torch.int64)
    miss = torch.zeros(num_experts, dtype=torch.int64)
    nh = 0
    n_miss = 0
    for slot in slots:
        mk = mark[slot]
        if mk == 0:
            nh += 1
            route[id_of_slot[slot] - layer_id * num_experts] += 1
        else:
            n_miss += 1
            r = mk - 1 if mk > 0 else -mk - 1
            e = src_indices[r]
            route[e] += 1
            miss[e] += 1
    return route, miss, nh, n_miss


class CpuTier:
    """Protocol owner for the RAM-resident miss tier (one per MoE cache).

    Lifecycle: ``attach()`` after the disk tier is attached (the tier needs its
    pin row map); the C++ worker pool starts lazily on the first ``split``;
    ``shutdown()`` at engine teardown. ``split`` runs between
    ``cache.ensure_experts`` and ``cache.copy_missing``; ``combine`` adds the
    CPU partials onto the GPU GEMM output of the same layer.
    """

    def __init__(self, cache, disk_tier, threads: int = 0, timeout_ns: int | None = None):
        self._cache = cache
        self._disk = disk_tier
        self._threads = threads
        self._timeout_ns = int(
            os.environ.get("FREETOKEN_CT_TIMEOUT_NS", _DEFAULT_TIMEOUT_NS)
            if timeout_ns is None else timeout_ns
        )
        self._cost = _env_cost()
        self._force_n = int(os.environ.get("FREETOKEN_CT_FORCE_N", "-1"))
        # 20261007 rescue pieces (design doc: notes/engines/20261007-cpu-tier-rescue-design.md)
        self._pre_ensure = os.environ.get("FREETOKEN_CT_PRE_ENSURE", "0").strip() == "1"
        self._reuse_min = float(os.environ.get("FREETOKEN_CT_REUSE_MIN", "-1"))
        self._xstep_w = float(os.environ.get("FREETOKEN_CT_XSTEP_W", "1.0"))
        self._lock = threading.Lock()
        self._service = None
        self._started = False
        self._built = False

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    def attach(self, top_k: int) -> None:
        """Register the pin row map and build the protocol buffers.

        ``top_k`` sizes the pick capacity (max_tokens * top_k pairs); the hidden
        and intermediate sizes come from the bank shapes.
        """
        cache = self._cache
        disk = self._disk
        assert cache is not None and disk is not None
        self._num_layers = int(cache.num_layers)
        self._num_experts = int(cache.num_experts)
        self._ram_rows = int(disk._ram)
        self._row_map = disk._row_map_dev  # [L, local_num] int32 device
        max_bs = int(getattr(cache, "cuda_graph_max_bs", 0) or 0)
        self._max_tokens = max(max_bs, 64)
        self._top_k = max(int(top_k), 1)
        self._max_picks = self._max_tokens * self._top_k
        plan = int(cache.src_indices.numel())
        # bank_sources may be keyed by canonical role ("gate_up") or by schema
        # name ("gate_up_packed") depending on the registration path -- accept both.
        from freetoken.moe.legacy_format import canonical_role

        banks_by_role = {canonical_role(n): v for n, v in cache.bank_sources.items()}
        if "gate_up" not in banks_by_role:
            raise KeyError(
                f"gate_up: cpu tier needs nvfp4 banks, have {sorted(cache.bank_sources)} "
                f"(quant_format={getattr(cache, 'quant_format', None)!r})")
        gu0 = banks_by_role["gate_up"][0]
        self._hidden = int(gu0.shape[2] * 2)
        dev = cache.slot_for_id.device
        self._ctrl_host = alloc_pinned_tensor(_CTRL_LEN, dtype=torch.int64)
        self._ctrl_host.zero_()
        # fp16 on the wire (matches the device staging so the D2H pull is a plain
        # memcpy -- a dtype-converting copy_ is not); the service converts to
        # bf16 through f32 on prep (exact).
        self._hx_host = alloc_pinned_tensor(self._max_tokens, self._hidden, dtype=torch.float16)
        self._picks_host = alloc_pinned_tensor(self._max_picks, 3, dtype=torch.int32)
        self._hout = alloc_pinned_tensor(self._max_tokens, self._hidden, dtype=torch.float32)
        self._done_host = alloc_pinned_tensor(1, dtype=torch.int64)
        self._done_host.zero_()
        with torch.inference_mode(False):
            self._ctrl_dev = torch.zeros(_CTRL_LEN, dtype=torch.int64, device=dev)
            self._hx_dev = torch.zeros(self._max_tokens, self._hidden, dtype=torch.float16, device=dev)
            self._picks_dev = torch.zeros(self._max_picks, 3, dtype=torch.int32, device=dev)
            self._seqc = torch.zeros(1, dtype=torch.int64, device=dev)
            self._dflag = torch.zeros(8, dtype=torch.int64, device=dev)
            self._stats_dev = torch.zeros(cache.num_layers * 16, dtype=torch.int64, device=dev)
            # Item ⑤'s admission signal (port/m4-perexpert-counts): cumulative
            # [L,E] per-expert route/miss counters. The split kernel writes them
            # during graph replay (device memory, so replay writes are real);
            # host reads are on-demand, so no per-replay D2H mirror is added.
            self._route_counts_dev = torch.zeros(cache.num_layers * cache.num_experts,
                                                 dtype=torch.int64, device=dev)
            self._miss_counts_dev = torch.zeros(cache.num_layers * cache.num_experts,
                                                dtype=torch.int64, device=dev)
            self._mark = torch.zeros(plan, dtype=torch.int32, device=dev)
            self._cnt = torch.zeros(plan, dtype=torch.int32, device=dev)
            self._order = torch.zeros(plan, dtype=torch.int32, device=dev)
            # Pre-ensure path (rescue A+B): GPU-bound-miss dedup marker and the
            # claimed-expert list the combine kernel restores sentinels from.
            self._gpum = torch.zeros(plan, dtype=torch.int32, device=dev)
            self._claimed_dev = torch.zeros(plan, dtype=torch.int32, device=dev)
            # Slot row of the layer currently between split and combine (the
            # combine kernel's sentinel-restore target; layers strictly
            # alternate split -> combine on one stream).
            self._cur_slot_row = cache.slot_for_id[0]
        self._counter_snapshot = (
            torch.zeros(self._num_layers, self._num_experts, dtype=torch.int64),
            torch.zeros(self._num_layers, self._num_experts, dtype=torch.int64),
        )
        self._built = True

    # ------------------------------------------------------------------
    # properties
    # ------------------------------------------------------------------
    @property
    def enabled(self) -> bool:
        return self._built

    @property
    def ram_rows(self) -> int:
        return self._ram_rows

    @property
    def pre_ensure(self) -> bool:
        """FREETOKEN_CT_PRE_ENSURE=1: the decode path calls split_pre before
        cache.ensure_experts instead of split after it (rescue pieces A+B)."""
        return self._pre_ensure

    # ------------------------------------------------------------------
    # item ⑤ admission signal: per-expert counters (port/m4-perexpert-counts)
    # ------------------------------------------------------------------
    def route_counters(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Cumulative per-expert (route, miss) counters on the host, [L, E] int64.

        The split kernel accumulates them during graph replay (device memory,
        so replay writes are real); this on-demand read is the only D2H they
        ever pay -- no per-replay mirror node. Call at the admission evaluation
        cadence, not per step.
        """
        if not self._built:
            raise RuntimeError("attach() first")
        shape = (self._num_layers, self._num_experts)
        # copy=True even on CPU-device tensors: .cpu() alone aliases the buffer,
        # and the delta snapshot must not track later kernel increments.
        return (
            self._route_counts_dev.view(shape).to("cpu", copy=True),
            self._miss_counts_dev.view(shape).to("cpu", copy=True),
        )

    def take_route_count_deltas(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-expert (route, miss) counts since the last take -- admission's feed."""
        cur = self.route_counters()
        prev, self._counter_snapshot = self._counter_snapshot, cur
        return cur[0] - prev[0], cur[1] - prev[1]

    # ------------------------------------------------------------------
    # decode hooks
    # ------------------------------------------------------------------
    def split(self, layer_id: int, cache, hidden: torch.Tensor, topk_weights: torch.Tensor,
              topk_ids: torch.Tensor) -> None:
        """Classify this layer's misses; publish the CPU picks; shrink the fetch list.

        Runs AFTER ``cache.ensure_experts`` (slots assigned, misses staged) and
        BEFORE ``cache.copy_missing`` (doorbell/PCIe fetch), so claimed rows are
        compacted out of the staged plan and the doorbell never sees them.
        """
        if not self._built:
            return
        bsz = int(hidden.shape[0])
        if bsz > self._max_tokens:
            return  # never happens on the decode path; combine guards the same
        # Start the service BEFORE the first seq can be published: start_protocol
        # snapshots ctrl[seq] as its ignore-baseline, so if the first job's D2H
        # lands before the protocol thread is up, that job is ignored and the
        # combine kernel spins on a seq nobody will ever serve (deadlock).
        self.start()
        self._cur_slot_row = cache.slot_for_id[layer_id]
        K = int(topk_ids.numel() // bsz)
        _ct_split_kernel[(1,)](
            topk_ids,
            topk_weights,
            hidden,
            cache.slot_for_id[layer_id],
            cache.id_of_slot,
            self._row_map[layer_id],
            cache.src_indices,
            cache.evict_slots,
            cache.num_indices,
            self._mark,
            self._cnt,
            self._order,
            self._ctrl_dev,
            self._seqc,
            self._hx_dev,
            self._picks_dev.view(-1),
            self._stats_dev,
            self._route_counts_dev,
            self._miss_counts_dev,
            self._ram_rows,
            int(cache.num_experts),
            layer_id,
            bsz,
            K=K,
            H=self._hidden,
            PLAN=int(cache.src_indices.numel()),
            TZC=self._cost["tzc"],
            THIT=self._cost["thit"],
            CA=self._cost["a"],
            CB=self._cost["b"],
            TOK=self._cost["tok"],
            MAXN=int(self._cost["maxn"]),
            FORCE_N=self._force_n,
            REUSE_MIN=self._reuse_min,
            XSTEP_W=self._xstep_w,
            BLOCK_H=1024,
            BLOCK_P=1024,
            num_warps=4,
        )
        # The graph PULLS: capture-time memcpy nodes deliver the staging to the
        # pinned host mirror; the host never touches CUDA. ctrl carries seq and
        # is copied LAST, so a host observer seeing a new seq knows hx/picks
        # (copied earlier on the same stream) have also landed.
        self._hx_host[:bsz].copy_(self._hx_dev[:bsz], non_blocking=True)
        self._picks_host.copy_(self._picks_dev, non_blocking=True)
        self._ctrl_host.copy_(self._ctrl_dev, non_blocking=True)
        self.start()

    def split_pre(self, layer_id: int, cache, hidden: torch.Tensor,
                  topk_weights: torch.Tensor, topk_ids: torch.Tensor) -> None:
        """Claim-BEFORE-ensure split (FREETOKEN_CT_PRE_ENSURE=1; rescue A+B+C).

        Runs on RAW routing BEFORE ``cache.ensure_experts``: claimed experts
        get a slot-0 sentinel so lru_ensure treats them as pseudo-hits (no
        victim eviction, no staged fetch -- the churn fix), and the CPU job
        is published before ensure+fetch+GEMM, widening the overlap window
        (the async fix). ``combine`` restores the sentinels. ``topk_ids`` is
        never written (ensure's in-place rewrite is preserved); claimed
        entries' weights are zeroed (ensure never touches weights).
        """
        if not self._built:
            return
        bsz = int(hidden.shape[0])
        if bsz > self._max_tokens:
            return  # never happens on the decode path; combine guards the same
        # Same B7 discipline as split(): the service must exist before the
        # first seq can be published.
        self.start()
        self._cur_slot_row = cache.slot_for_id[layer_id]
        K = int(topk_ids.numel() // bsz)
        _ct_split_pre_kernel[(1,)](
            topk_ids,
            topk_weights,
            hidden,
            cache.slot_for_id[layer_id],
            self._row_map[layer_id],
            self._mark,
            self._cnt,
            self._order,
            self._gpum,
            self._claimed_dev,
            self._ctrl_dev,
            self._seqc,
            self._hx_dev,
            self._picks_dev.view(-1),
            self._stats_dev,
            self._route_counts_dev,
            self._miss_counts_dev,
            self._ram_rows,
            int(cache.num_experts),
            layer_id,
            bsz,
            K=K,
            H=self._hidden,
            PLAN=int(cache.src_indices.numel()),
            REUSE_MIN=self._reuse_min,
            XSTEP_W=self._xstep_w,
            TZC=self._cost["tzc"],
            THIT=self._cost["thit"],
            CA=self._cost["a"],
            CB=self._cost["b"],
            TOK=self._cost["tok"],
            MAXN=int(self._cost["maxn"]),
            FORCE_N=self._force_n,
            BLOCK_H=1024,
            BLOCK_P=1024,
            num_warps=4,
        )
        # Same graph-pull publish contract as split() (ctrl with seq LAST).
        self._hx_host[:bsz].copy_(self._hx_dev[:bsz], non_blocking=True)
        self._picks_host.copy_(self._picks_dev, non_blocking=True)
        self._ctrl_host.copy_(self._ctrl_dev, non_blocking=True)
        self.start()

    def combine(self, out: torch.Tensor) -> torch.Tensor:
        """Spin for the CPU partials and add them onto the GPU GEMM output."""
        if not self._built:
            return out
        bsz = int(out.shape[0])
        if bsz > self._max_tokens:
            return out
        timeout_iters = int(self._timeout_ns // float(_SPIN_NS_PER_ITER.value)) if self._timeout_ns > 0 else 0
        _ct_combine_kernel[(1,)](
            out,
            self._hout,
            self._done_host,
            self._dflag,
            self._ctrl_dev,
            self._claimed_dev,
            self._cur_slot_row,
            bsz,
            H=int(out.shape[-1]),
            TIMEOUT_ITERS=timeout_iters,
            BLOCK_H=1024,
            num_warps=4,
        )
        return out

    # ------------------------------------------------------------------
    # service lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Spawn the C++ worker pool (idempotent; called lazily from split)."""
        with self._lock:
            if self._started or not self._built:
                return
            from freetoken.kernel import _cpu_moe
            from freetoken.moe.cpu_executor import (
                _WFMT_IDS,
                _cpu_act_id,
                resolve_threads_and_affinity,
            )

            cache = self._cache
            fmt_id = _WFMT_IDS.get(cache.quant_format)
            if fmt_id not in (1, 3):
                raise RuntimeError(
                    f"cpu tier requires nvfp4/ds_fp4 banks (got {cache.quant_format!r}); "
                    "bf16/mxfp4 misses stay on the fetch path"
                )
            n, core_ids = resolve_threads_and_affinity(self._threads)
            banks = self._bank_tables(cache)
            layer0 = cache.moe_layer_refs[0] if getattr(cache, "moe_layer_refs", None) else None
            act_name = getattr(layer0, "activation", "silu") if layer0 is not None else "silu"
            limit = float(getattr(layer0, "swiglu_limit", 0.0) or 0.0) if layer0 is not None else 0.0
            alpha = float(getattr(layer0, "swiglu_alpha", 1.702) or 1.702) if layer0 is not None else 1.702
            apply_on_input = (
                int(bool(getattr(layer0, "apply_router_weight_on_input", False)))
                if layer0 is not None else 0
            )
            act_id = _cpu_act_id(act_name, limit)
            self._service = _cpu_moe.CpuTierService(
                n,
                cache.num_layers,
                self._hidden,
                int(banks["inter_size"]),
                self._max_tokens,
                self._max_picks,
                act_id,
                apply_on_input,
                fmt_id,
                alpha,
                limit,
                banks["gate_up"].data_ptr(),
                banks["gate_up_scale"].data_ptr(),
                banks["gate_up_global"].data_ptr(),
                banks["down"].data_ptr(),
                banks["down_scale"].data_ptr(),
                banks["down_global"].data_ptr(),
                self._ctrl_host.data_ptr(),
                self._hx_host.data_ptr(),
                self._picks_host.data_ptr(),
                self._hout.data_ptr(),
                self._done_host.data_ptr(),
                core_ids,
                -1,  # service_core: unpinned (the worker pool owns the cores)
            )
            self._banks_ref = banks  # keep the pointer tables + bank tensors alive
            self._service.start_protocol()
            self._started = True

    @staticmethod
    def _bank_tables(cache) -> dict:
        """Per-layer pointer tables over the PINNED HOST bank rows (the tier's input)."""
        # Local equivalent of CpuExecutor._make_table: an int64 tensor of
        # per-layer base addresses (the C++ service indexes tbl[layer_id]).
        def _make_table(layers):
            assert len(layers) == cache.num_layers, (len(layers), cache.num_layers)
            return torch.tensor([t.data_ptr() for t in layers], dtype=torch.int64)

        # bank_sources may be keyed by canonical role ("gate_up") or by schema
        # name ("gate_up_packed") depending on the registration path.
        from freetoken.moe.legacy_format import canonical_role

        sources = {canonical_role(n): v for n, v in cache.bank_sources.items()}
        gate_up = sources["gate_up"]
        gu_scale = sources["gate_up_scale"]
        down = sources["down"]
        dn_scale = sources["down_scale"]
        gate_up_global = sources.get("gate_up_global", gate_up)  # dummy for ds_fp4
        down_global = sources.get("down_global", down)
        inter = int(gate_up[0].shape[1] // 2)
        return {
            "gate_up": _make_table(gate_up),
            "gate_up_scale": _make_table(gu_scale),
            "gate_up_global": _make_table(gate_up_global),
            "down": _make_table(down),
            "down_scale": _make_table(dn_scale),
            "down_global": _make_table(down_global),
            "inter_size": inter,
            "_keepalive": (gate_up, gu_scale, gate_up_global, down, dn_scale, down_global),
        }

    def shutdown(self) -> None:
        with self._lock:
            svc, self._service = self._service, None
            self._started = False
        if svc is not None:
            svc.shutdown()

    # engine.shutdown() calls tier.stop(); keep the alias so teardown with
    # --moe-cpu-tier on does not AttributeError after the workers are gone.
    stop = shutdown

    # ------------------------------------------------------------------
    # observability (dsv41 ct_vllm.py:436-443 field contract)
    # ------------------------------------------------------------------
    def stats(self) -> dict:
        if not self._built:
            return {"enabled": False}
        svc = self._service
        busy_ns = int(svc.host_busy_ns()) if svc is not None else 0
        jobs = int(svc.host_jobs()) if svc is not None else 0
        dev = self._stats_dev.view(-1, 16)
        hits = int(dev[:, 0].sum().item())
        misses = int(dev[:, 1].sum().item())
        picks = int(dev[:, 2].sum().item())
        calls = int(dev[:, 3].sum().item())
        wait_ns = int(self._dflag[_DFLAG_WAIT_NS.value].item())
        timeouts = int(self._dflag[_DFLAG_TIMEOUTS.value].item())
        routed = int(self._route_counts_dev.sum().item())
        missed_entries = int(self._miss_counts_dev.sum().item())
        total = hits + misses
        return {
            "enabled": True,
            "hit_rate": (hits / total) if total else 0.0,
            "cpu_pick_rate": (picks / misses) if misses else 0.0,
            "gpu_wait_ms": wait_ns / 1e6,
            "timeouts": timeouts,
            "host_busy_ms": busy_ns / 1e6,
            "host_jobs": jobs,
            "splits": calls,
            "isa": svc.isa_name() if svc is not None else None,
            "cost": dict(self._cost),
            "fatal": int(svc.fatal()) if svc is not None else 0,
            # Per-layer route classification [layer][hits, misses, picks, calls]
            # -- observability only. Item 5's online admission consumes the
            # per-expert counters below (pulled as deltas via
            # take_route_count_deltas), not this [L,16] aggregate.
            "per_layer": dev[:, :4].cpu().tolist(),
            # Item 5's admission signal is live: cumulative [L,E] route/miss
            # counters. Full matrices via route_counters(); JSON stats carries
            # totals only -- L*E*2 int64 would bloat /v1/stats. Invariants:
            # routed == hits + misses, miss_entries == misses.
            "per_expert_counts": True,
            "routed_entries": routed,
            "miss_entries": missed_entries,
        }

    summary = stats

    # ------------------------------------------------------------------
    # self-test (CPU-runnable; the GPU-fused comparison defers to the GPU battery)
    # ------------------------------------------------------------------
    def selftest_layer(self, layer_id: int, tol: float = 0.01) -> dict:
        """CPU tier vs a pure-PyTorch reference over the same rounding rules.

        Routes synthetic tokens through ``run_job_sync`` (the production pool +
        production bank rows) and compares against a bf16/fp8 round-trip
        reference computed on the CPU. The GPU fused kernel comparison is the
        deferred half (needs a working GPU), gated behind the same rel-RMS
        contract so the GPU battery reuses this harness.
        """
        if not self._built:
            raise RuntimeError("attach() first")
        self.start()
        svc = self._service
        assert svc is not None
        cache = self._cache
        banks = cache.bank_sources
        H = self._hidden
        fmt = cache.quant_format
        layer0 = cache.moe_layer_refs[0] if getattr(cache, "moe_layer_refs", None) else None
        act_name = getattr(layer0, "activation", "silu") if layer0 is not None else "silu"
        limit = float(getattr(layer0, "swiglu_limit", 0.0) or 0.0) if layer0 is not None else 0.0
        alpha = float(getattr(layer0, "swiglu_alpha", 1.702) or 1.702) if layer0 is not None else 1.702
        apply_on_input = (
            bool(getattr(layer0, "apply_router_weight_on_input", False))
            if layer0 is not None else False
        )
        torch.manual_seed(1234 + layer_id)
        n_tok, n_pick = 4, 3
        hidden = torch.randn(n_tok, H, dtype=torch.bfloat16) * 0.05
        rows = [0, min(1, self._ram_rows - 1), self._ram_rows - 1]
        weights = [0.5, 0.3, 0.2]
        self._hx_host.zero_()
        self._hx_host[:n_tok].copy_(hidden.to(torch.float16))
        self._picks_host.zero_()
        toks = [0, 1, 3]
        for p in range(n_pick):
            self._picks_host[p, 0] = toks[p]
            self._picks_host[p, 1] = rows[p]
            self._picks_host[p, 2] = torch.tensor(weights[p], dtype=torch.float32).view(torch.int32)
        svc.run_job_sync(layer_id, n_tok, n_pick)
        got = self._hout[:n_tok].clone()
        ref = _reference_moe(
            banks, layer_id, fmt, hidden, toks, rows, weights, n_tok,
            act=act_name, limit=limit, alpha=alpha, apply_on_input=apply_on_input,
        )
        num = (got - ref).pow(2).sum().sqrt().item()
        den = ref.pow(2).sum().sqrt().item() + 1e-12
        rel = num / den
        return {"layer": layer_id, "rel_rms": rel, "ok": rel <= tol, "tol": tol}


def _bf16(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16).to(torch.float32)


def _fp8_roundtrip(x: torch.Tensor) -> torch.Tensor:
    """Per-128-block e4m3 round trip (DSV4 act_quant grid, matches the C++ kernel)."""
    x = x.to(torch.float32).clone()
    flat = x.view(-1)
    for s in range(0, flat.numel(), 128):
        blk = flat[s : s + 128]
        amax = blk.abs().max().item()
        scale = 1.0 if amax <= 0 else float(2.0 ** math.ceil(math.log2(amax / 448.0)))
        q = (blk / scale).to(torch.float8_e4m3fn).to(torch.float32)
        flat[s : s + 128] = q * scale
    return x


_E2M1_LUT = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def _dequant_nvfp4(packed: torch.Tensor, scale: torch.Tensor, glob: torch.Tensor,
                   rows: int, cols: int) -> torch.Tensor:
    """nvfp4 dequant to [rows, cols]: e2m1 nibble * per-16 e4m3 scale * per-ROW global."""
    numel = rows * cols
    lut = torch.tensor(_E2M1_LUT, dtype=torch.float32)
    lo = lut[(packed & 0x0F).long()]
    hi = lut[(packed >> 4).long()]
    vals = torch.stack([lo, hi], dim=-1).reshape(-1)[:numel].to(torch.float32)
    sc = scale.view(torch.float8_e4m3fn).to(torch.float32).repeat_interleave(16)[:numel]
    g = glob.to(torch.float16).to(torch.float32).view(rows, 1)
    return vals.view(rows, cols) * sc.view(rows, cols) * g


def _dequant_dsfp4(packed: torch.Tensor, scale: torch.Tensor, numel: int) -> torch.Tensor:
    lut = torch.tensor(_E2M1_LUT, dtype=torch.float32)
    lo = lut[(packed & 0x0F).long()]
    hi = lut[(packed >> 4).long()]
    vals = torch.stack([lo, hi], dim=-1).reshape(-1)[:numel].to(torch.float32)
    e8m0 = torch.pow(2.0, scale.to(torch.int32).clamp(max=254) - 127).to(torch.float32)
    sc = e8m0.repeat_interleave(32)[:numel]
    return vals * sc


def _reference_moe(banks, layer_id: int, fmt: str, hidden: torch.Tensor,
                   toks: list[int], rows: list[int], weights: list[float],
                   n_tok: int, act: str, limit: float, alpha: float,
                   apply_on_input: bool) -> torch.Tensor:
    """Pure-PyTorch mirror of the service numerics (self-test reference)."""
    from freetoken.moe.legacy_format import canonical_role

    banks = {canonical_role(n): v for n, v in banks.items()}
    H = int(hidden.shape[-1])
    out = torch.zeros(n_tok, H, dtype=torch.float32)
    gu_p = banks["gate_up"][layer_id].cpu()
    gu_s = banks["gate_up_scale"][layer_id].cpu()
    dn_p = banks["down"][layer_id].cpu()
    dn_s = banks["down_scale"][layer_id].cpu()
    gu_g = banks.get("gate_up_global")
    gu_g = gu_g[layer_id].cpu() if gu_g is not None else None
    dn_g = banks.get("down_global")
    dn_g = dn_g[layer_id].cpu() if dn_g is not None else None
    x_all = _fp8_roundtrip(hidden.to(torch.float32)) if fmt == "ds_fp4" else hidden.to(torch.float32)
    for tok, row, w in zip(toks, rows, weights):
        x = x_all[tok].clone()
        w_in = w if apply_on_input else 1.0
        I = gu_p.shape[1] // 2
        if fmt == "ds_fp4":
            gate = _dequant_dsfp4(gu_p[row, :I].reshape(-1), gu_s[row, :I].reshape(-1), I * H)
            up = _dequant_dsfp4(gu_p[row, I:].reshape(-1), gu_s[row, I:].reshape(-1), I * H)
            gate = _bf16(gate.view(I, H) @ x)
            up = _bf16(up.view(I, H) @ x)
        else:
            gate = _dequant_nvfp4(gu_p[row, :I].reshape(-1), gu_s[row, :I].reshape(-1),
                                  gu_g[row, :I].reshape(-1), I, H)
            up = _dequant_nvfp4(gu_p[row, I:].reshape(-1), gu_s[row, I:].reshape(-1),
                                gu_g[row, I:].reshape(-1), I, H)
            gate = gate @ (x * w_in)
            up = up @ (x * w_in)
        if limit > 0:
            gate = gate.clamp(max=limit)
            up = up.clamp(min=-limit, max=limit)
        g = _bf16(torch.nn.functional.silu(gate) * up)
        if fmt == "ds_fp4":
            g = _fp8_roundtrip(g)
            down = _dequant_dsfp4(dn_p[row].reshape(-1), dn_s[row].reshape(-1), H * I).view(H, I)
            out[tok] += _bf16((down @ g) * w)
        else:
            down = _dequant_nvfp4(dn_p[row].reshape(-1), dn_s[row].reshape(-1),
                                  dn_g[row].reshape(-1), H, I)
            out[tok] += (down @ g) * (1.0 if apply_on_input else w)
    return out


# ----------------------------------------------------------------------
# calibration (item 4): per-ISA per-expert ms, pure CPU
# ----------------------------------------------------------------------
def bench_isa_tiers(cache, layer_id: int = 0, iters: int = 20,
                    isas: Iterable[str] = ("scalar", "avx2", "avx512", "avx512bf16"),
                    out_path: str | None = None) -> dict:
    """Measure the per-expert GEMV cost for each ISA tier on this machine.

    Feeds the cost model's B parameter (per-expert ms). Spawns one short-lived
    service per ISA (FREETOKEN_CPU_MOE_ISA forces the tier; it can only step
    DOWN from the build) and times ``run_job_sync`` on a fixed pick set.
    """
    results: dict[str, dict] = {}
    for isa in isas:
        os.environ["FREETOKEN_CPU_MOE_ISA"] = isa
        tier = CpuTier(cache, cache._disk_tier)
        tier.attach(top_k=8)
        try:
            tier.start()
            svc = tier._service
            assert svc is not None
            H = tier._hidden
            n_tok, n_pick = 8, 8
            hidden = torch.randn(n_tok, H, dtype=torch.bfloat16) * 0.05
            tier._hx_host[:n_tok].copy_(hidden.to(torch.float16))
            for p in range(n_pick):
                tier._picks_host[p, 0] = p % n_tok
                tier._picks_host[p, 1] = min(p, tier._ram_rows - 1)
                tier._picks_host[p, 2] = torch.tensor(0.125, dtype=torch.float32).view(torch.int32)
            svc.run_job_sync(layer_id, n_tok, n_pick)  # warmup
            import time

            t0 = time.perf_counter()
            for _ in range(iters):
                svc.run_job_sync(layer_id, n_tok, n_pick)
            dt = (time.perf_counter() - t0) / iters
            results[isa] = {
                "selected_isa": svc.isa_name(),
                "ms_per_job": dt * 1e3,
                "ms_per_expert": dt * 1e3 / n_pick,
                "picks": n_pick,
                "tokens": n_tok,
            }
        finally:
            tier.shutdown()
    os.environ.pop("FREETOKEN_CPU_MOE_ISA", None)
    if out_path:
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
    return results
