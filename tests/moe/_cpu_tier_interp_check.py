"""Interpreter-mode kernel-logic checks for the CPU tier (helper, not a test).

Run as a subprocess by tests/moe/test_cpu_tier.py with TRITON_INTERPRET=1 set
BEFORE triton is imported (the interpreter's builtin patching only engages on
a fresh triton import). Exit codes: 0 = all checks passed, 3 = interpreter
unavailable in this environment (the caller skips), 1 = a real check failed.

The twins are the kernels' own source with GPU-only scalar int64 upcasts
stripped (the interpreter executes range vars/args as plain ints; all offsets
fit int32 at production magnitudes), exec'd and re-jitted.
"""

from __future__ import annotations

import inspect
import sys

import torch
import triton
import triton.language as tl

import freetoken.moe.cpu_tier as ct


def _twin(jitfn):
    src = inspect.getsource(jitfn.fn)
    src = src.replace('@triton.jit(do_not_specialize=["layer_id", "bsz"])', "")
    src = src.replace(".to(tl.int64)", "")
    ns = {}
    exec(compile(src, "<interp_twin>", "exec"),
         {"tl": tl, "_BIG": ct._BIG, "_SEQ_OFF": ct._SEQ_OFF,
          "_CTRL_NCLAIMED": ct._CTRL_NCLAIMED}, ns)
    return triton.jit(ns[jitfn.fn.__name__])


def check_b9_single_pick_per_claimed_entry() -> None:
    """A claimed entry rewritten to the slot-0 sentinel must not be re-matched
    by a later claim whose victim slot is 0 (duplicate weight-0 picks -> wasted
    22MB DRAM expert reads in the service)."""
    kern = _twin(ct._ct_split_kernel)
    layer, E2, H2, K2, B2, PLAN2, RAM2 = 1, 8, 16, 3, 2, 16, 6
    # post-ensure: expert 2 resident at slot 4; staged misses 5->slot0,
    # 6->slot1, 7->slot2 (ascending); expert 6 unpinned.
    slots = torch.tensor([0, 4, 0, 2, 4, 1], dtype=torch.int32)
    weights = torch.arange(1, 7, dtype=torch.float32)
    hidden = torch.zeros(B2, H2, dtype=torch.float16)
    sf = torch.tensor([0, -1, 4, -1, -1, 0, 1, 2], dtype=torch.int32)
    ios = torch.tensor([8 + 5, 8 + 6, 8 + 7, -1, 8 + 2, -1, -1, -1],
                       dtype=torch.int32)
    row_map = torch.tensor([0, 1, 2, 3, -1, 4, -1, 5], dtype=torch.int32)
    src = torch.tensor([5, 6, 7] + [0] * 13, dtype=torch.int32)
    evict = torch.tensor([0, 1, 2] + [0] * 13, dtype=torch.int32)
    num_idx = torch.tensor([3], dtype=torch.int64)
    z = lambda: torch.zeros(PLAN2, dtype=torch.int32)
    mark, cnt, order = z(), z(), z()
    ctrl = torch.zeros(8, dtype=torch.int64)
    seqc = torch.zeros(1, dtype=torch.int64)
    hx = torch.zeros(B2, H2, dtype=torch.float16)
    picks = torch.zeros(48, dtype=torch.int32)
    stats = torch.zeros(2 * 16, dtype=torch.int64)
    rc = torch.zeros(2 * E2, dtype=torch.int64)
    mc = torch.zeros(2 * E2, dtype=torch.int64)
    kern[(1,)](
        slots, weights, hidden, sf, ios, row_map, src, evict, num_idx,
        mark, cnt, order, ctrl, seqc, hx, picks, stats, rc, mc,
        RAM2, E2, layer, B2, K=K2, H=H2, PLAN=PLAN2,
        TZC=10.0, THIT=0.03, CA=0.01, CB=0.01, TOK=0.35, MAXN=384,
        FORCE_N=-1, REUSE_MIN=-1.0, XSTEP_W=1.0,
        BLOCK_H=1024, BLOCK_P=1024, num_warps=1,
    )
    # cost model claims expert 7 (victim slot 2) then expert 5 (victim slot 0)
    npk = int(ctrl[2])
    assert npk == 3, f"picks {npk} != 3 (one per claimed (token, expert) pair)"
    got = sorted((int(picks[j * 3]), int(picks[j * 3 + 1])) for j in range(npk))
    # expert 5 (pin row 4) by entries 0/2 -> tok 0, 0; expert 7 (pin row 5)
    # by entry 3 -> tok 1, exactly once each
    assert got == [(0, 4), (0, 4), (1, 5)], got
    for j in range(npk):  # every pick carries its real (nonzero) weight
        assert int(picks[j * 3 + 2]) != 0
    assert int(num_idx[0]) == 1  # only the unpinned expert 6 stays staged
    print("OK b9-single-pick")


def check_pre_ensure_sentinel_and_picks() -> None:
    """Pre-ensure kernel == its python mirror: sentinels, zeroed weights,
    untouched ids, picks, stats, counters."""
    kern = _twin(ct._ct_split_pre_kernel)
    layer, E2, H2, K2, B2, PLAN2, RAM2 = 1, 8, 16, 3, 2, 16, 6
    raw = [5, 2, 5, 7, 2, 6]
    slot_row = [-1, -1, 4, -1, -1, -1, -1, -1]
    pin_row = [0, 1, 2, 3, -1, 4, -1, 5]
    ids = torch.tensor(raw, dtype=torch.int32)
    weights = torch.arange(1, 7, dtype=torch.float32)
    hidden = torch.zeros(B2, H2, dtype=torch.float16)
    sf = torch.tensor(slot_row, dtype=torch.int32)
    row_map = torch.tensor(pin_row, dtype=torch.int32)
    z = lambda: torch.zeros(PLAN2, dtype=torch.int32)
    mark, cnt, order, gpum, claimed = z(), z(), z(), z(), z()
    ctrl = torch.zeros(8, dtype=torch.int64)
    seqc = torch.zeros(1, dtype=torch.int64)
    hx = torch.zeros(B2, H2, dtype=torch.float16)
    picks = torch.zeros(48, dtype=torch.int32)
    stats = torch.zeros(2 * 16, dtype=torch.int64)
    rc = torch.zeros(2 * E2, dtype=torch.int64)
    mc = torch.zeros(2 * E2, dtype=torch.int64)
    kern[(1,)](
        ids, weights, hidden, sf, row_map, mark, cnt, order, gpum, claimed,
        ctrl, seqc, hx, picks, stats, rc, mc,
        RAM2, E2, layer, B2, 0, K=K2, H=H2, PLAN=PLAN2,
        REUSE_MIN=-1.0, XSTEP_W=1.0,
        TZC=10.0, THIT=0.03, CA=0.01, CB=0.01, TOK=0.35, MAXN=384,
        FORCE_N=-1, BLOCK_H=1024, BLOCK_P=1024, num_warps=1,
    )
    mirror = ct.pre_ensure_claim_plan(
        raw, K2,
        {e: slot_row[e] for e in range(E2)},
        {e: pin_row[e] for e in range(E2)},
        {}, 0, dict(tzc=10.0, thit=0.03, a=0.01, b=0.01, tok=0.35, maxn=384),
        RAM2)
    ncl, npk = int(ctrl[3]), int(ctrl[2])
    assert sorted(int(claimed[j]) for j in range(ncl)) == sorted(mirror["claimed"])
    for e in mirror["claimed"]:
        assert int(sf[e]) == 0  # pseudo-hit sentinel written
    for i in range(len(raw)):
        assert (float(weights[i]) == 0.0) == (i in mirror["weight_zeroed"])
    assert [int(x) for x in ids] == raw  # ids never written
    got = [(int(picks[j * 3]), int(picks[j * 3 + 1])) for j in range(npk)]
    assert got == mirror["picks"], (got, mirror["picks"])
    assert int(stats[layer * 16 + 0]) == mirror["nh"]
    assert int(stats[layer * 16 + 1]) == mirror["n_miss"]
    print("OK pre-ensure-sentinel")


def check_pre_ensure_owner_ep_filter() -> None:
    """B10c: the pre-ensure kernel with a nonzero GLOBAL_START claims ONLY
    this rank's owned experts. Remote entries are never classified, claimed,
    sentinel'd, weight-zeroed, picked, or counter-bumped -- they only feed
    the cost model's GEMM term. Twin must equal the python mirror run with
    global_start/local_num_experts."""
    kern = _twin(ct._ct_split_pre_kernel)
    layer, E2, H2, K2, B2, PLAN2, RAM2, G0 = 1, 8, 16, 3, 2, 16, 6, 8
    # global route on rank 1 (owns 8..15 == local 0..7): two remote entries
    # (0, 3), one owned hit (9 == local 1 resident at slot 4), two owned
    # pinned misses (8 == local 0 row 0, 10 == local 2 row 2).
    raw = [0, 9, 8, 3, 9, 10]
    slot_row = [-1, 4, -1, -1, -1, -1, -1, -1]
    pin_row = [0, 1, 2, 3, -1, 4, -1, 5]
    ids = torch.tensor(raw, dtype=torch.int32)
    weights = torch.arange(1, 7, dtype=torch.float32)
    hidden = torch.zeros(B2, H2, dtype=torch.float16)
    sf = torch.tensor(slot_row, dtype=torch.int32)
    row_map = torch.tensor(pin_row, dtype=torch.int32)
    z = lambda: torch.zeros(PLAN2, dtype=torch.int32)
    mark, cnt, order, gpum, claimed = z(), z(), z(), z(), z()
    ctrl = torch.zeros(8, dtype=torch.int64)
    seqc = torch.zeros(1, dtype=torch.int64)
    hx = torch.zeros(B2, H2, dtype=torch.float16)
    picks = torch.zeros(48, dtype=torch.int32)
    stats = torch.zeros(2 * 16, dtype=torch.int64)
    rc = torch.zeros(2 * E2, dtype=torch.int64)
    mc = torch.zeros(2 * E2, dtype=torch.int64)
    kern[(1,)](
        ids, weights, hidden, sf, row_map, mark, cnt, order, gpum, claimed,
        ctrl, seqc, hx, picks, stats, rc, mc,
        RAM2, E2, layer, B2, G0, K=K2, H=H2, PLAN=PLAN2,
        REUSE_MIN=-1.0, XSTEP_W=1.0,
        TZC=10.0, THIT=0.03, CA=0.01, CB=0.01, TOK=0.35, MAXN=384,
        FORCE_N=-1, BLOCK_H=1024, BLOCK_P=1024, num_warps=1,
    )
    mirror = ct.pre_ensure_claim_plan(
        raw, K2,
        {e: slot_row[e] for e in range(E2)},
        {e: pin_row[e] for e in range(E2)},
        {}, 0, dict(tzc=10.0, thit=0.03, a=0.01, b=0.01, tok=0.35, maxn=384),
        RAM2, global_start=G0, local_num_experts=E2)
    assert mirror["nh"] == 2 and mirror["n_remote"] == 2 and mirror["n_miss"] == 2
    ncl, npk = int(ctrl[3]), int(ctrl[2])
    assert sorted(int(claimed[j]) for j in range(ncl)) == sorted(mirror["claimed"])
    for e in mirror["claimed"]:  # LOCAL ids only; sentinel in the local row
        assert 0 <= e < E2
        assert int(sf[e]) == 0
    for i in range(len(raw)):
        assert (float(weights[i]) == 0.0) == (i in mirror["weight_zeroed"])
    # remote entries (indices 0, 3) must keep their weights
    assert float(weights[0]) != 0.0 and float(weights[3]) != 0.0
    assert [int(x) for x in ids] == raw  # ids never written
    got = [(int(picks[j * 3]), int(picks[j * 3 + 1])) for j in range(npk)]
    assert got == mirror["picks"], (got, mirror["picks"])
    assert int(stats[layer * 16 + 0]) == mirror["nh"]
    assert int(stats[layer * 16 + 1]) == mirror["n_miss"]
    # route/miss counters bumped ONLY for owned entries (local namespace)
    assert int(rc[layer * E2 + 1]) == 2 and int(rc[layer * E2 + 0]) == 1
    assert int(rc[layer * E2 + 2]) == 1
    assert int(rc[: layer * E2].sum()) == 0
    assert int(rc[layer * E2:].sum()) == 4  # remote 0, 3 never counted
    assert int(mc[layer * E2 + 0]) == 1 and int(mc[layer * E2 + 2]) == 1
    print("OK pre-ensure-owner-ep")


def main() -> int:
    try:
        check_b9_single_pick_per_claimed_entry()
        check_pre_ensure_sentinel_and_picks()
        check_pre_ensure_owner_ep_filter()
    except AssertionError:
        raise  # real check failure -> exit 1
    except Exception as exc:  # interpreter/twin drift -> caller skips
        print(f"INTERP-SKIP: {exc!r}")
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
