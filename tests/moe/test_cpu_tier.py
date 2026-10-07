"""CPU tier (RAM-resident miss compute) -- pure-CPU protocol and accounting tests.

No CUDA anywhere: the C++ ``CpuTierService`` is driven directly through plain
CPU tensors (the pinned-ness of the buffers only matters for PCIe visibility,
which the GPU battery covers). The Triton split/combine kernels and the GPU
fused comparison defer to the GPU battery (broken driver here).

Covered:
  * ctrl field semantics + seq monotonicity through the real protocol thread
  * FATAL header contract (out-of-range layer: fatal set, never acked)
  * picks encoding (token routing, bank-row addressing, weight application)
  * cost-model selection boundaries (dsv41 semantics, python mirror)
  * run_job_sync numerics vs the pure-torch reference (nvfp4 + ds_fp4)
"""

from __future__ import annotations

import os
import time

import pytest
import torch

from freetoken.kernel import _cpu_moe
from freetoken.moe.cpu_tier import (
    _SEQ_OFF,
    CpuTier,
    _reference_moe,
    cost_select,
    pre_ensure_claim_plan,
    reuse_score,
)

L, E, H, I = 2, 4, 256, 128
MAX_TOKENS, MAX_PICKS = 8, 16


def _ptr_table(per_layer: list[torch.Tensor]) -> torch.Tensor:
    return torch.tensor([t.data_ptr() for t in per_layer], dtype=torch.int64)


def _nvfp4_banks(seed: int = 0) -> dict[str, list[torch.Tensor]]:
    g = torch.Generator().manual_seed(seed)
    # Random-but-valid quantized grids: any nibble is a valid e2m1; scale bytes
    # stay off the e4m3 NaN pattern (0x7F/0xFF) and away from zero/denormals.
    gu_p = list(torch.randint(0, 256, (L, E, 2 * I, H // 2), dtype=torch.uint8, generator=g))
    gu_s = list((torch.randint(0, 0x40, (L, E, 2 * I, H // 16), dtype=torch.uint8, generator=g) | 0x10))
    gu_g = list((torch.rand(L, E, 2 * I, generator=g) * 0.1 + 0.01).to(torch.float16))
    dn_p = list(torch.randint(0, 256, (L, E, H, I // 2), dtype=torch.uint8, generator=g))
    dn_s = list((torch.randint(0, 0x40, (L, E, H, I // 16), dtype=torch.uint8, generator=g) | 0x10))
    dn_g = list((torch.rand(L, E, H, generator=g) * 0.1 + 0.01).to(torch.float16))
    return {
        "gate_up_packed": gu_p,
        "gate_up_scale": gu_s,
        "gate_up_global": gu_g,
        "down_packed": dn_p,
        "down_scale": dn_s,
        "down_global": dn_g,
    }


def _dsfp4_banks(seed: int = 0) -> dict[str, list[torch.Tensor]]:
    g = torch.Generator().manual_seed(seed)
    gu_p = list(torch.randint(0, 256, (L, E, 2 * I, H // 2), dtype=torch.uint8, generator=g))
    gu_s = list(torch.randint(110, 130, (L, E, 2 * I, H // 32), dtype=torch.uint8, generator=g))
    dn_p = list(torch.randint(0, 256, (L, E, H, I // 2), dtype=torch.uint8, generator=g))
    dn_s = list(torch.randint(110, 130, (L, E, H, I // 32), dtype=torch.uint8, generator=g))
    return {
        "gate_up_packed": gu_p,
        "gate_up_scale": gu_s,
        "down_packed": dn_p,
        "down_scale": dn_s,
    }


def _make_service(banks: dict[str, list[torch.Tensor]], fmt_id: int,
                  apply_on_input: int = 0, num_threads: int = 2):
    gu_global = banks.get("gate_up_global", banks["gate_up_packed"])
    dn_global = banks.get("down_global", banks["down_packed"])
    bufs = {
        "ctrl": torch.zeros(8, dtype=torch.int64),
        "hx": torch.zeros(MAX_TOKENS, H, dtype=torch.float16),
        "picks": torch.zeros(MAX_PICKS, 3, dtype=torch.int32),
        "hout": torch.zeros(MAX_TOKENS, H, dtype=torch.float32),
        "done": torch.zeros(1, dtype=torch.int64),
    }
    tables = {
        name: _ptr_table(tensors)
        for name, tensors in (
            ("gate_up", banks["gate_up_packed"]),
            ("gate_up_scale", banks["gate_up_scale"]),
            ("gate_up_global", gu_global),
            ("down", banks["down_packed"]),
            ("down_scale", banks["down_scale"]),
            ("down_global", dn_global),
        )
    }
    svc = _cpu_moe.CpuTierService(
        num_threads, L, H, I, MAX_TOKENS, MAX_PICKS,
        0,  # act: silu
        apply_on_input, fmt_id, 1.702, 0.0,
        tables["gate_up"].data_ptr(), tables["gate_up_scale"].data_ptr(),
        tables["gate_up_global"].data_ptr(), tables["down"].data_ptr(),
        tables["down_scale"].data_ptr(), tables["down_global"].data_ptr(),
        bufs["ctrl"].data_ptr(), bufs["hx"].data_ptr(), bufs["picks"].data_ptr(),
        bufs["hout"].data_ptr(), bufs["done"].data_ptr(), [], -1,
    )
    return svc, bufs, tables, banks


def _write_pick(bufs, p: int, tok: int, row: int, weight: float) -> None:
    bufs["picks"][p, 0] = tok
    bufs["picks"][p, 1] = row
    bufs["picks"][p, 2] = torch.tensor(weight, dtype=torch.float32).view(torch.int32)


def _serve(bufs, layer: int, bsz: int, npk: int, seq: int) -> None:
    """Simulate the graph's D2H pull: header fields land, seq LAST."""
    ctrl = bufs["ctrl"]
    ctrl[0] = layer
    ctrl[1] = bsz
    ctrl[2] = npk
    ctrl[int(_SEQ_OFF)] = seq


def _wait_done(bufs, seq: int, timeout_s: float = 10.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if int(bufs["done"][0]) >= seq:
            return True
        time.sleep(0.0005)
    return False


# ----------------------------------------------------------------------
# protocol accounting
# ----------------------------------------------------------------------
def test_protocol_ctrl_fields_and_seq_monotonic():
    svc, bufs, _t, banks = _make_service(_nvfp4_banks(), fmt_id=1)
    try:
        assert svc.fatal() == 0
        svc.start_protocol()
        bufs["hx"][:] = (torch.randn(MAX_TOKENS, H) * 0.05).to(torch.float16)
        _write_pick(bufs, 0, tok=0, row=1, weight=0.5)
        _write_pick(bufs, 1, tok=1, row=2, weight=0.25)
        _serve(bufs, layer=0, bsz=2, npk=2, seq=1)
        assert _wait_done(bufs, 1), "service did not ack seq 1"
        assert svc.host_jobs() == 1
        assert svc.host_busy_ns() > 0
        first = bufs["hout"][:2].clone()
        assert first.abs().sum() > 0

        # A second, different frame at a higher seq must be served (monotonic).
        _write_pick(bufs, 0, tok=0, row=3, weight=1.0)
        _write_pick(bufs, 1, tok=1, row=3, weight=1.0)
        _serve(bufs, layer=1, bsz=2, npk=2, seq=2)
        assert _wait_done(bufs, 2), "service did not ack seq 2"
        assert svc.host_jobs() == 2
        second = bufs["hout"][:2].clone()
        assert not torch.allclose(first, second)
        assert bufs["hout"][2:].abs().sum() == 0  # rows beyond bsz untouched
    finally:
        svc.shutdown()


def test_protocol_npak_zero_acks_without_work():
    svc, bufs, _t, _b = _make_service(_nvfp4_banks(), fmt_id=1)
    try:
        svc.start_protocol()
        _serve(bufs, layer=0, bsz=4, npk=0, seq=1)
        assert _wait_done(bufs, 1)
        assert bufs["hout"].abs().sum() == 0  # prep zeroes, reduce writes zeros
        assert svc.host_jobs() == 1
    finally:
        svc.shutdown()


def test_protocol_fatal_header_never_acked():
    svc, bufs, _t, _b = _make_service(_nvfp4_banks(), fmt_id=1)
    try:
        svc.start_protocol()
        _serve(bufs, layer=99, bsz=2, npk=1, seq=1)  # layer out of range
        time.sleep(0.2)
        assert svc.fatal() == 1
        assert int(bufs["done"][0]) == 0, "corrupt frame must never be acked"
        # A following VALID frame is still served (the corrupt seq is skipped).
        _serve(bufs, layer=0, bsz=1, npk=0, seq=2)
        assert _wait_done(bufs, 2)
    finally:
        svc.shutdown()


def test_picks_encoding_token_row_weight():
    svc, bufs, _t, banks = _make_service(_nvfp4_banks(), fmt_id=1)
    try:
        bufs["hx"][:] = (torch.randn(MAX_TOKENS, H) * 0.05).to(torch.float16)
        # One pick on token 2 only.
        _write_pick(bufs, 0, tok=2, row=1, weight=0.5)
        svc.run_job_sync(0, MAX_TOKENS, 1)
        out = bufs["hout"]
        assert out[0].abs().sum() == 0 and out[1].abs().sum() == 0
        assert out[3:].abs().sum() == 0
        row1 = out[2].clone()
        assert row1.abs().sum() > 0
        # A different bank row gives a different result (rows address experts).
        _write_pick(bufs, 0, tok=2, row=3, weight=0.5)
        svc.run_job_sync(0, MAX_TOKENS, 1)
        row3 = bufs["hout"][2].clone()
        assert not torch.allclose(row1, row3)
        # Weight scales the output (apply_on_input=0: applied at the reduction).
        _write_pick(bufs, 0, tok=2, row=1, weight=1.0)
        svc.run_job_sync(0, MAX_TOKENS, 1)
        row1_w2 = bufs["hout"][2].clone()
        ratio = (row1_w2.norm() / row1.norm().clamp_min(1e-12)).item()
        assert 1.5 < ratio < 2.5  # weight 1.0 vs 0.5 (bf16 route rounding)
        # Zero weight zeroes the contribution exactly.
        _write_pick(bufs, 0, tok=2, row=1, weight=0.0)
        svc.run_job_sync(0, MAX_TOKENS, 1)
        assert bufs["hout"][2].abs().sum() == 0
    finally:
        svc.shutdown()


# ----------------------------------------------------------------------
# cost-model boundaries (dsv41 semantics, python mirror of kernel passes 2-3)
# ----------------------------------------------------------------------
_COST = {"tzc": 0.58, "thit": 0.03, "a": 0.11, "b": 0.20, "tok": 0.35, "maxn": 384}


def test_cost_select_force_n_overrides_model():
    entries = [(3, 0), (1, 1), (2, 2)]
    assert cost_select(entries, nh=10, nm=4, cost=_COST, force_n=2) == 2
    assert cost_select(entries, nh=10, nm=4, cost=_COST, force_n=99) == 3
    assert cost_select(entries, nh=10, nm=4, cost=_COST, force_n=0) == 0


def test_cost_select_no_misses_means_no_work():
    # nh only (nm == 0): gpu(k) = thit*(nh-k) - tzc*k shrinks with k, but there
    # are no CPU-ok experts at all, so nothing to serve.
    assert cost_select([], nh=16, nm=0, cost=_COST) == 0


def test_cost_select_expensive_cpu_serves_nothing():
    # cpu(k) >> gpu(k) for every k > 0 and tt(0) is the minimum, but tt(0) > 0
    # means "no beneficial CPU work": gpu(0)=thit*nh+tzc*nm dominates.
    dear = dict(_COST, b=100.0, a=100.0)
    assert cost_select([(1, 0)], nh=1, nm=1, cost=dear) == 0


def test_cost_select_picks_cheapest_prefix():
    # With a cheap CPU and expensive PCIe, every CPU-ok expert goes to the CPU.
    cheap = dict(_COST, tzc=10.0, b=0.01, a=0.01)
    entries = [(1, 0), (4, 1), (2, 2)]
    assert cost_select(entries, nh=2, nm=8, cost=cheap) == 3


def test_cost_select_maxn_caps_scan():
    tight = dict(_COST, tzc=10.0, b=0.001, a=0.001, maxn=2)
    entries = [(1, i) for i in range(5)]
    assert cost_select(entries, nh=2, nm=10, cost=tight) == 2


def test_cost_select_duplicates_make_experts_cheaper_first():
    # Sort order is (count asc, ordinal asc): duplicates cost TOK extra each,
    # so the singleton is selected first -- verified indirectly through the
    # ordering contract by forcing k=1 and checking the mirror's determinism.
    entries = [(2, 0), (1, 1)]
    cheap = dict(_COST, tzc=10.0, b=0.01, a=0.01)
    assert cost_select(entries, nh=0, nm=3, cost=cheap, force_n=1) == 1
    # k=1 picks the count-1 expert: cpu(1) = a + b*(1+tok*(1-1)) = a+b.
    # k=2 picks both: a + b + b*(1+tok). Stable across repeated calls.
    assert cost_select(entries, nh=0, nm=3, cost=cheap) == 2


# ----------------------------------------------------------------------
# 20261007 rescue piece C: reuse-aware claim filter (kernel pass mirror)
# ----------------------------------------------------------------------
def test_reuse_score_components():
    assert reuse_score(1, 0, 0) == 1.0  # cold start: no history, freq 0
    assert reuse_score(1, 50, 100) == 1.5  # cross-step freq 0.5
    assert reuse_score(3, 10, 10, xstep_w=0.0) == 3.0  # w=0: this-step only
    assert reuse_score(1, 10, 10, xstep_w=3.0) == 4.0  # w scales the freq term


def test_cost_select_reuse_min_demotes_low_reuse():
    cheap = dict(_COST, tzc=10.0, b=0.01, a=0.01)
    entries = [(1, 0), (1, 1), (1, 2)]
    # no filter: the cheap-CPU model claims all three
    assert cost_select(entries, nh=2, nm=3, cost=cheap) == 3
    # scores below reuse_min are demoted before ordering
    assert cost_select(entries, nh=2, nm=3, cost=cheap,
                       reuse_scores=[0.2, 1.5, 0.7], reuse_min=1.0) == 1
    # boundary is inclusive: score == reuse_min is kept
    assert cost_select(entries, nh=2, nm=3, cost=cheap,
                       reuse_scores=[1.0, 0.1, 0.1], reuse_min=1.0) == 1
    # everything demoted -> nothing served (tt(0) > 0 keeps the fetch path)
    assert cost_select(entries, nh=2, nm=3, cost=cheap,
                       reuse_scores=[0.0, 0.0, 0.0], reuse_min=0.5) == 0


def test_cost_select_reuse_filter_off_by_default():
    cheap = dict(_COST, tzc=10.0, b=0.01, a=0.01)
    assert cost_select([(1, 0)], nh=1, nm=1, cost=cheap,
                       reuse_scores=[0.0], reuse_min=-1.0) == 1


def test_cost_select_force_n_bypasses_reuse_filter():
    entries = [(1, 0), (1, 1)]
    assert cost_select(entries, nh=2, nm=2, cost=_COST, force_n=2,
                       reuse_scores=[0.0, 0.0], reuse_min=99.0) == 2


# ----------------------------------------------------------------------
# 20261007 rescue pieces A+B: pre-ensure claim bookkeeping mirror
# ----------------------------------------------------------------------
def _plan(**overrides):
    args = dict(
        ids=[5, 2, 5, 7, 2, 9],
        K=3,
        slot_for_id_row={5: -1, 2: 4, 7: -1, 9: -1},
        row_map_row={5: 0, 2: 1, 7: 2, 9: -1},
        route_counts_pre={5: 10, 7: 0, 9: 3},
        calls_pre=10,
        cost=dict(_COST, tzc=10.0, b=0.01, a=0.01),
        ram_rows=64,
    )
    args.update(overrides)
    return pre_ensure_claim_plan(**args)


def test_pre_ensure_plan_claims_only_missed_pinned():
    p = _plan()
    # expert 2 is slot-resident (slot 4): never claimed; expert 9 misses but
    # has no pin row: stays GPU-bound. 5 and 7 are CPU-ok misses.
    assert set(p["claimed"]) <= {5, 7}
    assert 2 not in p["claimed"] and 9 not in p["claimed"]
    assert p["nh"] == 2 and p["n_miss"] == 4  # two entries hit slot 4


def test_pre_ensure_plan_sentinel_weight_and_picks():
    p = _plan()
    assert set(p["sentinels"]) == set(p["claimed"])
    zeroed = set(p["weight_zeroed"])
    for i, e in enumerate([5, 2, 5, 7, 2, 9]):
        assert (i in zeroed) == (e in p["claimed"])
    # every pick addresses a pin row of a claimed expert, token = entry // K
    assert len(p["picks"]) == len(zeroed)
    for tok, row in p["picks"]:
        assert 0 <= row < 64 and tok in (0, 1)
    # publish order: selection order (5 has cnt 2 > 7's cnt 1 -> 7 first),
    # entries in routed order within an expert
    assert p["claimed"] == [7, 5]
    assert p["picks"] == [(1, 2), (0, 0), (0, 0)]


def test_pre_ensure_plan_reuse_demotion_leaves_expert_to_fetch():
    # 5: cnt 2 + freq 10/10 = 3.0 >= 1.5 kept; 7: cnt 1 + freq 0 = 1.0 demoted
    p = _plan(reuse_min=1.5)
    assert set(p["claimed"]) == {5}
    assert p["demoted"] == [7]
    p0 = _plan()  # filter off: both CPU-ok misses are claimable
    assert set(p0["claimed"]) == {5, 7}
    assert p0["demoted"] == []


def test_pre_ensure_plan_force_n_bypasses_reuse():
    p = _plan(reuse_min=99.0, force_n=2)
    assert set(p["claimed"]) == {5, 7}
    assert p["demoted"] == []


def test_pre_ensure_plan_determinism():
    assert _plan() == _plan()


def test_rescue_env_flags_default_off(monkeypatch):
    for v in ("FREETOKEN_CT_PRE_ENSURE", "FREETOKEN_CT_REUSE_MIN",
              "FREETOKEN_CT_XSTEP_W"):
        monkeypatch.delenv(v, raising=False)
    tier = CpuTier(None, None)
    assert tier.pre_ensure is False
    assert tier._reuse_min < 0.0
    assert tier._xstep_w == 1.0


def test_rescue_env_flags_parse(monkeypatch):
    monkeypatch.setenv("FREETOKEN_CT_PRE_ENSURE", "1")
    monkeypatch.setenv("FREETOKEN_CT_REUSE_MIN", "1.5")
    monkeypatch.setenv("FREETOKEN_CT_XSTEP_W", "3")
    tier = CpuTier(None, None)
    assert tier.pre_ensure is True
    assert tier._reuse_min == 1.5
    assert tier._xstep_w == 3.0


# ----------------------------------------------------------------------
# numerics vs the pure-torch reference (nvfp4 + ds_fp4)
# ----------------------------------------------------------------------
def _rel_rms(a: torch.Tensor, b: torch.Tensor) -> float:
    num = (a - b).pow(2).sum().sqrt().item()
    den = b.pow(2).sum().sqrt().item() + 1e-12
    return num / den


@pytest.mark.parametrize(
    "fmt,banks_fn,fmt_id",
    [("nvfp4", _nvfp4_banks, 1), ("ds_fp4", _dsfp4_banks, 3)],
)
@pytest.mark.parametrize("apply_on_input", [0, 1])
def test_run_job_sync_matches_reference(fmt, banks_fn, fmt_id, apply_on_input):
    banks = banks_fn(seed=7)
    svc, bufs, _t, _ = _make_service(banks, fmt_id=fmt_id, apply_on_input=apply_on_input)
    try:
        torch.manual_seed(99)
        n_tok = 4
        hidden = torch.randn(n_tok, H, dtype=torch.bfloat16) * 0.05
        bufs["hx"][:n_tok].copy_(hidden.to(torch.float16))
        toks = [0, 1, 1, 3]
        rows = [0, 1, 3, 2]
        weights = [0.5, 0.3, 0.2, 0.7]
        for p in range(len(toks)):
            _write_pick(bufs, p, toks[p], rows[p], weights[p])
        layer = 1
        svc.run_job_sync(layer, n_tok, len(toks))
        got = bufs["hout"][:n_tok].clone()
        ref = _reference_moe(
            banks, layer, fmt, hidden, toks, rows, weights, n_tok,
            act="silu", limit=0.0, alpha=1.702,
            apply_on_input=bool(apply_on_input),
        )
        rel = _rel_rms(got, ref)
        assert rel <= 0.01, f"{fmt} apply_on_input={apply_on_input}: rel_rms={rel}"
    finally:
        svc.shutdown()


def test_determinism_across_repeated_jobs():
    banks = _nvfp4_banks(seed=3)
    svc, bufs, _t, _ = _make_service(banks, fmt_id=1, num_threads=4)
    try:
        torch.manual_seed(5)
        n_tok = MAX_TOKENS
        hidden = torch.randn(n_tok, H, dtype=torch.bfloat16) * 0.05
        bufs["hx"][:n_tok].copy_(hidden.to(torch.float16))
        for p in range(MAX_PICKS):
            _write_pick(bufs, p, tok=p % n_tok, row=p % E, weight=0.125)
        svc.run_job_sync(0, n_tok, MAX_PICKS)
        first = bufs["hout"][:n_tok].clone()
        for _ in range(3):
            svc.run_job_sync(0, n_tok, MAX_PICKS)
            assert torch.equal(first, bufs["hout"][:n_tok]), (
                "per-pick scratch + ordered reduction must be bitwise stable"
            )
    finally:        svc.shutdown()

# ----------------------------------------------------------------------
# Kernel-logic regression via the triton interpreter (pure CPU): the helper
# runs in a subprocess with TRITON_INTERPRET=1 set before triton is imported
# (the interpreter's builtin patching only engages on a fresh import). It
# exits 3 when the interpreter is unavailable -> the test skips.
# ----------------------------------------------------------------------
def _run_interp_helper() -> None:
    import subprocess
    import sys
    from pathlib import Path

    helper = Path(__file__).with_name("_cpu_tier_interp_check.py")
    env = dict(os.environ, TRITON_INTERPRET="1")
    repo_python = str(Path(__file__).resolve().parents[2] / "python")
    env["PYTHONPATH"] = os.pathsep.join(
        [repo_python] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    proc = subprocess.run([sys.executable, str(helper)], env=env,
                          capture_output=True, text=True, timeout=300)
    if proc.returncode == 3:
        pytest.skip(f"triton interpreter unavailable: {proc.stdout.strip()}")
    assert proc.returncode == 0, (
        f"interp helper failed (rc={proc.returncode})\n{proc.stdout}\n{proc.stderr}")


def test_split_kernel_b9_single_pick_per_claimed_entry():
    """B9: a claimed entry rewritten to the slot-0 sentinel must not be
    re-matched by a later claim whose victim slot is 0 (duplicate weight-0
    picks -> wasted 22MB DRAM expert reads in the service)."""
    _run_interp_helper()


def test_split_pre_kernel_sentinel_and_picks_interp():
    """Pre-ensure kernel == its python mirror: slot-0 sentinels, zeroed
    weights, untouched ids, picks, stats (claim-before-ensure bookkeeping)."""
    _run_interp_helper()
