"""CPU tests for the 2c prefill fat-GEMM split: decision boundary, bucket
correctness, dequant semantics, the fidelity-gate metric, and the env dispatch.

The GPU half (fat path vs the production grouped kernel, bf16, D061 gate) is
marked ``cuda`` and runs in the GPU battery; the module selftest
(``python -m freetoken.moe.prefill_fat_gemm``) is its entry point.
"""

from __future__ import annotations

import pytest
import torch

import freetoken.moe.fused_nvfp4 as fnv4
from freetoken.moe import prefill_fat_gemm as fat

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _flat_ids(counts: list[int]) -> torch.Tensor:
    """Flat route ids with ``counts[e]`` routes on expert e, shuffled deterministically."""
    ids = torch.cat([torch.full((c,), e, dtype=torch.int64) for e, c in enumerate(counts) if c])
    g = torch.Generator().manual_seed(7)
    return ids[torch.randperm(ids.numel(), generator=g)]


# ---------------------------------------------------------------------------
# Decision boundary / bucketing
# ---------------------------------------------------------------------------


def test_no_busy_expert_returns_none():
    split = fat.plan_fat_split(_flat_ids([3, 1, 2]), top_k=2, num_experts=3, min_rows=4, block_m=8)
    assert split is None


def test_threshold_boundary_is_inclusive():
    # count == min_rows -> busy; min_rows - 1 -> light
    split = fat.plan_fat_split(_flat_ids([4, 3]), top_k=1, num_experts=2, min_rows=4, block_m=8)
    assert split is not None
    assert split.busy == [0]
    assert split.light_expert_ids.tolist() == [1]


def test_min_rows_one_makes_every_routed_expert_busy():
    split = fat.plan_fat_split(_flat_ids([2, 0, 1]), top_k=1, num_experts=3, min_rows=1, block_m=8)
    assert split is not None
    assert split.busy == [0, 2]
    assert split.light_ntpp == 0
    assert split.light_sorted_ids.numel() == 0
    assert split.light_expert_ids.numel() == 0


def test_busy_positions_partition_the_routes():
    counts = [5, 2, 9, 0, 4]
    flat_ids = _flat_ids(counts)
    split = fat.plan_fat_split(flat_ids, top_k=3, num_experts=5, min_rows=5, block_m=4)
    assert split is not None
    assert split.busy == [0, 2]
    for e, pos in zip(split.busy, split.busy_positions):
        assert pos.numel() == counts[e]
        assert (flat_ids[pos] == e).all()
    # light routes land in the align, padded to block_m per light expert
    light = flat_ids[split.light_sorted_ids[split.light_sorted_ids < flat_ids.numel()].long()]
    assert sorted(light.tolist()) == sorted([1] * 2 + [4] * 4)
    # expert 1: 2 routes -> 1 block (2 pads); expert 4: 4 routes -> 1 block (0 pads)
    assert split.light_expert_ids.tolist() == [1, 4]
    assert split.light_ntpp == 8


def test_light_align_matches_production_semantics():
    """Host align == moe_align_block_size semantics on the light subset: routes
    grouped by ascending expert, stable within an expert, sentinel padding."""
    flat = torch.tensor([1, 0, 1, 2, 0, 2, 3, 3, 3], dtype=torch.int32)
    num_valid = flat.numel()
    # e0/e1/e2 hold 2 routes each (light at min_rows=3); e3 holds 3 (busy).
    split = fat.plan_fat_split(flat, top_k=3, num_experts=4, min_rows=3, block_m=4)
    assert split is not None
    assert split.busy == [3]
    assert split.light_expert_ids.tolist() == [0, 1, 2]
    s = num_valid
    assert split.light_sorted_ids.tolist() == [
        1, 4, s, s,   # expert 0: flat positions 1, 4 (stable), padded
        0, 2, s, s,   # expert 1: flat positions 0, 2
        3, 5, s, s,   # expert 2: flat positions 3, 5
    ]
    assert split.light_ntpp == 12


def test_min_rows_env_override(monkeypatch):
    monkeypatch.setenv(fat.ENV_MIN_ROWS, "17")
    assert fat._min_rows() == 17
    monkeypatch.setenv(fat.ENV_MIN_ROWS, "0")
    assert fat._min_rows() == fat.DEFAULT_MIN_ROWS
    monkeypatch.delenv(fat.ENV_MIN_ROWS)
    assert fat._min_rows() == fat.DEFAULT_MIN_ROWS == 32


# ---------------------------------------------------------------------------
# Dequant semantics
# ---------------------------------------------------------------------------


def test_dequant_matches_kernel_layout():
    # byte j holds codes k=2j (lo nibble) and k=2j+1 (hi nibble); one e4m3 scale
    # per 16 k-values (8 bytes); per-row fp16 global folds multiplicatively.
    n, k = 2, 32
    packed = torch.arange(n * k // 2, dtype=torch.int32).reshape(n, k // 2).to(torch.uint8)
    scale = torch.ones(n, k // 16).to(torch.float8_e4m3fn)
    glob = torch.tensor([2.0, 0.5], dtype=torch.float16)
    w = fat.dequant_nvfp4_weight(packed, scale, glob)
    lut = torch.tensor(fat._E2M1_VALUES)
    for row in range(n):
        for j in range(k // 2):
            byte = int(packed[row, j])
            assert w[row, 2 * j].item() == pytest.approx(lut[byte & 0xF].item() * float(glob[row]))
            assert w[row, 2 * j + 1].item() == pytest.approx(lut[(byte >> 4) & 0xF].item() * float(glob[row]))


def test_dequant_scale_blocks():
    n, k = 1, 32
    packed = torch.full((n, k // 2), 0x11, dtype=torch.uint8)  # code 1 -> 0.5
    scale = torch.tensor([[1.0, 2.0]]).to(torch.float8_e4m3fn)
    glob = torch.ones(n, dtype=torch.float16)
    w = fat.dequant_nvfp4_weight(packed, scale, glob)
    assert (w[0, :16] == 0.5).all()
    assert (w[0, 16:] == 1.0).all()


# ---------------------------------------------------------------------------
# Fat forward vs naive reference (CPU, fp32)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("apply_weight_on_input", [False, True])
def test_fat_forward_matches_naive_reference(apply_weight_on_input):
    x, ids, w, banks = fat._synthetic_case("cpu", torch.float32, m=48, seed=3)
    ref = fat._naive_reference(x, ids, w, banks, min_rows=4,
                               apply_router_weight_on_input=apply_weight_on_input)
    got = fat._fat_forward_all_busy(x, ids, w, banks, min_rows=4,
                                    apply_router_weight_on_input=apply_weight_on_input)
    assert fat.rel_err(got, ref) < 1e-5
    assert fat.top1_agreement(got, ref) == 1.0


def test_fat_forward_with_swiglu_limit():
    x, ids, w, banks = fat._synthetic_case("cpu", torch.float32, m=48, seed=5)
    ref = fat._naive_reference(x, ids, w, banks, min_rows=4, limit=10.0)
    got = fat._fat_forward_all_busy(x, ids, w, banks, min_rows=4, limit=10.0)
    assert fat.rel_err(got, ref) < 1e-5


# ---------------------------------------------------------------------------
# Fidelity gate metric + env dispatch
# ---------------------------------------------------------------------------


def test_fidelity_gate_thresholds():
    ref = torch.eye(8)
    assert fat.fidelity_gate(ref.clone(), ref)["passed"]
    flipped = ref.clone()
    flipped[0] = flipped[0, torch.tensor([1, 0] + list(range(2, 8)))]  # 1/8 rows disagree -> 0.875 top1
    gate = fat.fidelity_gate(flipped, ref)
    assert gate["top1"] == pytest.approx(0.875)
    assert not gate["passed"]
    noisy = ref + 0.5
    gate = fat.fidelity_gate(noisy, ref)
    assert gate["rel_err"] > fat.FIDELITY_REL_ERR_MAX
    assert not gate["passed"]


def test_dispatch_off_calls_production_path(monkeypatch):
    monkeypatch.delenv(fat.ENV_FLAG, raising=False)
    sentinel = object()
    called = {}

    def fake_production(*a):
        called["args"] = a
        return sentinel

    monkeypatch.setattr(fnv4, "fused_experts_nvfp4", fake_production)
    args = (torch.zeros(2, 4),) + (None,) * 6 + (torch.zeros(2, 2), torch.zeros(2, 2, dtype=torch.int32), 8, "silu", False, 1.0, float("inf"))
    assert fat.dispatch_prefill(*args) is sentinel
    assert called["args"] == args


def test_dispatch_on_without_busy_falls_back(monkeypatch):
    monkeypatch.setenv(fat.ENV_FLAG, "1")
    monkeypatch.setenv(fat.ENV_MIN_ROWS, "999")
    sentinel = object()
    monkeypatch.setattr(fnv4, "fused_experts_nvfp4", lambda *a: sentinel)
    x = torch.zeros(4, 8)
    ids = torch.tensor([[0, 1], [1, 2], [0, 2], [1, 0]], dtype=torch.int32)
    w = torch.rand(4, 2)
    assert fat.fused_experts_nvfp4_fat(x, None, None, None, None, None, None, w, ids, 3) is sentinel
    assert fat.fat_gemm_stats()["fallback_calls"] >= 1


def test_stats_reset():
    fat.reset_fat_gemm_stats()
    s = fat.fat_gemm_stats()
    assert s["calls"] == 0 and s["busy_experts"] == 0 and s["busy_routes"] == 0
    assert s["min_rows"] == fat.DEFAULT_MIN_ROWS
    assert s["enabled"] is False


# ---------------------------------------------------------------------------
# GPU gate (deferred to the GPU battery): fat vs production grouped, bf16
# ---------------------------------------------------------------------------


@cuda
@pytest.mark.parametrize("apply_weight_on_input", [False, True])
def test_gpu_fat_vs_production_grouped(apply_weight_on_input):
    x, ids, w, banks = fat._synthetic_case("cpu", torch.float32, m=96, seed=11)
    xg = x.to("cuda", torch.bfloat16)
    idsg, wg = ids.to("cuda"), w.to("cuda")
    bg = {k: v.to("cuda") for k, v in banks.items()}
    e = bg["gate_up_packed"].shape[0]
    ref = fnv4.fused_experts_nvfp4(
        xg, bg["gate_up_packed"], bg["gate_up_scale"], bg["gate_up_global"],
        bg["down_packed"], bg["down_scale"], bg["down_global"],
        wg, idsg, e, "silu", apply_weight_on_input,
    )
    got = fat.fused_experts_nvfp4_fat(
        xg, bg["gate_up_packed"], bg["gate_up_scale"], bg["gate_up_global"],
        bg["down_packed"], bg["down_scale"], bg["down_global"],
        wg, idsg, e, "silu", apply_weight_on_input,
    )
    gate = fat.fidelity_gate(got, ref)
    assert gate["passed"], gate
