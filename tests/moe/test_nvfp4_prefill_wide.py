"""Bit-contract tests for the wide-load NVFP4 prefill kernel + its env dispatch.

``_prefill_nvfp4_moe_wide_kernel`` must stay ``torch.equal`` to the production
``_prefill_nvfp4_moe_kernel`` on every input (spec
``202609-nvfp4-prefill-kernel-spec.md``: KB=32 lo->hi double-dot fp32 order, same
epilogue), and the ``FREETOKEN_NVFP4_PREFILL_WIDE`` dispatch must reroute only
M >= 2048. Small geometry (E=32, H=2560, I=640, top_k=10) keeps CI cost bounded;
the full matrix (M up to 12288, config scan, numeric corners) lives in
``research/bench/check_nvfp4_prefill_wide_parity.py``.
"""

from __future__ import annotations

import sys

import pytest
import torch

import freetoken.moe.fused_nvfp4 as fnv4
import freetoken.moe.fused_nvfp4_wide as fnv4w

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

E, H, I, TOP_K = 32, 2560, 640, 10
ENV = fnv4w.WIDE_ENV


@pytest.fixture(autouse=True)
def _restore_dispatch(monkeypatch):
    """The dispatch rebinds ``fnv4._prefill_gemm``; never leak that into other tests."""
    yield
    fnv4w.restore_prefill(fnv4.__dict__)
    assert fnv4._prefill_gemm is fnv4w._ORIG_PREFILL_GEMM


@pytest.fixture(scope="module")
def weights():
    torch.manual_seed(0)
    dev = "cuda"
    return dict(
        gate_up_packed=torch.randint(0, 256, (E, 2 * I, H // 2), device=dev, dtype=torch.uint8),
        gate_up_scale=torch.rand(E, 2 * I, H // 16, device=dev, dtype=torch.float16).to(torch.float8_e4m3fn),
        gate_up_global=torch.randn(E, 2 * I, device=dev, dtype=torch.float16),
        down_packed=torch.randint(0, 256, (E, H, I // 2), device=dev, dtype=torch.uint8),
        down_scale=torch.rand(E, H, I // 16, device=dev, dtype=torch.float16).to(torch.float8_e4m3fn),
        down_global=torch.randn(E, H, device=dev, dtype=torch.float16),
    )


def _routes(M: int, mode: str) -> torch.Tensor:
    gen = torch.Generator().manual_seed(42 + M)
    if mode == "uniform":
        return torch.randint(0, E, (M, TOP_K), generator=gen).to("cuda")
    ranks = torch.arange(1, E + 1, dtype=torch.float64)
    probs = 1.0 / ranks.pow(1.1)
    probs /= probs.sum()
    perm = torch.randperm(E, generator=gen)
    idx = torch.multinomial(probs, M * TOP_K, replacement=True, generator=gen)
    return perm[idx].reshape(M, TOP_K).to("cuda")


def _run(fn, hidden, W, topk_w, topk_ids):
    return fn(hidden, W["gate_up_packed"], W["gate_up_scale"], W["gate_up_global"],
              W["down_packed"], W["down_scale"], W["down_global"], topk_w, topk_ids, E)


@cuda
@pytest.mark.parametrize("M", [2048, 4096])
@pytest.mark.parametrize("mode", ["zipf", "uniform"])
def test_wide_matches_production_bitwise(weights, M, mode, monkeypatch):
    monkeypatch.delenv(ENV, raising=False)  # production entry must take the production kernel
    torch.manual_seed(M)
    hidden = torch.randn(M, H, device="cuda", dtype=torch.bfloat16)
    topk_w = torch.rand(M, TOP_K, device="cuda", dtype=torch.float32)
    topk_ids = _routes(M, mode)
    ref = _run(fnv4.fused_experts_nvfp4, hidden, weights, topk_w, topk_ids)
    out = _run(fnv4w.fused_experts_nvfp4_wide, hidden, weights, topk_w, topk_ids)
    assert torch.equal(ref, out)


def test_dispatch_threshold(monkeypatch):
    monkeypatch.setenv(ENV, "1")
    orig = fnv4w._ORIG_PREFILL_GEMM
    cfg = fnv4._prefill_config(2047)
    assert fnv4._prefill_gemm is orig  # below the tier: production path untouched
    assert cfg["BLOCK_SIZE_M"] == 32
    cfg = fnv4._prefill_config(2048)
    assert fnv4._prefill_gemm is fnv4w._prefill_gemm_wide
    assert cfg["BLOCK_SIZE_M"] == fnv4w._prefill_config_wide(2048)["BLOCK_SIZE_M"]
    assert cfg["BLOCK_SIZE_KB"] == 32  # bit contract C2 holds in the wide table


def test_env_off_restores_production_binding(monkeypatch):
    orig = fnv4w._ORIG_PREFILL_GEMM
    monkeypatch.setenv(ENV, "1")
    fnv4._prefill_config(2048)
    assert fnv4._prefill_gemm is fnv4w._prefill_gemm_wide
    monkeypatch.setenv(ENV, "0")
    fnv4._prefill_config(2048)
    assert fnv4._prefill_gemm is orig
    monkeypatch.delenv(ENV)
    fnv4._prefill_config(8192)
    assert fnv4._prefill_gemm is orig


def test_env_off_never_imports_wide(monkeypatch):
    """M1: with the fuse off, production must not touch the wide module at all --
    a broken/moved wide module cannot fail a production prefill."""
    monkeypatch.delenv(ENV, raising=False)
    saved = sys.modules.pop("freetoken.moe.fused_nvfp4_wide", None)
    try:
        cfg = fnv4._prefill_config(2048)
        assert "freetoken.moe.fused_nvfp4_wide" not in sys.modules
        assert cfg["BLOCK_SIZE_M"] == 128  # production row
    finally:
        if saved is not None:
            sys.modules["freetoken.moe.fused_nvfp4_wide"] = saved
