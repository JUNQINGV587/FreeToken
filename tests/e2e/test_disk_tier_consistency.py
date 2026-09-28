"""P0-5 consistency gate: the NVMe disk tier must be token-identical to the
all-RAM offload baseline under greedy decode -- first pass (cold slots) and
repeat pass (warm slots) alike. This is the acceptance gate for the disk-tier
read path (P0-3 merged reads, P0-4 PILOT prefetch): any divergence means an
expert row landed in the wrong slot, was torn by a slab/pinned-buffer race, or
the miss-list rewrite dropped a RAM expert.

Each configuration runs in a SUBPROCESS (the engine asserts
``not torch.cuda.is_initialized()`` at construction, and CUDA cannot be
de-initialized in-process). The worker generates twice against the same LLM
instance: pass 1 exercises cold slots, pass 2 the warm/stash boundary.

Env:
    FREETOKEN_TEST_MODEL          local NVFP4 MoE checkpoint (required)
    FREETOKEN_TEST_MOE_CACHE_SIZE slot count for both runs (default 512)
    FREETOKEN_TEST_MAX_TOKENS     greedy tokens per prompt (default 48)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

PROMPTS = [
    "Give me a short introduction to the French Revolution.",
    "Write a Python function that returns the n-th Fibonacci number.",
    "List three differences between TCP and UDP.",
]

MARK = "RESULT_JSON:"


def _optional_path(env: str) -> Path | None:
    raw = os.environ.get(env, "").strip()
    if not raw:
        return None
    return Path(os.path.expanduser(raw)).expanduser().resolve()


def _num_experts(model_path: Path) -> int:
    cfg = json.loads((model_path / "config.json").read_text())
    # Multimodal checkpoints nest the LM under text_config (Qwen3.8-Flash-Next).
    sub = cfg.get("text_config")
    if isinstance(sub, dict):
        cfg = {**cfg, **sub}
    for key in ("num_experts", "n_routed_experts", "moe_num_experts"):
        if key in cfg:
            return int(cfg[key])
    pytest.fail(f"cannot find an expert count in {model_path}/config.json")


def _worker(args: argparse.Namespace) -> None:
    """Runs in a fresh process: build one LLM, generate twice, print JSON."""
    from freetoken.core import SamplingParams
    from freetoken.llm import LLM

    if args.prefetch > 0:
        os.environ["FT_DISK_TIER_PREFETCH"] = str(args.prefetch)
    kwargs = dict(
        model_path=args.model,
        dtype=torch.bfloat16,
        attention_backend="auto",
        max_running_req=1,
        max_extend_tokens=8192,
        moe_strategy="offload",
        moe_cache_policy="lru",
        moe_cache_size=int(os.environ.get("FREETOKEN_TEST_MOE_CACHE_SIZE", "512")),
        # Prefill overlap borrows 2*num_experts slots; a 512-slot cache on a
        # 512-expert model leaves nothing for decode. The gate targets the
        # decode read path, so overlap stays off here.
        moe_prefill_overlap=False,
        moe_disk_tier=args.disk_tier,
        cuda_graph_max_bs=1,
    )
    if args.disk_tier != "off":
        kwargs["expert_ram_experts"] = args.ram_experts
        # engine.py:857 -- disk-tier v0 requires cuda graphs disabled.
        kwargs["cuda_graph_max_bs"] = 0
    if args.ep_size > 1:
        # Owner-local EP: the tier must translate local<->global expert rows.
        kwargs["tensor_parallel_size"] = args.ep_size
        kwargs["moe_ep_size"] = args.ep_size
    llm = LLM(**kwargs)
    try:
        sampling = SamplingParams(
            temperature=0.0,
            max_tokens=int(os.environ.get("FREETOKEN_TEST_MAX_TOKENS", "48")),
        )
        cold = [str(llm.generate([p], sampling)[0]["text"]) for p in PROMPTS]
        warm = [str(llm.generate([p], sampling)[0]["text"]) for p in PROMPTS]
        print(MARK + json.dumps({"cold": cold, "warm": warm}))
    finally:
        llm.shutdown()


def _run_config(model_path: Path, disk_tier: str, ram_experts: int,
                prefetch_window: int, ep_size: int = 1) -> dict:
    env = dict(os.environ)
    env.setdefault("PYTHONPATH", str(Path(__file__).resolve().parents[2] / "python"))
    proc = subprocess.run(
        [sys.executable, __file__, "--worker",
         "--model", str(model_path), "--disk-tier", disk_tier,
         "--ram-experts", str(ram_experts), "--prefetch", str(prefetch_window),
         "--ep-size", str(ep_size)],
        capture_output=True, text=True, timeout=1800, env=env)
    for line in proc.stdout.splitlines():
        if line.startswith(MARK):
            return json.loads(line[len(MARK):])
    raise AssertionError(
        f"worker for disk_tier={disk_tier} prefetch={prefetch_window} ep={ep_size} "
        f"produced no result (rc={proc.returncode}):\n"
        f"stdout tail:\n{proc.stdout[-2000:]}\nstderr tail:\n{proc.stderr[-2000:]}")


@pytest.fixture(scope="module")
def gate_results() -> dict:
    model_path = _optional_path("FREETOKEN_TEST_MODEL")
    if model_path is None:
        pytest.skip("set FREETOKEN_TEST_MODEL to a local NVFP4 MoE checkpoint")
    if not model_path.is_dir():
        pytest.skip(f"model is not downloaded: {model_path}")
    n_experts = _num_experts(model_path)
    ram_experts = n_experts // 2  # half the experts resolve from NVMe
    results = {
        "baseline": _run_config(model_path, "off", ram_experts=0, prefetch_window=0),
        "disk": _run_config(model_path, "on", ram_experts, prefetch_window=0),
        "prefetch": _run_config(model_path, "on", ram_experts, prefetch_window=1),
    }
    if torch.cuda.device_count() >= 2:
        # Owner-local EP (TP2+EP2): each rank's tier serves only its owned
        # experts, translated from the global checkpoint rows.
        results["owner"] = _run_config(
            model_path, "on", ram_experts, prefetch_window=1, ep_size=2)
    return results


def _diff(name: str, got: list[str], want: list[str]) -> str:
    return (f"{name} diverged from the all-RAM baseline:\n"
            + "\n".join(f"  prompt {i}: {g!r} != {w!r}"
                        for i, (g, w) in enumerate(zip(got, want)) if g != w))


@pytest.mark.needs_weights
@pytest.mark.skipif(not torch.cuda.is_available(), reason="disk-tier gate needs CUDA")
def test_disk_tier_greedy_matches_all_ram_baseline(gate_results):
    base = gate_results["baseline"]["cold"]
    disk = gate_results["disk"]
    # Cold pass: slots empty, every disk expert comes from NVMe reads.
    assert disk["cold"] == base, _diff("disk-tier cold decode", disk["cold"], base)
    # Warm pass: slots populated by the cold pass change which misses hit the
    # stash/RAM vs disk boundary -- the second pass re-exercises it.
    assert disk["warm"] == base, _diff("disk-tier warm decode", disk["warm"], base)


@pytest.mark.needs_weights
@pytest.mark.skipif(not torch.cuda.is_available(), reason="disk-tier gate needs CUDA")
def test_disk_tier_prefetch_greedy_matches_baseline(gate_results):
    """PILOT on: prefetched experts must produce the same greedy tokens."""
    base = gate_results["baseline"]["cold"]
    pf = gate_results["prefetch"]
    assert pf["cold"] == base, _diff(
        "PILOT cold decode (stash path writing wrong bytes?)", pf["cold"], base)
    assert pf["warm"] == base, _diff("PILOT warm decode", pf["warm"], base)


@pytest.mark.needs_weights
@pytest.mark.skipif(not torch.cuda.is_available(), reason="disk-tier gate needs CUDA")
def test_disk_tier_owner_ep_greedy_matches_baseline(gate_results):
    """Owner-local EP (TP2+EP2) with the disk tier must be token-identical to
    the single-rank all-RAM baseline: a namespace slip would fetch a remote
    rank's expert rows and corrupt the tokens."""
    owner = gate_results.get("owner")
    if owner is None:
        pytest.skip("owner-EP gate needs >= 2 GPUs")
    base = gate_results["baseline"]["cold"]
    assert owner["cold"] == base, _diff(
        "owner-EP cold decode (local<->global translation wrong?)",
        owner["cold"], base)
    assert owner["warm"] == base, _diff("owner-EP warm decode", owner["warm"], base)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--model", required=True)
    ap.add_argument("--disk-tier", choices=["off", "on"], required=True)
    ap.add_argument("--ram-experts", type=int, default=0)
    ap.add_argument("--prefetch", type=int, default=0)
    ap.add_argument("--ep-size", type=int, default=1)
    ns = ap.parse_args()
    if ns.worker:
        _worker(ns)
