"""P0-5 consistency gate: the NVMe disk tier must be token-identical to the
all-RAM offload baseline under greedy decode -- first run (cold slots) and
repeat run (warm slots) alike. This is the acceptance gate for the disk-tier
read path (P0-3 merged reads, P0-4 PILOT prefetch): any divergence means an
expert row landed in the wrong slot, was torn by a slab/pinned-buffer race, or
the miss-list rewrite dropped a RAM expert.

Env:
    FREETOKEN_TEST_MODEL          local NVFP4 MoE checkpoint (required)
    FREETOKEN_TEST_MOE_CACHE_SIZE slot count for both runs (default 512)
    FREETOKEN_TEST_MAX_TOKENS     greedy tokens per prompt (default 48)
"""
from __future__ import annotations

import gc
import json
import os
from pathlib import Path

import pytest
import torch

from freetoken.core import SamplingParams
from freetoken.llm import LLM

PROMPTS = [
    "Give me a short introduction to the French Revolution.",
    "Write a Python function that returns the n-th Fibonacci number.",
    "List three differences between TCP and UDP.",
]


def _optional_path(env: str) -> Path | None:
    raw = os.environ.get(env, "").strip()
    if not raw:
        return None
    return Path(os.path.expanduser(raw)).expanduser().resolve()


def _num_experts(model_path: Path) -> int:
    cfg = json.loads((model_path / "config.json").read_text())
    for key in ("num_experts", "n_routed_experts", "moe_num_experts"):
        if key in cfg:
            return int(cfg[key])
    pytest.fail(f"cannot find an expert count in {model_path}/config.json")


def _greedy_texts(model_path: Path, disk_tier: str, ram_experts: int,
                  prefetch_window: int) -> list[str]:
    kwargs = dict(
        model_path=str(model_path),
        dtype=torch.bfloat16,
        attention_backend="auto",
        max_running_req=1,
        max_extend_tokens=8192,
        moe_strategy="offload",
        moe_cache_policy="lru",
        moe_cache_size=int(os.environ.get("FREETOKEN_TEST_MOE_CACHE_SIZE", "512")),
        moe_disk_tier=disk_tier,
        cuda_graph_max_bs=1,
    )
    if disk_tier != "off":
        kwargs["expert_ram_experts"] = ram_experts
    if prefetch_window > 0:
        os.environ["FT_DISK_TIER_PREFETCH"] = str(prefetch_window)
    else:
        os.environ.pop("FT_DISK_TIER_PREFETCH", None)
    llm = LLM(**kwargs)
    try:
        sampling = SamplingParams(
            temperature=0.0,
            max_tokens=int(os.environ.get("FREETOKEN_TEST_MAX_TOKENS", "48")),
        )
        return [str(llm.generate([p], sampling)[0]["text"]) for p in PROMPTS]
    finally:
        llm.shutdown()
        del llm
        gc.collect()
        torch.cuda.empty_cache()


@pytest.mark.needs_weights
@pytest.mark.skipif(not torch.cuda.is_available(), reason="disk-tier gate needs CUDA")
def test_disk_tier_greedy_matches_all_ram_baseline():
    model_path = _optional_path("FREETOKEN_TEST_MODEL")
    if model_path is None:
        pytest.skip("set FREETOKEN_TEST_MODEL to a local NVFP4 MoE checkpoint")
    if not model_path.is_dir():
        pytest.skip(f"model is not downloaded: {model_path}")

    n_experts = _num_experts(model_path)
    ram_experts = n_experts // 2  # half the experts resolve from NVMe

    baseline = _greedy_texts(model_path, "off", ram_experts=0, prefetch_window=0)
    cold = _greedy_texts(model_path, "on", ram_experts, prefetch_window=0)
    assert cold == baseline, (
        "disk-tier cold decode diverged from the all-RAM baseline:\n"
        + "\n".join(f"  prompt {i}: {c!r} != {b!r}"
                    for i, (c, b) in enumerate(zip(cold, baseline)) if c != b))

    # Warm pass: the slots populated by the cold run change which misses hit
    # the stash/disk vs RAM, so the second run re-exercises the boundary.
    warm = _greedy_texts(model_path, "on", ram_experts, prefetch_window=0)
    assert warm == baseline, "disk-tier warm decode diverged from the baseline"


@pytest.mark.needs_weights
@pytest.mark.skipif(not torch.cuda.is_available(), reason="disk-tier gate needs CUDA")
def test_disk_tier_prefetch_greedy_matches_baseline():
    """PILOT on: prefetched experts must produce the same greedy tokens."""
    model_path = _optional_path("FREETOKEN_TEST_MODEL")
    if model_path is None:
        pytest.skip("set FREETOKEN_TEST_MODEL to a local NVFP4 MoE checkpoint")
    if not model_path.is_dir():
        pytest.skip(f"model is not downloaded: {model_path}")

    n_experts = _num_experts(model_path)
    ram_experts = n_experts // 2

    baseline = _greedy_texts(model_path, "off", ram_experts=0, prefetch_window=0)
    prefetched = _greedy_texts(model_path, "on", ram_experts, prefetch_window=1)
    assert prefetched == baseline, (
        "PILOT-prefetched decode diverged from the all-RAM baseline -- the stash "
        "path is writing wrong bytes into slots")
