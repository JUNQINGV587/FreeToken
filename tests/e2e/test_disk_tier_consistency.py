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
    if args.ep_size > 1:
        _worker_owner(args)
        return
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
        # decode read path, so overlap stays off unless --overlap asks for it
        # (which also grows the cache to fit the ring plus decode slack).
        moe_prefill_overlap=bool(args.overlap),
        moe_disk_tier=args.disk_tier,
        cuda_graph_max_bs=1,
    )
    if args.overlap:
        kwargs["moe_cache_size"] = 2 * 512 + 256
    if args.disk_tier != "off":
        kwargs["expert_ram_experts"] = args.ram_experts
        # engine.py:857 -- disk-tier v0 requires cuda graphs disabled.
        kwargs["cuda_graph_max_bs"] = 0
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


def _free_port() -> int:
    import socket
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _post_json(base: str, path: str, payload: dict, timeout: float):
    import urllib.error
    import urllib.request
    request = urllib.request.Request(
        base + path, data=json.dumps(payload).encode(),
        headers={"content-type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _worker_owner(args: argparse.Namespace) -> None:
    """Owner-local EP config: the offline LLM class is hardwired single-rank
    (tp_info=DistributedInfo(0, 1)), so boot the real server with TP/EP and
    drive greedy /v1/completions over HTTP -- same topology as production."""
    import tempfile
    import time
    import urllib.error
    import urllib.request

    repo_python = Path(__file__).resolve().parents[2] / "python"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_python)
    if args.prefetch > 0:
        env["FT_DISK_TIER_PREFETCH"] = str(args.prefetch)
    port = _free_port()
    base = f"http://127.0.0.1:{port}"
    log = tempfile.NamedTemporaryFile(
        mode="w", prefix="owner-gate-", suffix=".log", delete=False)
    cmd = [sys.executable, "-m", "freetoken",
           "--model-path", args.model,
           "--served-model-name", "owner-gate",
           "--host", "127.0.0.1", "--port", str(port),
           "--tensor-parallel-size", str(args.ep_size),
           "--moe-ep-size", str(args.ep_size),
           "--moe-strategy", "offload", "--moe-cache-policy", "lru",
           "--moe-cache-size", os.environ.get("FREETOKEN_TEST_MOE_CACHE_SIZE", "512"),
           # Graphs stay at the same setting in reference and test configs so
           # the tier is the only difference between them.
           "--cuda-graph-max-bs", str(args.graph_bs),
           "--max-running-requests", "1"]
    if not args.overlap:
        # Overlap defaults on (config.py); keep the historical gate behaviour
        # unless --overlap opts in.
        cmd.append("--disable-moe-prefill-overlap")
    if args.disk_tier != "off":
        cmd += ["--moe-disk-tier", "on",
                "--expert-ram-experts", str(args.ram_experts)]
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)

    def _serving(deadline: float) -> None:
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                log.flush()
                raise RuntimeError(
                    f"owner-gate server exited early (rc={proc.returncode}); "
                    f"log tail:\n{Path(log.name).read_text()[-3000:]}")
            try:
                with urllib.request.urlopen(base + "/v1/cache/status", timeout=10) as r:
                    if json.loads(r.read())["state"] == "serving":
                        return
            except (urllib.error.URLError, OSError, KeyError, json.JSONDecodeError):
                pass
            time.sleep(2.0)
        raise TimeoutError("owner-gate server never reached the serving state")

    max_tokens = int(os.environ.get("FREETOKEN_TEST_MAX_TOKENS", "48"))

    def _gen(prompt: str) -> str:
        code, body = _post_json(
            base, "/v1/completions",
            {"model": "owner-gate", "prompt": prompt,
             "temperature": 0.0, "max_tokens": max_tokens},
            timeout=600)
        if code != 200:
            raise RuntimeError(f"completion failed ({code}): {body}")
        return str(body["choices"][0]["text"])

    try:
        _serving(time.monotonic() + 1200)
        cold = [_gen(p) for p in PROMPTS]
        warm = [_gen(p) for p in PROMPTS]
        print(MARK + json.dumps({"cold": cold, "warm": warm}))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            proc.kill()


def _run_config(model_path: Path, disk_tier: str, ram_experts: int,
                prefetch_window: int, ep_size: int = 1,
                graph_bs: int = 0, overlap: int = 0) -> dict:
    env = dict(os.environ)
    env.setdefault("PYTHONPATH", str(Path(__file__).resolve().parents[2] / "python"))
    proc = subprocess.run(
        [sys.executable, __file__, "--worker",
         "--model", str(model_path), "--disk-tier", disk_tier,
         "--ram-experts", str(ram_experts), "--prefetch", str(prefetch_window),
         "--ep-size", str(ep_size), "--graph-bs", str(graph_bs),
         "--overlap", str(overlap)],
        capture_output=True, text=True, timeout=1800, env=env)
    for line in proc.stdout.splitlines():
        if line.startswith(MARK):
            return json.loads(line[len(MARK):])
    raise AssertionError(
        f"worker for disk_tier={disk_tier} prefetch={prefetch_window} ep={ep_size} "
        f"graph_bs={graph_bs} produced no result (rc={proc.returncode}):\n"
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
        # Prefill-overlap × disk tier: the ring streams only the pinned RAM
        # prefix and the routed disk rows are patched into the borrowed buffer
        # at layer entry (DiskTier.fetch_routed_into). TP1 reference is the
        # tier with overlap off (same topology, so greedy tokens are
        # comparable).
        "disk_overlap": _run_config(
            model_path, "on", ram_experts, prefetch_window=0, overlap=1),
    }
    if torch.cuda.device_count() >= 2:
        # Owner-local EP (TP2+EP2): each rank's tier serves only its owned
        # experts, translated from the global checkpoint rows. The reference
        # must be the SAME TP2 topology with the tier off -- TP changes
        # all-reduce summation order, so a TP1 baseline legitimately diverges
        # from TP2 under greedy decoding.
        results["owner_ref"] = _run_config(
            model_path, "off", ram_experts=0, prefetch_window=0, ep_size=2)
        results["owner"] = _run_config(
            model_path, "on", ram_experts, prefetch_window=1, ep_size=2)
        # CUDA-graph variants of the same pair: the tiered config exercises the
        # graph-doorbell fetch path (moe/graph_fetch.py) recorded into the
        # decode graphs; the reference is the same topology, tier off.
        graph_bs = int(os.environ.get("FREETOKEN_TEST_GRAPH_BS", "8"))
        results["owner_graph_ref"] = _run_config(
            model_path, "off", ram_experts=0, prefetch_window=0, ep_size=2,
            graph_bs=graph_bs)
        results["owner_graph"] = _run_config(
            model_path, "on", ram_experts, prefetch_window=1, ep_size=2,
            graph_bs=graph_bs)
        # Full production stack: owner-EP + CUDA graphs + prefill overlap, with
        # and without the disk tier. Same TP2 topology both sides, so the
        # greedy tokens must be byte-identical.
        results["owner_full_ref"] = _run_config(
            model_path, "off", ram_experts=0, prefetch_window=0, ep_size=2,
            graph_bs=graph_bs, overlap=1)
        results["owner_full"] = _run_config(
            model_path, "on", ram_experts, prefetch_window=1, ep_size=2,
            graph_bs=graph_bs, overlap=1)
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
    the SAME TP2 topology with the tier off: a namespace slip would fetch a
    remote rank's expert rows and corrupt the tokens. (TP1-vs-TP2 comparison
    is invalid -- all-reduce order legitimately diverges under greedy.)"""
    owner = gate_results.get("owner")
    owner_ref = gate_results.get("owner_ref")
    if owner is None or owner_ref is None:
        pytest.skip("owner-EP gate needs >= 2 GPUs")
    base = owner_ref["cold"]
    assert owner["cold"] == base, _diff(
        "owner-EP cold decode (local<->global translation wrong?)",
        owner["cold"], base)
    assert owner["warm"] == base, _diff("owner-EP warm decode", owner["warm"], base)


@pytest.mark.needs_weights
@pytest.mark.skipif(not torch.cuda.is_available(), reason="disk-tier gate needs CUDA")
def test_disk_tier_cuda_graphs_greedy_matches_graphed_reference(gate_results):
    """CUDA graphs ON: the disk-tiered owner-EP engine (graph-doorbell fetch,
    moe/graph_fetch.py) must be token-identical to the same graphed topology
    with the tier off. A doorbell/staging/install bug would surface here as
    wrong expert bytes under replay; a capture-time bug fails the boot."""
    owner = gate_results.get("owner_graph")
    owner_ref = gate_results.get("owner_graph_ref")
    if owner is None or owner_ref is None:
        pytest.skip("owner-EP graph gate needs >= 2 GPUs")
    base = owner_ref["cold"]
    assert owner["cold"] == base, _diff(
        "graphed owner-EP cold decode (doorbell fetch wrong?)",
        owner["cold"], base)
    assert owner["warm"] == base, _diff(
        "graphed owner-EP warm decode", owner["warm"], base)


@pytest.mark.needs_weights
@pytest.mark.skipif(not torch.cuda.is_available(), reason="disk-tier gate needs CUDA")
def test_disk_tier_overlap_greedy_matches_tier_only(gate_results):
    """Prefill overlap ON with the disk tier (TP1): the overlap ring streams
    only the pinned RAM prefix and DiskTier.fetch_routed_into patches the
    routed disk rows into the borrowed buffer at layer entry. Tokens must be
    identical to the same tiered topology with overlap off (a ring-buffer
    scribble or a missing disk-row patch corrupts the prefill)."""
    disk = gate_results["disk"]
    overlap = gate_results["disk_overlap"]
    assert overlap["cold"] == disk["cold"], _diff(
        "overlap cold prefill (buffer rows stale / disk rows missing?)",
        overlap["cold"], disk["cold"])
    assert overlap["warm"] == disk["cold"], _diff(
        "overlap warm decode", overlap["warm"], disk["cold"])


@pytest.mark.needs_weights
@pytest.mark.skipif(not torch.cuda.is_available(), reason="disk-tier gate needs CUDA")
def test_disk_tier_full_stack_greedy_matches_reference(gate_results):
    """The production stack -- owner-EP (TP2) + CUDA graphs + prefill overlap +
    disk tier -- must be token-identical to the same topology with the tier
    off. This is the config the production server would actually run."""
    owner = gate_results.get("owner_full")
    owner_ref = gate_results.get("owner_full_ref")
    if owner is None or owner_ref is None:
        pytest.skip("full-stack gate needs >= 2 GPUs")
    base = owner_ref["cold"]
    assert owner["cold"] == base, _diff(
        "full-stack cold decode (overlap buffer patch or doorbell wrong?)",
        owner["cold"], base)
    assert owner["warm"] == base, _diff(
        "full-stack warm decode", owner["warm"], base)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--model", required=True)
    ap.add_argument("--disk-tier", choices=["off", "on"], required=True)
    ap.add_argument("--ram-experts", type=int, default=0)
    ap.add_argument("--prefetch", type=int, default=0)
    ap.add_argument("--ep-size", type=int, default=1)
    ap.add_argument("--graph-bs", type=int, default=0,
                    help="cuda-graph max bs (0 = graphs disabled)")
    ap.add_argument("--overlap", type=int, default=0,
                    help="1 = keep prefill overlap on (default strips it)")
    ns = ap.parse_args()
    if ns.worker:
        _worker(ns)
