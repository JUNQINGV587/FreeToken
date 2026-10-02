"""Tests for moe/route_policy.py: LRU parity with the mirror, pin, LFU, and the CLI.

Pure CPU, no torch. ``route_policy`` is loaded by path (it is stdlib-only), and the same
for ``route_trace`` so the policies can be checked against the LRU mirror that is already
locked to ``flashlib.kernels.slot_cache.lru_ensure`` semantics.
"""
from __future__ import annotations

import importlib.util
import json
import random
import struct
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, _ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_rt = _load("route_trace", "python/freetoken/moe/route_trace.py")
_rp = _load("route_policy", "python/freetoken/moe/route_policy.py")

LRU = _rt.LRU
Sim = _rp.Sim
simulate = _rp.simulate
learn_pin_set = _rp.learn_pin_set


def _random_trace(rng, calls: int, layers: int, num_experts: int, k: int, skew: float = 1.0):
    """Zipf-ish per-layer id draws; ``skew`` < 1 flattens the distribution."""
    weights = [(i + 1) ** -skew for i in range(num_experts)]
    return [
        (0, rng.randrange(layers), tuple(rng.choices(range(num_experts), weights, k=k)))
        for _ in range(calls)
    ]


# ----------------------------------------------------------------- Loops: LRU parity
def test_sim_lru_is_identical_to_the_mirror():
    """The one lock that matters: our LRU must be the kernel mirror, miss for miss."""
    rng = random.Random(20261002)
    for trial in range(200):
        cache = rng.choice([1, 2, 3, 4, 8, 16, 64])
        num_experts = rng.choice([8, 32, 512])
        k = rng.choice([1, 2, 4, 8])
        trace = _random_trace(rng, calls=60, layers=rng.choice([1, 2, 5]),
                              num_experts=num_experts, k=k, skew=rng.choice([0.5, 1.0, 2.0]))
        mirror = LRU(cache, num_experts)
        mine = Sim(cache, num_experts, policy="lru")
        for _ph, layer, ids in trace:
            mirror.ensure(layer, ids)
            try:
                mine.ensure(layer, ids)
            except RuntimeError:
                # pool cannot hold one call's distinct ids: the kernel has no valid
                # victim either, so skip the degenerate case rather than diverge
                break
        assert mine.misses == mirror.miss, (
            f"trial {trial}: pool={cache} E={num_experts} k={k} "
            f"sim={mine.misses} mirror={mirror.miss}")
        assert mine.active == mirror.active, f"trial {trial}: active diverged"


def test_lru_stack_property_holds_for_the_sim():
    rng = random.Random(7)
    trace = _random_trace(rng, calls=400, layers=3, num_experts=64, k=4, skew=1.0)
    counts = [simulate(trace, C, 64, policy="lru").misses for C in (8, 16, 32, 64)]
    assert all(a >= b for a, b in zip(counts, counts[1:])), counts


def test_batch_protection_never_evicts_the_call_s_own_row():
    """Pool of 2: the second call's hit on row 2 must keep it, so only row 3 is copied."""
    sim = Sim(2, 512, policy="lru")
    sim.ensure(0, [1, 2])
    before = sim.misses
    sim.ensure(0, [2, 3])
    assert sim.misses - before == 1, (sim.misses, before)


def test_a_call_wider_than_the_pool_is_refused_not_silently_corrupted():
    """C=2 asked for 3 distinct ids: the kernel has no valid victim either.

    The LRU mirror instead re-evicts a row it admitted in the same call; refusing loudly
    is the property the replay relies on, because a silent double-evict would understate
    the miss count.
    """
    sim = Sim(2, 512, policy="lru")
    try:
        sim.ensure(0, [1, 2, 3])
    except RuntimeError as exc:
        assert "no evictable slot" in str(exc)
    else:  # pragma: no cover - the guard is the point of the test
        raise AssertionError("a 3-id call into a 2-slot pool must not silently succeed")


# ----------------------------------------------------------------- Pinning
def _synthetic_stationary(calls: int, layers: int, num_experts: int, k: int,
                          hot: int, hot_share: float, seed: int = 3):
    """Stationary skew: a fixed ``hot`` set supplies ``hot_share`` of activations."""
    rng = random.Random(seed)
    rows = []
    for _ in range(calls):
        layer = rng.randrange(layers)
        ids = []
        for _ in range(k):
            if rng.random() < hot_share:
                ids.append(rng.randrange(hot))
            else:
                ids.append(rng.randrange(hot, num_experts))
        rows.append((0, layer, tuple(ids)))
    return rows


def test_pinned_hit_rate_is_the_static_topk_global_share_up_to_the_fill_copies():
    """Cross-check against ``decode_routing_stats``' own definition of the static figure.

    ``static_topk_hit_global`` is "the share of routed activations that land on the C
    most-activated rows". Recomputed by hand here, the learner must pick exactly those
    rows, and the pin replay must charge no more than the one-off fill copies on top.
    """
    trace = _synthetic_stationary(calls=400, layers=4, num_experts=64, k=4,
                                  hot=8, hot_share=0.85, seed=29)
    C = 32
    counts: dict[int, int] = {}
    active = 0
    for _ph, layer, ids in trace:
        uniq = set(ids)
        active += len(uniq)
        for e in uniq:
            fid = layer * 64 + e
            counts[fid] = counts.get(fid, 0) + 1
    top = sum(sorted(counts.values(), reverse=True)[:C])
    static_hit = top / active

    pin = learn_pin_set(trace, C, 64, warmup_frac=1.0)
    pinned_acts = sum(c for fid, c in counts.items() if fid in pin)
    assert pinned_acts == top, (pinned_acts, top)
    assert abs(pinned_acts / active - static_hit) < 1e-12

    # the replay itself needs a pool strictly wider than the pinned set, so that the rows
    # outside it have somewhere to live; the budget above is what the engine's own
    # static_topk_hit_global pins (C of C), which a dynamic pool cannot host by definition.
    P = 16
    small = learn_pin_set(trace, P, 64, warmup_frac=1.0)
    stripped = [(ph, layer, tuple(e for e in ids if layer * 64 + e not in small))
                for ph, layer, ids in trace]
    managed = simulate(stripped, C - P, 64, policy="lru").misses
    res = simulate(trace, C, 64, policy="pin", pin_rows=small)
    fills = len(small & set(counts))
    assert res.misses == fills + managed, (res.misses, fills, managed)
    assert res.hits == active - res.misses
    assert res.misses < pinned_acts, "a per-hit charge would be far larger than the fills"


def test_pin_model_decomposes_into_a_fill_plus_an_lru_over_the_rest():
    """The pinned pool is exactly "P reserved slots + LRU over C-P" for the other rows."""
    trace = _synthetic_stationary(calls=300, layers=3, num_experts=48, k=4,
                                  hot=10, hot_share=0.8, seed=23)
    C, P = 24, 12
    pin = learn_pin_set(trace, P, 48, warmup_frac=1.0)
    res = simulate(trace, C, 48, policy="pin", pin_rows=pin)

    routed_pins = {layer * 48 + e for _ph, layer, ids in trace
                   for e in ids if layer * 48 + e in pin}
    stripped = [(ph, layer, tuple(e for e in ids if layer * 48 + e not in pin))
                for ph, layer, ids in trace]
    rest = simulate(stripped, C - P, 48, policy="lru")
    assert res.misses == len(routed_pins) + rest.misses, (
        res.misses, len(routed_pins), rest.misses)


def test_pin_reserves_slots_and_never_evicts_a_pinned_row():
    trace = _synthetic_stationary(calls=200, layers=2, num_experts=32, k=4,
                                  hot=6, hot_share=0.9)
    pin = learn_pin_set(trace, 6, 32, warmup_frac=1.0)
    assert len(pin) <= 6
    for _ph, layer, ids in trace:
        for e in ids:
            if layer * 32 + e in pin:
                assert layer * 32 + e in pin  # ids are global row ids
                break
        break
    sim = Sim(10, 32, policy="pin", pin_rows=pin)
    for _ph, layer, ids in trace:
        sim.ensure(layer, ids)
    # no pinned row may ever appear as a managed slot owner
    assert not (set(sim.pin) & set(sim.slot_of)), "a pinned row leaked into the LRU pool"
    assert sim.n_slots == 10 - len(pin)
    # .. and every pinned row that was ever routed was charged its fill copy exactly once
    routed_pins = {layer * 32 + e for _ph, layer, ids in trace for e in ids
                   if layer * 32 + e in pin}
    assert sim._pinned_filled == routed_pins


def test_pin_beats_or_matches_lru_on_a_stationary_hot_set():
    trace = _synthetic_stationary(calls=500, layers=4, num_experts=64, k=4,
                                  hot=8, hot_share=0.85)
    lru = simulate(trace, 24, 64, policy="lru")
    pin = simulate(trace, 24, 64, policy="pin",
                   pin_rows=learn_pin_set(trace, 12, 64, warmup_frac=1.0))
    assert pin.misses < lru.misses, (pin.misses, lru.misses)


def test_causal_pin_set_is_learned_only_from_the_prefix():
    trace = _synthetic_stationary(calls=400, layers=3, num_experts=48, k=4,
                                  hot=10, hot_share=0.8, seed=11)
    cut = int(len(trace) * 0.2)
    seen = {layer * 48 + e for _ph, layer, ids in trace[:cut] for e in set(ids)}
    pin = learn_pin_set(trace, 12, 48, warmup_frac=0.2)
    assert len(pin) <= 12
    assert pin <= seen, "a causal pin set must only draw on the warmup prefix"


# ----------------------------------------------------------------- LFU
def test_lfu_is_not_worse_than_lru_on_a_stationary_skew():
    trace = _synthetic_stationary(calls=500, layers=2, num_experts=64, k=4,
                                  hot=6, hot_share=0.9, seed=5)
    lru = simulate(trace, 20, 64, policy="lru")
    lfu = simulate(trace, 20, 64, policy="lfu")
    assert lfu.misses <= lru.misses, (lfu.misses, lru.misses)


def test_lfu_evicts_the_coldest_by_frequency():
    sim = Sim(2, 8, policy="lfu")
    sim.ensure(0, [1])          # 1: count 1
    sim.ensure(0, [1, 2])       # 1: count 2, 2: count 1
    sim.ensure(0, [3])          # pool full: 2 (count 1, older) must go, not 1
    assert 1 in sim.slot_of and 3 in sim.slot_of


# ----------------------------------------------------------------- plumbing / CLI
def _write_trace(path: Path, records, *, num_experts, num_layers, cache_size, top_k, rank=None):
    rec = _rt.RouteTraceRecorder(str(path), num_experts=num_experts, num_layers=num_layers,
                                 cache_size=cache_size, top_k=top_k, model="synthetic",
                                 rank=rank)
    for phase, layer, ids in records:
        rec._buf += struct.Struct("<bii").pack(phase, layer, len(ids))
        rec._buf += struct.pack(f"<{len(ids)}i", *ids)
        rec._n += 1
    rec.close()
    return rec


def test_cli_replays_a_trace_end_to_end(tmp_path):
    trace = _synthetic_stationary(calls=120, layers=2, num_experts=32, k=4,
                                  hot=6, hot_share=0.8, seed=13)
    body = tmp_path / "route.bin"
    _write_trace(body, trace, num_experts=32, num_layers=2, cache_size=8, top_k=4)
    md = tmp_path / "out.md"
    out = subprocess.run(
        [sys.executable, str(_ROOT / "tools" / "trace" / "replay_route_trace.py"),
         str(body), "--cache-sizes", "8,12", "--policies", "lru,lfu,pin",
         "--pin-fracs", "0.5", "--row-mib", "74", "--md", str(md)],
        capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "| trace | policy |" in out.stdout
    # 2 pool sizes x (lru, lfu, pin) rows
    assert out.stdout.count("\n| route.bin |") == 6, out.stdout
    assert "pin50%" in out.stdout
    text = md.read_text()
    assert "copies priced at" in text and "| lru | 8 |" in text


def test_cli_reports_per_rank_traces(tmp_path):
    trace = _synthetic_stationary(calls=60, layers=1, num_experts=16, k=3,
                                  hot=4, hot_share=0.8, seed=17)
    base = tmp_path / "r.bin"
    for rank in (0, 1):
        _write_trace(base, trace, num_experts=16, num_layers=1, cache_size=8, top_k=3,
                     rank=rank)
    out = subprocess.run(
        [sys.executable, str(_ROOT / "tools" / "trace" / "replay_route_trace.py"),
         str(base), "--ranks", "--cache-sizes", "8", "--policies", "lru"],
        capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "r.bin.rank0" in out.stdout and "r.bin.rank1" in out.stdout


def test_cli_says_what_is_wrong_with_a_missing_trace(tmp_path):
    out = subprocess.run(
        [sys.executable, str(_ROOT / "tools" / "trace" / "replay_route_trace.py"),
         str(tmp_path / "nope.bin")],
        capture_output=True, text=True)
    assert out.returncode != 0
    assert "no such trace" in out.stderr


def test_wipe_band_makes_the_borrowed_slots_the_first_victims():
    """Owner-EP prefill overlap invalidates its borrowed slots at each chunk boundary.

    ``offload_cache._invalidate_prefill_buffer`` clears ``slot_for_id`` and zeroes
    ``usage`` over the borrowed range, so those slots become the coldest in the pool and
    the LRU side refills them first. Modelled by ``Sim.wipe_band``.
    """
    sim = Sim(8, 512, policy="lru", band_slots=4)
    for e in range(12):
        sim.ensure(0, [e])
    dropped = sim.wipe_band()
    assert dropped > 0, "the band must have held live rows"
    before = sim.misses
    sim.ensure(0, [99])
    assert sim.misses == before + 1, "a fresh row after the wipe is still a miss"
    assert sim.slot_of[99] < 4, "the refill must land in the wiped band"


def test_a_zero_band_wipe_is_a_no_op():
    sim = Sim(16, 64, policy="lru", band_slots=0)
    trace = _synthetic_stationary(calls=120, layers=2, num_experts=64, k=4,
                                  hot=6, hot_share=0.8, seed=11)
    for _ph, layer, ids in trace:
        sim.ensure(layer, ids)
    misses, calls = sim.misses, sim.calls
    assert sim.band == 0
    assert sim.wipe_band() == 0
    assert (sim.misses, sim.calls) == (misses, calls)
