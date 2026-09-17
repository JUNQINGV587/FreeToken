"""freq_pin hybrid pinning policy (spec: 202609-cache-freqpin-spec.md).

Covers the eviction-correctness contract (spec section 3):
  * cold start (empty histogram / pin region inactive) is bit-identical to plain LRU
    -- against flashlib's lru_ensure on GPU, and miss/residency-identical to the
    route_trace.LRU mirror on CPU (free-slot fill order is not part of the lock);
  * GPU kernel == CPU reference mirror == route_trace.FreqPinLRU, pins active;
  * pinned ids are never evicted by non-pinned misses, pinned misses are
    force-placed into the pin region [2E, 2E+K), and a pinned occupant of the pin
    region is only evicted as a last resort (pinned-vs-pinned);
  * the prefill borrow region [0, 2E) is never pinned and its per-chunk
    invalidation leaves the pin set untouched;
  * batch protection (a slot touched by the current call is never its victim)
    holds with pins active;
  * maybe_repin: warmup gate, first-pin placement with byte verification,
    hysteresis-bounded repin delta, in-flight slot deferral, rebuild reset.
"""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from freetoken.moe.offload_cache import OffloadMoeCache

# route_trace.py is stdlib-only by design; load it by path (same convention as
# tests/moe/test_route_trace.py) so the mirror check needs no package import.
_MOD = Path(__file__).resolve().parents[2] / "python" / "freetoken" / "moe" / "route_trace.py"
_spec = importlib.util.spec_from_file_location("route_trace", _MOD)
_rt = importlib.util.module_from_spec(_spec)
sys.modules["route_trace"] = _rt
_spec.loader.exec_module(_rt)

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _make_cache(policy="freq_pin", L=2, E=8, C=32, device="cpu", pin_slots=16, **kw):
    return OffloadMoeCache(
        num_layers=L,
        num_experts=E,
        cache_size=C,
        device=torch.device(device),
        cache_policy=policy,
        pin_slots=pin_slots,
        quant_format="bf16",
        **kw,
    )


def _with_banks(cache, seed=0):
    """bf16 banks with 128B-multiple rows so every copy path is exercised."""
    g = torch.Generator().manual_seed(seed)
    L, E = cache.num_layers, cache.num_experts
    sources = {
        "gate_up": [torch.randn(E, 16, 16, generator=g) for _ in range(L)],
        "down": [torch.randn(E, 16, 8, generator=g) for _ in range(L)],
    }
    cache.set_bank_sources(sources)
    return sources


def _activate_pins(cache, pinned_flat_ids, k):
    cache.pin_id_mask.zero_()
    if pinned_flat_ids:
        idx = torch.tensor(sorted(pinned_flat_ids), dtype=torch.long, device=cache.device)
        cache.pin_id_mask[idx] = 1
    cache.pin_cfg[1] = k


def _random_steps(L, E, n, top_k, seed):
    g = torch.Generator().manual_seed(seed)
    return [
        (layer, torch.randperm(E, generator=g)[:top_k].to(torch.int32))
        for layer in range(L)
        for _ in range(n)
    ]


def _assert_state_equal(a, b, msg=""):
    assert torch.equal(a.slot_for_id, b.slot_for_id), f"slot_for_id {msg}"
    assert torch.equal(a.id_of_slot, b.id_of_slot), f"id_of_slot {msg}"
    assert torch.equal(a.usage, b.usage), f"usage {msg}"
    assert torch.equal(a.num_indices, b.num_indices), f"num_indices {msg}"


# ---------------------------------------------------------------- registration


def test_policy_registration_and_defaults():
    cache = _make_cache()
    assert cache.cache_policy_id == 1
    assert cache.pin_freq.shape == (2, 8)
    assert int(cache.pin_cfg[0]) == 16  # pin_base = 2 * num_experts
    assert int(cache.pin_cfg[1]) == 0  # cold start: pin region inactive
    assert cache.pin_slots_target == 16
    # lru stays inert: no pin state at all
    lru = _make_cache(policy="lru")
    assert lru.cache_policy_id == 0
    assert lru.pin_freq is None and lru.pin_id_mask is None and lru.pin_cfg is None


def test_pin_slots_clamped_to_persistent_region():
    # persistent region is [2E, C) = 16 slots; a larger K must clamp at activation
    cache = _make_cache(pin_slots=10_000)
    pin_base, k, quota = cache._pin_geometry()
    assert (pin_base, k, quota) == (16, 16, 8)


def test_freq_pin_rejects_hybrid_decode_target():
    with pytest.raises(ValueError, match="hybrid"):
        _make_cache(decode_target="hybrid")


def test_freq_pin_rejects_cpu_decode_target():
    # construction-time defense: without it the combination was silently inert
    with pytest.raises(ValueError, match="decode_target='gpu' only"):
        _make_cache(decode_target="cpu")


def test_env_default_pin_slots(monkeypatch):
    monkeypatch.setenv("FREETOKEN_PIN_SLOTS", "6144")
    cache = _make_cache(pin_slots=None, C=64, E=8)
    assert cache.pin_slots_target == 6144  # target kept; activation clamps to geometry
    assert cache._pin_geometry()[1] == 64 - 16


# ------------------------------------------------------- cold start == LRU (C4)


@CUDA
def test_cold_start_bit_identical_to_flashlib_lru():
    L, E, C = 2, 32, 40
    steps = _random_steps(L, E, 64, 8, seed=0)
    lru = _make_cache(policy="lru", L=L, E=E, C=C, device="cuda")
    pin = _make_cache(policy="freq_pin", L=L, E=E, C=C, device="cuda", pin_slots=0)
    lru.collect_stats = pin.collect_stats = True
    for layer, ids in steps:
        g_ids, c_ids = ids.cuda(), ids.cuda()
        lru.ensure_experts(layer, g_ids)
        pin.ensure_experts(layer, c_ids)
        assert torch.equal(g_ids, c_ids)
        _assert_state_equal(lru, pin, msg=f"layer {layer}")
    assert torch.equal(lru.lru_stats, pin.lru_stats)


def test_cold_start_matches_route_trace_lru_mirror():
    """CPU mirror with an inactive pin region vs the stdlib LRU mirror (C4 lock).

    The lock is on the admission DECISIONS (miss/active counts and the resident
    set, which for LRU are determined by the reference string alone), not the
    free-slot fill order -- the heap mirror pops the free list from the tail."""
    L, E, C = 2, 16, 24
    cache = _make_cache(L=L, E=E, C=C, pin_slots=0)
    cache.collect_stats = True
    mirror = _rt.LRU(C, E)
    for layer, ids in _random_steps(L, E, 64, 6, seed=1):
        cache.ensure_experts(layer, ids.clone())
        mirror.ensure(layer, ids.tolist())
    assert int(cache.lru_stats[:, 1].sum()) == mirror.miss
    assert int(cache.lru_stats[:, 0].sum()) == mirror.active
    residents = {
        layer * E + e
        for layer in range(L)
        for e in range(E)
        if int(cache.slot_for_id[layer, e]) != -1
    }
    assert residents == set(mirror.slot_of)


# ------------------------------------------- kernel == CPU mirror == trace mirror


def _assert_mirror_match(cache, mirror, L, E, C):
    slot_of = cache.slot_for_id
    for layer in range(L):
        for e in range(E):
            fid = layer * E + e
            assert (int(slot_of[layer, e]) != -1) == (fid in mirror.slot_of)
            if fid in mirror.slot_of:
                assert int(slot_of[layer, e]) == mirror.slot_of[fid]
    for s in range(C):
        expected = mirror.owner[s] if mirror.owner[s] is not None else -1
        assert int(cache.id_of_slot[s]) == expected, f"slot {s}"
        assert int(cache.usage[s]) == mirror.usage[s], f"usage slot {s}"


def test_cpu_mirror_matches_freqpin_trace_mirror():
    L, E, C, K = 2, 16, 48, 12  # borrow [0,32), pin [32,44), lru remainder [44,48)
    cache = _make_cache(L=L, E=E, C=C, pin_slots=K)
    mirror = _rt.FreqPinLRU(C, E, pin_base=2 * E, pin_slots=K)
    pinned = {0, 1, 2, 3, E + 4, E + 5}  # layer0 experts 0-3, layer1 experts 4-5
    _activate_pins(cache, pinned, K)
    mirror.set_pins(pinned)
    for layer, ids in _random_steps(L, E, 128, 8, seed=2):
        cache.ensure_experts(layer, ids.clone())
        mirror.ensure(layer, ids.tolist())
    _assert_mirror_match(cache, mirror, L, E, C)


@CUDA
def test_gpu_kernel_matches_cpu_reference():
    L, E, C, K = 2, 32, 96, 16  # borrow [0,64), pin [64,80), lru remainder [80,96)
    gpu = _make_cache(L=L, E=E, C=C, device="cuda", pin_slots=K)
    ref = _make_cache(L=L, E=E, C=C, device="cuda", pin_slots=K)
    gpu.collect_stats = ref.collect_stats = True
    pinned = set(range(0, 6)) | set(range(E, E + 6))
    for cache in (gpu, ref):
        _activate_pins(cache, pinned, K)
    for layer, ids in _random_steps(L, E, 64, 10, seed=3):
        g, c = ids.cuda(), ids.clone()  # a CPU ids tensor drives the reference path
        gpu.ensure_experts(layer, g)
        ref.ensure_experts(layer, c)
        assert torch.equal(g.cpu(), c), f"rewrite mismatch layer {layer}"
        assert torch.equal(gpu.slot_for_id.cpu(), ref.slot_for_id.cpu())
        assert torch.equal(gpu.id_of_slot.cpu(), ref.id_of_slot.cpu())
        assert torch.equal(gpu.usage.cpu(), ref.usage.cpu())
        n = int(gpu.num_indices.item())
        assert n == int(ref.num_indices.item())
        assert torch.equal(gpu.evict_slots[:n].cpu(), ref.evict_slots[:n].cpu())
        assert torch.equal(gpu.src_indices[:n].cpu(), ref.src_indices[:n].cpu())
    assert torch.equal(gpu.lru_stats.cpu(), ref.lru_stats.cpu())


# ------------------------------------------------------- eviction correctness


def test_pinned_slots_survive_lru_pressure():
    """Non-pinned misses recycle only the non-pin pool; pinned ids stay resident."""
    L, E, C, K = 4, 8, 32, 8  # borrow [0,16), pin [16,24), remainder [24,32)
    cache = _make_cache(L=L, E=E, C=C, pin_slots=K)
    pinned = {0, 1}  # layer 0, experts 0-1
    _activate_pins(cache, pinned, K)
    cache.ensure_experts(0, torch.tensor([0, 1], dtype=torch.int32))
    slots = {int(cache.slot_for_id[0, 0]), int(cache.slot_for_id[0, 1])}
    assert all(16 <= s < 24 for s in slots)
    g = torch.Generator().manual_seed(4)
    for _ in range(200):
        for layer in (1, 2, 3):
            ids = torch.randperm(E, generator=g)[:6].to(torch.int32)
            cache.ensure_experts(layer, ids)
            for s in range(16, 24):
                fid = int(cache.id_of_slot[s])
                assert fid == -1 or fid in pinned, f"non-pinned id {fid} in pin slot {s}"
    assert {int(cache.slot_for_id[0, 0]), int(cache.slot_for_id[0, 1])} == slots


def test_nonpinned_miss_never_enters_pin_region():
    # Kernel-level branch check: a reserved-but-empty pin region (not a state
    # maybe_repin produces -- it always swaps mask and region together) must
    # still keep non-pinned misses out of the pin slots.
    L, E, C, K = 1, 8, 32, 8
    cache = _make_cache(L=L, E=E, C=C, pin_slots=K)
    _activate_pins(cache, set(), K)
    for _ in range(50):
        cache.ensure_experts(0, torch.arange(8, dtype=torch.int32))
    assert int(cache.id_of_slot[16:24].ge(0).sum()) == 0
    assert int(cache.id_of_slot.ge(0).sum()) > 0


def test_pinned_occupant_evicted_only_as_last_resort():
    """A pinned miss prefers a NON-pinned occupant of the pin region; a pinned
    occupant is evicted only when the region holds nothing else -- and a slot
    touched by the same call is never the victim (batch protection, C1)."""
    L, E, C, K = 1, 8, 24, 4  # pin region [16, 20)
    cache = _make_cache(L=L, E=E, C=C, pin_slots=K)
    _activate_pins(cache, {0, 1, 2, 3}, K)
    cache.ensure_experts(0, torch.tensor([0, 1, 2, 3], dtype=torch.int32))
    slots = {e: int(cache.slot_for_id[0, e]) for e in range(4)}
    assert all(16 <= s < 20 for s in slots.values())
    # give expert 3 a fresher timestamp
    cache.ensure_experts(0, torch.tensor([3], dtype=torch.int32))
    # swap the pin set: 0 leaves, 4 joins; 3 is hit in the same call as the miss
    _activate_pins(cache, {1, 2, 4, 3}, K)
    cache.ensure_experts(0, torch.tensor([3, 4], dtype=torch.int32))
    # 4 takes the slot of the departed (now non-pinned) 0; 3's hit slot is protected
    assert int(cache.slot_for_id[0, 4]) == slots[0]
    assert int(cache.slot_for_id[0, 3]) == slots[3]
    assert int(cache.slot_for_id[0, 0]) == -1
    slots[4] = slots[0]
    # decisive batch protection: 3 (non-pinned now, but hit this call) must not be
    # evicted even though every other pin slot holds a pinned occupant
    _activate_pins(cache, {0, 1, 2, 4}, K)
    cache.ensure_experts(0, torch.tensor([3, 0], dtype=torch.int32))
    assert int(cache.slot_for_id[0, 3]) == slots[3]
    assert int(cache.slot_for_id[0, 0]) in (slots[1], slots[2], slots[4])
    evicted = [e for e in (1, 2, 4) if int(cache.slot_for_id[0, e]) == -1]
    assert len(evicted) == 1


def test_borrow_region_invalidation_leaves_pins_untouched():
    """The per-chunk prefill buffer invalidation (slots [0, 2E), usage=0 coldest)
    must not disturb pinned residents (C2)."""
    L, E, C, K = 2, 8, 32, 8
    cache = _make_cache(L=L, E=E, C=C, pin_slots=K, prefill_overlap=True)
    _with_banks(cache)
    _activate_pins(cache, {0, 1, E + 2}, K)
    cache.ensure_experts(0, torch.tensor([0, 1, 3], dtype=torch.int32))
    cache.ensure_experts(1, torch.tensor([2, 5], dtype=torch.int32))
    before = cache.slot_for_id.clone()
    for buf in (0, 1):
        cache._invalidate_prefill_buffer(buf)
    # pinned ids keep their pin-region slots; borrow-region residents are dropped
    for layer, e in ((0, 0), (0, 1), (1, 2)):
        s = int(cache.slot_for_id[layer, e])
        assert 16 <= s < 24 and s == int(before[layer, e])
    assert int(cache.slot_for_id[0, 3]) == -1
    assert int(cache.slot_for_id[1, 5]) == -1
    assert torch.all(cache.id_of_slot[:16] == -1)
    assert torch.all(cache.usage[:16] == 0)


# ------------------------------------------------------------------ repinning


def _zipf_freq(L, E, alpha, scale=1_000_000, seed=0):
    g = torch.Generator().manual_seed(seed)
    ranks = torch.arange(1, E + 1, dtype=torch.float64)
    base = (ranks ** -alpha) * scale
    freq = torch.zeros(L, E)
    for layer in range(L):
        perm = torch.randperm(E, generator=g)
        freq[layer] = base[perm]
    return freq


def test_repin_warmup_gate_and_first_pin_placement():
    L, E, C, K = 2, 8, 32, 16  # quota 8 == E: everything pinned
    cache = _make_cache(L=L, E=E, C=C, pin_slots=K)
    sources = _with_banks(cache)
    cache.pin_warmup_count = 100
    # below the warmup threshold: pure LRU, no pin activation
    cache._last_warmup_check = -64  # let this maybe_repin check immediately
    ids = torch.tensor([0, 1, 2], dtype=torch.int32)
    cache.ensure_experts(0, ids.clone())
    assert cache.maybe_repin() is None
    assert int(cache.pin_cfg[1]) == 0
    # past the threshold: first pin activates the region and force-places
    freq = _zipf_freq(L, E, 1.31, scale=400, seed=7)
    cache.pin_freq.copy_(freq.long())
    cache._last_warmup_check = -64
    report = cache.maybe_repin()
    assert report is not None and report["first"] is True
    assert report["k"] == K and report["quota"] == 8
    assert int(cache.pin_cfg[1]) == K
    # every expert pinned (quota == E) and resident in the pin region
    for layer in range(L):
        for e in range(E):
            s = int(cache.slot_for_id[layer, e])
            assert 16 <= s < 32, f"layer {layer} expert {e} in slot {s}"
    # bytes actually moved: the slots hold the source rows
    for layer in range(L):
        for e in range(E):
            s = int(cache.slot_for_id[layer, e])
            assert torch.equal(cache.bank_caches["gate_up"][s], sources["gate_up"][layer][e])
            assert torch.equal(cache.bank_caches["down"][s], sources["down"][layer][e])
    # histogram decayed by the EMA halving after the repin
    assert torch.equal(cache.pin_freq, freq.long().bitwise_right_shift(1))


def test_repin_place_max_rows_throttles_first_pin():
    """FREETOKEN_PIN_PLACE_MAX_ROWS caps force-placed rows per epoch: the first-pin
    pulse spreads over consecutive epochs (same deferral semantics as in-flight
    slots); the final pinned set is identical."""
    L, E, C, K = 1, 8, 32, 8  # pin region [16, 24), quota 8 == E
    cache = _make_cache(L=L, E=E, C=C, pin_slots=K)
    sources = _with_banks(cache)
    cache.pin_warmup_count = 1
    cache.pin_repin_interval = 1
    cache._pin_decay = False
    cache.pin_place_max_rows = 3
    cache._last_warmup_check = -64
    cache.pin_freq.fill_(100)  # every expert pinned, tie order = ascending id
    r1 = cache.maybe_repin()
    assert r1["first"] and r1["loaded_rows"] == 3 and r1["deferred_rows"] == 5
    # the first 3 (ascending-id order) are already in the region with real bytes
    for e in range(3):
        s = int(cache.slot_for_id[0, e])
        assert 16 <= s < 24
        assert torch.equal(cache.bank_caches["gate_up"][s], sources["gate_up"][0][e])
    r2 = cache.maybe_repin()
    assert r2["loaded_rows"] == 3 and r2["deferred_rows"] == 2
    r3 = cache.maybe_repin()
    assert r3["loaded_rows"] == 2 and r3["deferred_rows"] == 0
    # final state equals the uncapped first pin: all 8 pinned, all in region
    for e in range(E):
        s = int(cache.slot_for_id[0, e])
        assert 16 <= s < 24
        assert torch.equal(cache.bank_caches["gate_up"][s], sources["gate_up"][0][e])
    assert int(cache.pin_id_mask.sum()) == E
    # default 0 = unlimited: a fresh cache places everything in one epoch
    full = _make_cache(L=L, E=E, C=C, pin_slots=K)
    _with_banks(full)
    full.pin_warmup_count = 1
    full.pin_freq.fill_(100)
    full._last_warmup_check = -64
    r = full.maybe_repin()
    assert r["loaded_rows"] == 8 and r["deferred_rows"] == 0


def test_repin_interval_gate():
    cache = _make_cache()
    _with_banks(cache)
    cache.pin_warmup_count = 1
    cache.pin_repin_interval = 10
    cache.pin_freq.fill_(1000)
    cache._last_warmup_check = -64
    assert cache.maybe_repin() is not None  # first pin
    for _ in range(9):
        assert cache.maybe_repin() is None  # interval not reached
    assert cache.maybe_repin() is not None  # 10 decode steps later: second repin


def test_repin_hysteresis_bounds_delta():
    """Rank noise in the histogram tail must not churn the pin set: with the 0.5
    hysteresis margin the repin delta stays far below the naive top-K re-selection
    and under the 2 GiB/repin budget at production geometry (sim report section 8),
    while a genuine distribution shift still rotates the set."""
    L, E, C = 48, 256, 8800
    K = 7168
    cache = _make_cache(L=L, E=E, C=C, pin_slots=K)
    cache._copy_feat_bytes_host = [2_772_480]  # production unit bytes (BENCH section 1)
    freq = _zipf_freq(L, E, 1.31, scale=1_000_000, seed=11)
    cache.pin_freq.copy_(freq.long())
    first = cache._compute_pin_plan()
    quota = K // L
    assert first["changed_rows"] == quota * L  # full first pin
    cache.pin_id_mask.copy_(first["new_mask"].reshape(-1))
    cache._pinned_once = True

    def naive_change(new_freq):
        total = 0
        for layer in range(L):
            old_set = set(first["new_mask"][layer].nonzero(as_tuple=False).flatten().tolist())
            top = set(torch.argsort(new_freq[layer], descending=True)[:quota].tolist())
            total += len(top - old_set)
        return total

    # pure tail noise (+/-20%): hysteresis must nearly freeze the set
    g = torch.Generator().manual_seed(12)
    noise = torch.empty(L, E).uniform_(0.8, 1.25, generator=g)
    noisy = (freq * noise).long()
    naive = naive_change(noisy)
    cache.pin_freq.copy_(noisy)
    plan = cache._compute_pin_plan()
    gib = plan["changed_rows"] * 2_772_480 / 2**30
    print(
        f"\n[noise] hysteresis delta {plan['changed_rows']} rows ({gib:.2f} GiB) "
        f"vs naive {naive} rows ({naive * 2_772_480 / 2**30:.2f} GiB)"
    )
    assert plan["changed_rows"] <= naive // 3
    assert plan["changed_rows"] * 2_772_480 <= 2 * 2**30

    # genuine shift (5% of experts boosted 5x): the set must still rotate
    boost = (torch.rand(L, E, generator=g) < 0.05).float() * 4.0 + 1.0
    shifted = (freq * boost).long()
    cache.pin_freq.copy_(shifted)
    plan = cache._compute_pin_plan()
    naive_shift = naive_change(shifted)
    print(
        f"\n[shift] hysteresis delta {plan['changed_rows']} rows "
        f"({plan['changed_rows'] * 2_772_480 / 2**30:.2f} GiB) vs naive {naive_shift}"
    )
    assert plan["changed_rows"] >= 0.5 * naive_shift


def test_repin_defers_in_flight_slots():
    """Slots touched by the very latest ensure call are in-flight: their force
    placement defers to the next epoch instead of evicting under a running GEMM."""
    L, E, C, K = 1, 8, 32, 4  # pin region [16, 20), quota 4
    cache = _make_cache(L=L, E=E, C=C, pin_slots=K)
    _with_banks(cache)
    cache.pin_warmup_count = 1
    cache.pin_repin_interval = 1
    cache._pin_decay = False
    cache._last_warmup_check = -64
    cache.ensure_experts(0, torch.tensor([0, 1, 2, 3], dtype=torch.int32))  # step 1
    cache.pin_freq[0, :4] = 100  # pin {0,1,2,3}
    assert cache.maybe_repin()["first"]  # force-places {0..3} into [16, 20)
    pinned_slots = {e: int(cache.slot_for_id[0, e]) for e in range(4)}
    assert all(16 <= s < 20 for s in pinned_slots.values())
    # the next decode step hits the pinned experts: their slots are genuinely
    # in-flight at the fence (usage == step of the latest ensure call)
    cache.ensure_experts(0, torch.tensor([0, 1, 2, 3], dtype=torch.int32))  # step 2
    # rotate the distribution: now {4,5,6,7} are hot, {0..3} cold
    cache.pin_freq.zero_()
    cache.pin_freq[0, 4:] = 100
    step_now = int(cache.step.item())
    report = cache.maybe_repin()
    assert report is not None and report["first"] is False
    assert report["deferred_rows"] == 4 and report["loaded_rows"] == 0
    # the old pinned residents keep their slots this epoch
    for e, s in pinned_slots.items():
        assert int(cache.slot_for_id[0, e]) == s
    # ...but the bitmap swap is atomic: the mask already names the new set
    assert cache.pin_id_mask.view(L, E)[0].tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    # next epoch (the step moved on): the deferred placements complete
    cache.step.fill_(step_now + 1)
    report = cache.maybe_repin()
    assert report["loaded_rows"] == 4 and report["deferred_rows"] == 0
    for e in range(4, 8):
        assert 16 <= int(cache.slot_for_id[0, e]) < 20


def test_rebuild_resets_pin_state():
    L, E, C, K = 2, 8, 32, 16
    cache = _make_cache(L=L, E=E, C=C, pin_slots=K)
    _with_banks(cache)
    cache.pin_warmup_count = 1
    cache.pin_freq.fill_(10)
    cache._last_warmup_check = -64
    assert cache.maybe_repin() is not None
    assert int(cache.pin_cfg[1]) == K
    cache.rebuild(C)
    assert int(cache.pin_cfg[1]) == 0
    assert cache._pinned_once is False
    assert int(cache.pin_freq.sum()) == 0
    assert int(cache.pin_id_mask.sum()) == 0
    # cold again: pure-LRU behaviour until the warmup threshold is re-met
    assert cache.maybe_repin() is None


def test_stats_snapshot_pin_block():
    cache = _make_cache()
    _with_banks(cache)
    cache.pin_warmup_count = 1
    cache.pin_freq.fill_(10)
    cache._last_warmup_check = -64
    cache.maybe_repin()
    snap = cache.stats_snapshot()
    assert snap["pin"]["pin_slots"] == 16
    assert snap["pin"]["warmed"] is True
    assert snap["pin"]["repins"] == 1
    assert snap["pin"]["last_repin"]["first"] is True
    lru = _make_cache(policy="lru")
    assert "pin" not in lru.stats_snapshot()


def test_stats_snapshot_raw_decode_freq_export():
    """Spec section 6 anchor: the raw [L, E] decode_freq histogram rides /v1/stats
    for offline pin-set / hit-curve observability."""
    cache = _make_cache()
    cache.collect_decode_freq = True
    cache.ensure_experts(0, torch.tensor([0, 1, 1, 3], dtype=torch.int32))
    cache.ensure_experts(1, torch.tensor([2], dtype=torch.int32))
    snap = cache.stats_snapshot()
    raw = snap["raw"]["decode_freq"]
    assert len(raw) == 2 and len(raw[0]) == 8
    assert raw[0][0] == 1 and raw[0][1] == 2 and raw[0][3] == 1
    assert raw[1][2] == 1
    # no collection flag: no raw block at all (98KB stays out of the reply)
    assert "raw" not in _make_cache().stats_snapshot()


# ------------------------------------------------------- owner graph counting (C5)


def test_owner_graph_pin_freq_counts_owned_only():
    """Under owner EP the graph path admits remote entries as borrows of the first
    owned row; the freq_pin histogram must count owned positions only (C5), while
    the legacy decode_freq keeps its existing (borrow-inflated) behaviour."""
    from freetoken.moe.offload_cache import OwnerOffloadMoeCache
    from freetoken.moe.ownership import OwnerCacheGeometry

    geometry = OwnerCacheGeometry(16, 2, 0, num_layers=1, cache_size=16)
    cache = OwnerOffloadMoeCache(
        geometry, torch.device("cpu"), cache_policy="freq_pin", pin_slots=4
    )
    inner = cache._cache
    inner.collect_decode_freq = True
    # route: two owned (0 -> local 0, 3 -> local 3), two remote (8, 9 borrow row 0)
    ids = torch.tensor([[0, 8, 3, 9]], dtype=torch.int32)
    weights = torch.full((1, 4), 0.25)
    cache.ensure_route_graph(0, weights, ids)
    pin_freq = inner.pin_freq[0].tolist()
    assert pin_freq[0] == 1 and pin_freq[3] == 1
    assert sum(pin_freq) == 2, pin_freq
    # decode_freq is untouched by the fix: it still counts the borrowed rows
    # (4 admissions, 3 of them row 0) exactly as before.
    decode_freq = inner.decode_freq[0].tolist()
    assert decode_freq[0] == 3 and decode_freq[3] == 1


# ---------------------------------------------------- t4 stress geometry locks
# Folded in from research/bench/kernel_freqpin_stress_20260918.py (K1/K2): global
# ids exceed capacity, so every geometry forces evictions; large batches create
# same-step ties. These supplement the short cold-start locks above.


@CUDA
@pytest.mark.parametrize(
    "L,E,C,steps,max_batch,seed",
    [
        (3, 48, 96, 10000, 20, 0),
        (3, 48, 96, 10000, 20, 1),
        (2, 16, 24, 20000, 12, 2),  # tiny geometry, dense ties
        (2, 32, 40, 20000, 8, 3),  # the t3 geometry at 300x length
    ],
)
def test_cold_start_high_pressure_bit_identical(L, E, C, steps, max_batch, seed):
    """freq_pin(K=0) kernel vs flashlib lru under forced-eviction pressure (K1)."""
    g = torch.Generator().manual_seed(seed)
    lru = _make_cache(policy="lru", L=L, E=E, C=C, device="cuda")
    pin = _make_cache(policy="freq_pin", L=L, E=E, C=C, device="cuda", pin_slots=0)
    lru.collect_stats = pin.collect_stats = True
    for i in range(steps):
        layer = int(torch.randint(L, (1,), generator=g))
        k = int(torch.randint(1, max_batch + 1, (1,), generator=g))
        ids = torch.randperm(E, generator=g)[:k].to(torch.int32)
        lru.ensure_experts(layer, ids.clone().cuda())
        pin.ensure_experts(layer, ids.clone().cuda())
        if i % 50 == 0 or i == steps - 1:
            _assert_state_equal(lru, pin, msg=f"geom {(L, E, C)} step {i}")
    assert torch.equal(lru.lru_stats, pin.lru_stats)


@pytest.mark.parametrize(
    "L,E,C,steps,max_batch,seed",
    [(3, 48, 96, 5000, 20, 10), (2, 16, 24, 10000, 12, 11)],
)
def test_cold_start_high_pressure_matches_trace_mirror(L, E, C, steps, max_batch, seed):
    """CPU reference mirror vs stdlib FreqPinLRU under the same pressure (K2)."""
    g = torch.Generator().manual_seed(seed)
    cache = _make_cache(L=L, E=E, C=C, pin_slots=0)
    mirror = _rt.FreqPinLRU(C, E)
    for i in range(steps):
        layer = int(torch.randint(L, (1,), generator=g))
        k = int(torch.randint(1, max_batch + 1, (1,), generator=g))
        ids = torch.randperm(E, generator=g)[:k].to(torch.int32)
        cache.ensure_experts(layer, ids.clone())
        mirror.ensure(layer, ids.tolist())
        if i % 50 == 0 or i == steps - 1:
            _assert_mirror_match(cache, mirror, L, E, C)
