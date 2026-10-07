"""Per-expert route/miss counters (review G1, port/m4-perexpert-counts) -- pure-CPU tests.

The split kernel accumulates the [L,E] counters on the device during graph
replay; production counting is GPU-side, so these tests drive the Python
mirror (``route_count_mirror`` -- the same accounting as scalar pass 1) and
the host-side delta/stats/wiring paths with CPU tensors. The kernel-vs-mirror
cross-check on real GPU traffic is a GPU-battery item.

Covered:
  * mirror totals == per-layer aggregates (route.sum() == nh + n_miss ==
    entries, miss.sum() == n_miss), exact per-expert values, determinism
  * take_route_count_deltas: windowed deltas, zero between takes
  * stats() passthrough: per-expert totals consistent with the [L,16] aggregates
  * admission self-sourcing: real per-expert counts drive a swap (G1: the
    policy no longer idles on miss_counts=None), free no-op between
    evaluations, zero-miss fallback without a tier, explicit counts bypass
  * honesty bit: guard/admission summaries carry wired:true since the GPU
    battery (review item #12) landed the engine hooks; the flag still
    tracks the module constant
"""

from __future__ import annotations

import torch

from freetoken.moe import cache_admission as ca
from freetoken.moe.cpu_tier import CpuTier, route_count_mirror
from freetoken.moe.offload_cache import OffloadMoeCache

L, E, SLOTS, DEPTH = 2, 4, 16, 2
BORROWED = DEPTH * E


# ---------------------------------------------------------------------------
# route_count_mirror: kernel pass-1 accounting on pure CPU
# ---------------------------------------------------------------------------

def _mirror_world():
    """One staged plan with hits, GPU-bound misses, CPU-ok misses, duplicates."""
    E8, layer, cache_size, ram_rows = 8, 1, 16, 4
    row_map = [0, -1, 1, -1, 2, 3, -1, -1]      # cpu rows for experts 0,2,4,5
    src_indices = [3, 1, 5]                     # staged miss experts
    evict_slots = [10, 11, 12]                  # their victim slots
    id_of_slot = [-1] * cache_size
    id_of_slot[3] = layer * E8 + 2              # resident (layer 1, expert 2)
    id_of_slot[4] = layer * E8 + 6              # resident (layer 1, expert 6)
    id_of_slot[5] = layer * E8 + 2              # expert 2 routed twice
    slots = [3, 4, 5, 10, 12, 12]               # 6 routed entries this layer
    return dict(slots=slots, evict_slots=evict_slots, src_indices=src_indices,
                row_map=row_map, ram_rows=ram_rows, id_of_slot=id_of_slot,
                cache_size=cache_size, layer_id=layer, num_experts=E8)


def test_mirror_exact_per_expert_values():
    route, miss, nh, n_miss = route_count_mirror(**_mirror_world())
    assert route.tolist() == [0, 0, 2, 1, 0, 2, 1, 0]  # e2 hit twice; e3,e5 misses
    assert miss.tolist() == [0, 0, 0, 1, 0, 2, 0, 0]
    assert nh == 3 and n_miss == 3


def test_mirror_totals_match_per_layer_aggregates():
    w = _mirror_world()
    route, miss, nh, n_miss = route_count_mirror(**w)
    # the invariants the kernel keeps by construction: per-expert sums equal
    # the per-layer [L,16] aggregate increments (sb+0 hits, sb+1 misses)
    assert int(route.sum()) == nh + n_miss == len(w["slots"])
    assert int(miss.sum()) == n_miss


def test_mirror_deterministic_three_runs():
    results = [route_count_mirror(**_mirror_world()) for _ in range(3)]
    for route, miss, nh, n_miss in results[1:]:
        assert torch.equal(route, results[0][0])
        assert torch.equal(miss, results[0][1])
        assert (nh, n_miss) == (results[0][2], results[0][3])


# ---------------------------------------------------------------------------
# CpuTier counter plumbing (bare instance: attach() needs CUDA pinned allocs)
# ---------------------------------------------------------------------------

def _bare_tier() -> CpuTier:
    """CpuTier with the counter buffers attach() would build, on CPU tensors."""
    t = CpuTier.__new__(CpuTier)
    t._built = True
    t._service = None
    t._num_layers = L
    t._num_experts = 8
    t._stats_dev = torch.zeros(L * 16, dtype=torch.int64)
    t._dflag = torch.zeros(8, dtype=torch.int64)
    t._route_counts_dev = torch.zeros(L * 8, dtype=torch.int64)
    t._miss_counts_dev = torch.zeros(L * 8, dtype=torch.int64)
    t._counter_snapshot = (torch.zeros(L, 8, dtype=torch.int64),
                           torch.zeros(L, 8, dtype=torch.int64))
    t._cost = {"tzc": 0.58, "thit": 0.03, "a": 0.11, "b": 0.20,
               "tok": 0.35, "maxn": 384.0}
    return t


def test_take_route_count_deltas_windows():
    t = _bare_tier()
    t._route_counts_dev.view(L, 8)[0, 3] = 10
    t._miss_counts_dev.view(L, 8)[0, 3] = 4
    dr, dm = t.take_route_count_deltas()
    assert dr.shape == (L, 8) and dr.dtype == torch.int64
    assert int(dr[0, 3]) == 10 and int(dm[0, 3]) == 4
    assert int(dr.sum()) == 10 and int(dm.sum()) == 4
    # nothing new between takes: the window is empty (no double-counting)
    dr2, dm2 = t.take_route_count_deltas()
    assert int(dr2.sum()) == 0 and int(dm2.sum()) == 0
    # later increments show up as exactly the increment
    t._route_counts_dev.view(L, 8)[1, 5] += 3
    dr3, _ = t.take_route_count_deltas()
    assert int(dr3.sum()) == 3 and int(dr3[1, 5]) == 3


def test_stats_passthrough_per_expert_totals():
    t = _bare_tier()
    dev = t._stats_dev.view(-1, 16)
    dev[0, 0], dev[0, 1] = 7, 3   # layer 0: 7 hits, 3 misses
    dev[1, 0], dev[1, 1] = 2, 1   # layer 1: 2 hits, 1 miss
    t._route_counts_dev.view(L, 8)[0, 2] = 10   # == 7 + 3
    t._route_counts_dev.view(L, 8)[1, 4] = 3    # == 2 + 1
    t._miss_counts_dev.view(L, 8)[0, 2] = 3
    t._miss_counts_dev.view(L, 8)[1, 4] = 1
    s = t.stats()
    assert s["per_expert_counts"] is True
    # per-expert totals are consistent with the per-layer aggregates
    assert s["routed_entries"] == 13 == 7 + 3 + 2 + 1
    assert s["miss_entries"] == 4 == 3 + 1
    assert s["per_layer"] == [[7, 3, 0, 0], [2, 1, 0, 0]]


# ---------------------------------------------------------------------------
# admission self-sourcing (G1: real per-expert counts end the idle spin)
# ---------------------------------------------------------------------------

class _StubTier:
    """CpuTier stand-in serving canned [L,E] deltas; no CUDA anywhere."""

    enabled = True

    def __init__(self, route: torch.Tensor, miss: torch.Tensor) -> None:
        self._route, self._miss = route, miss
        self.calls = 0

    def take_route_count_deltas(self):
        self.calls += 1
        return self._route.clone(), self._miss.clone()


def _admission_cache(monkeypatch, period: int = 1) -> OffloadMoeCache:
    monkeypatch.setattr(ca, "ONLINE_ADMISSION", True)
    cache = OffloadMoeCache(num_layers=L, num_experts=E, cache_size=SLOTS,
                            device=torch.device("cpu"))
    adm = cache.admission
    assert adm is not None
    adm.config.period = period
    adm.set_pin_rows([[0, 1, 2], [0, 1]])
    return cache


def _hot_counts():
    route = torch.zeros(L, E, dtype=torch.int64)
    miss = torch.zeros(L, E, dtype=torch.int64)
    route[0, 0], miss[0, 0] = 100, 50   # pinned: miss rate 0.5 > 0.10 -> out
    route[0, 3], miss[0, 3] = 100, 95   # challenger: 95 > 1.5 * 50 -> in
    return route, miss


def test_admission_self_sources_real_counts_and_swaps(monkeypatch):
    cache = _admission_cache(monkeypatch, period=1)
    route, miss = _hot_counts()
    tier = _StubTier(route, miss)
    cache._cpu_tier = tier
    cache.note_decode_step()          # arm the evaluation period
    swaps = cache.admission_update()  # self-sources (route, miss) from the tier
    assert swaps == [((0, 0), (0, 3))]
    assert tier.calls == 1
    snap = cache.stats_snapshot()
    assert snap["admission"]["swaps"] == 1
    assert snap["admission"]["wired"] is True  # battery #12 wired the hooks
    # with miss_counts=None (the pre-G1 wiring) the misses decayed to zero and
    # this swap could never fire -- the policy idled. Real per-expert misses
    # from the split kernel end that.


def test_admission_update_between_evaluations_never_reads_the_tier(monkeypatch):
    cache = _admission_cache(monkeypatch, period=2)
    tier = _StubTier(*_hot_counts())
    cache._cpu_tier = tier
    assert cache.admission_update() == [] and tier.calls == 0
    cache.note_decode_step()  # since_eval = 1 < period: still a free no-op
    assert cache.admission_update() == [] and tier.calls == 0
    cache.note_decode_step()  # since_eval = 2 >= period: due -> one device read
    assert cache.admission_update() == [((0, 0), (0, 3))]
    assert tier.calls == 1


def test_admission_update_without_tier_is_conservative_zero_miss(monkeypatch):
    cache = _admission_cache(monkeypatch, period=1)
    cache.note_decode_step()
    assert cache.admission_update() == []  # zero counts in; layout stays put
    assert cache.admission.evaluations == 1


def test_admission_update_explicit_counts_bypass_the_tier(monkeypatch):
    cache = _admission_cache(monkeypatch, period=1)
    tier = _StubTier(*_hot_counts())
    cache._cpu_tier = tier
    route, miss = _hot_counts()
    cache.note_decode_step()
    assert cache.admission_update(route, miss) == [((0, 0), (0, 3))]
    assert tier.calls == 0  # explicit counts never touch the device


def test_admission_env_off_update_no_counts_is_empty(monkeypatch):
    monkeypatch.setattr(ca, "ONLINE_ADMISSION", False)
    cache = OffloadMoeCache(num_layers=L, num_experts=E, cache_size=SLOTS,
                            device=torch.device("cpu"))
    assert cache.admission_update() == []


# ---------------------------------------------------------------------------
# honesty bit (m4 finding): enabled:true must not read as "working"
# ---------------------------------------------------------------------------

def test_summaries_carry_the_wired_honesty_bit(monkeypatch):
    g = ca.ElasticCacheGuard(L, E, SLOTS, BORROWED)
    a = ca.OnlineAdmission(L, E)
    # WIRED flipped True in the battery-#12 commit (engine hooks landed)
    assert g.summary()["wired"] is True
    assert a.summary()["wired"] is True
    # the summaries track the module constant either way
    monkeypatch.setattr(ca, "WIRED", False)
    assert g.summary()["wired"] is False
    assert a.summary()["wired"] is False
