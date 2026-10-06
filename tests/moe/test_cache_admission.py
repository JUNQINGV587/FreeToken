"""Item 5 (dsv41 port): elastic release/rewarm + online admission -- pure-CPU tests.

No CUDA anywhere: the policy objects are host-side bookkeeping over CPU tensors.
The CUDA apply (unmap + empty_cache + re-add) and the scheduler call sites defer
to the GPU battery.

Covered:
  * release scope: borrowed ring prefix and pin rows are never released
  * release gates: warmup, token threshold (strictly greater), calm countdown
  * side profile: 3 release -> rewarm rounds count rows and bytes
  * admission boundaries: miss threshold (strict), hysteresis (strict), batch
    cap, evaluation period, score decay, no-signal conservatism
  * swapped layouts stay in the pin_manager shape (pin_rows_to_row_map-valid)
  * env-off == pre-item-5 behavior (no guards, no stats keys)
"""

from __future__ import annotations

import torch

from freetoken.moe import cache_admission as ca
from freetoken.moe.disk_tier import pin_rows_to_row_map
from freetoken.moe.offload_cache import OffloadMoeCache

L, E, SLOTS, DEPTH = 2, 4, 16, 2
BORROWED = DEPTH * E  # prefill ring borrows the first depth*E slots (mainline)


def _guard(warmup: int = 4, calm: int = 2, threshold: int = 512) -> ca.ElasticCacheGuard:
    cfg = ca.ElasticConfig()
    cfg.prefill_tokens = threshold
    cfg.calm_steps = calm
    cfg.warmup_steps = warmup
    g = ca.ElasticCacheGuard(L, E, SLOTS, BORROWED, config=cfg)
    g.set_pin_rows([[0, 1], [3]])  # matches the residents in _slot_map()
    return g


def _slot_map() -> torch.Tensor:
    """Slot -> flat id: residents inside and outside the borrowed region."""
    ids = torch.full((SLOTS,), -1, dtype=torch.int32)
    ids[1] = 0   # borrowed region resident: never releasable
    ids[7] = 3   # borrowed region resident: never releasable
    ids[8] = 0   # pinned (layer 0, expert 0)
    ids[9] = 1   # pinned (layer 0, expert 1)
    ids[10] = 2  # releasable
    ids[12] = 7  # pinned (layer 1, expert 3 -> flat 7)
    ids[13] = 6  # releasable
    return ids


def _arm(g: ca.ElasticCacheGuard) -> None:
    for _ in range(g.config.warmup_steps):
        g.note_decode()


def test_release_scope_excludes_borrowed_and_pinned():
    g = _guard()
    _arm(g)
    slots = g.note_prefill(600, _slot_map())
    assert slots == [10, 13]
    assert g.keep == [2, 6]  # flat ids of the released slots
    assert g.released


def test_release_skipped_before_warmup():
    g = _guard(warmup=8)
    g.set_pin_rows([[0, 1], [3]])
    for _ in range(7):
        g.note_decode()
    assert g.note_prefill(600, _slot_map()) is None
    g.note_decode()  # 8th step: armed now
    assert g.note_prefill(600, _slot_map()) == [10, 13]


def test_release_threshold_is_strict():
    g = _guard(threshold=512)
    _arm(g)
    assert g.note_prefill(512, _slot_map()) is None  # == threshold: no release
    assert g.note_prefill(513, _slot_map()) == [10, 13]


def test_rewarm_after_calm_steps_and_prefill_resets_calm():
    g = _guard(calm=3)
    _arm(g)
    g.note_prefill(600, _slot_map())
    assert g.note_decode() is None
    g.note_prefill(10, _slot_map())  # any prefill resets the calm countdown
    assert g.note_decode() is None
    assert g.note_decode() is None
    assert g.note_decode() == [2, 6]  # third calm decode step: rewarm
    assert not g.released


def test_rewarm_counters_over_three_rounds():
    g = _guard(calm=1)
    g.slot_bytes = 100
    _arm(g)
    for _ in range(3):
        assert g.note_prefill(600, _slot_map()) == [10, 13]
        assert g.note_decode() == [2, 6]
    assert g.releases == 3
    assert g.rewarms == 3
    assert g.released_rows_total == 6
    assert g.rewarmed_rows_total == 6
    assert g.rewarmed_bytes_total == 600
    s = g.summary()
    assert s["releases"] == 3 and s["rewarmed_bytes"] == 600


def _admission(period: int = 2, max_swaps: int = 2, threshold: float = 0.10,
               hysteresis: float = 1.5, decay: float = 0.9,
               pin=((0, 1, 2), (0, 1)), arm: bool = True) -> ca.OnlineAdmission:
    cfg = ca.AdmissionConfig()
    cfg.period = period
    cfg.max_swaps = max_swaps
    cfg.miss_threshold = threshold
    cfg.hysteresis = hysteresis
    cfg.decay = decay
    a = ca.OnlineAdmission(L, 8, pin_rows=[list(r) for r in pin], config=cfg)
    if arm:
        a.note_step()  # one decode step so a period=1 update evaluates immediately
    return a


def _counts(pairs: dict[tuple[int, int], float]) -> torch.Tensor:
    out = torch.zeros(L, 8, dtype=torch.float64)
    for (layer, e), v in pairs.items():
        out[layer, e] = v
    return out


def test_admission_waits_for_the_period():
    a = _admission(period=3, arm=False)
    counts = _counts({(0, 0): 100, (0, 5): 100})
    miss = _counts({(0, 0): 20, (0, 5): 90})
    a.note_step()
    assert a.update(counts, miss) == []  # 1 < period: accumulate only
    a.note_step()
    assert a.update(counts, miss) == []  # 2 < period
    a.note_step()
    assert len(a.update(counts, miss)) == 1  # period reached: evaluate
    assert a.evaluations == 1


def test_admission_miss_threshold_is_strict():
    # pin row (0,0): 10 misses / 100 routes = 0.10 exactly -> NOT a candidate
    a = _admission(period=1)
    counts = _counts({(0, 0): 100, (0, 5): 100})
    miss = _counts({(0, 0): 10, (0, 5): 90})
    assert a.update(counts, miss) == []
    # one more miss tips the decayed rate over 0.10 -> swap with (0,5)
    a2 = _admission(period=1)
    miss2 = _counts({(0, 0): 11, (0, 5): 90})
    counts2 = _counts({(0, 0): 100, (0, 5): 100})
    assert a2.update(counts2, miss2) == [((0, 0), (0, 5))]


def test_admission_hysteresis_is_strict():
    # challenger misses == hysteresis * incumbent misses (30 == 1.5*20) -> no swap
    a = _admission(period=1, hysteresis=1.5)
    counts = _counts({(0, 0): 100, (0, 5): 100})
    miss = _counts({(0, 0): 20, (0, 5): 30})
    assert a.update(counts, miss) == []
    a2 = _admission(period=1, hysteresis=1.5)
    miss2 = _counts({(0, 0): 20, (0, 5): 31})
    assert a2.update(counts, miss2) == [((0, 0), (0, 5))]


def test_admission_batch_cap_picks_worst_first():
    # three pin rows over threshold, cap 2: the two worst miss rates swap out
    a = _admission(period=1, max_swaps=2)
    counts = _counts({(0, 0): 100, (0, 1): 100, (0, 2): 100,
                      (0, 5): 100, (0, 6): 100, (0, 7): 100})
    miss = _counts({(0, 0): 50, (0, 1): 30, (0, 2): 20,
                    (0, 5): 95, (0, 6): 90, (0, 7): 80})
    swaps = a.update(counts, miss)
    assert swaps == [((0, 0), (0, 5)), ((0, 1), (0, 6))]
    assert a.pin_rows[0] == [2, 5, 6]
    assert a.swaps == 2


def test_admission_no_miss_signal_never_swaps():
    a = _admission(period=1)
    counts = _counts({(0, 0): 100, (0, 5): 100})
    assert a.update(counts, None) == []
    assert a.pin_rows[0] == [0, 1, 2]


def test_admission_scores_decay():
    a = _admission(period=100, decay=0.9)
    a.note_step()
    a.update(_counts({(0, 0): 10}), _counts({(0, 0): 1}))
    a.note_step()
    a.update(_counts({}), _counts({}))  # empty cycle: decay only
    assert float(a._scores[0, 0]) == 9.0
    assert float(a._misses[0, 0]) == 0.9


def test_admission_layout_stays_pin_manager_valid():
    a = _admission(period=1, max_swaps=3)
    counts = _counts({(0, 0): 100, (0, 1): 100, (0, 2): 100,
                      (0, 5): 100, (0, 6): 100, (0, 7): 100})
    miss = _counts({(0, 0): 50, (0, 1): 40, (0, 2): 30,
                    (0, 5): 95, (0, 6): 90, (0, 7): 85})
    a.update(counts, miss)
    for rows in a.pin_rows:
        assert rows == sorted(set(rows))
    # the swapped layout must still feed the disk tier's row-map builder
    row_map = pin_rows_to_row_map(a.pin_rows, 8)
    for layer, rows in enumerate(a.pin_rows):
        for r, e in enumerate(rows):
            assert row_map[layer][e] == r


def test_cache_env_off_has_no_guards(monkeypatch):
    monkeypatch.setattr(ca, "VRAM_ELASTIC", False)
    monkeypatch.setattr(ca, "ONLINE_ADMISSION", False)
    cache = OffloadMoeCache(num_layers=L, num_experts=E, cache_size=SLOTS,
                            device=torch.device("cpu"))
    assert cache.elastic_guard is None
    assert cache.admission is None
    assert cache.note_prefill_step(1000) is None
    assert cache.note_decode_step() is None
    assert cache.admission_update(torch.zeros(L, E)) == []
    snap = cache.stats_snapshot()
    assert "elastic" not in snap and "admission" not in snap


def test_cache_hooks_drive_the_guard(monkeypatch):
    monkeypatch.setattr(ca, "VRAM_ELASTIC", True)
    cache = OffloadMoeCache(num_layers=L, num_experts=E, cache_size=SLOTS,
                            device=torch.device("cpu"), prefill_overlap=True)
    guard = cache.elastic_guard
    assert guard is not None and guard.borrowed_slots == BORROWED
    guard.set_pin_rows([[0, 1], [3]])
    cache.id_of_slot = _slot_map()
    guard.config.warmup_steps = 2
    cache.note_decode_step()
    cache.note_decode_step()
    assert cache.note_prefill_step(600) == [10, 13]
    guard.config.calm_steps = 1
    assert cache.note_decode_step() == [2, 6]
    snap = cache.stats_snapshot()
    assert snap["elastic"]["releases"] == 1
    assert snap["elastic"]["rewarms"] == 1


def test_cache_slot_bytes_feeds_rewarmed_bytes(monkeypatch):
    monkeypatch.setattr(ca, "VRAM_ELASTIC", True)
    cache = OffloadMoeCache(num_layers=1, num_experts=4, cache_size=8,
                            device=torch.device("cpu"))
    cache.set_bank_sources({"gate_up": [torch.randn(4, 8, 4)], "down": [torch.randn(4, 4, 4)]})
    # fp32 rows: (8*4 + 4*4) * 4 = 192 bytes per slot
    assert cache.elastic_guard.slot_bytes == 192
