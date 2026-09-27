"""CPU tests for the measured-placement autotune safety gate."""
import torch

from freetoken.moe.tier_autotune import AutotuneCache, decide


def test_fingerprint_is_stable_and_machine_scoped():
    fp = AutotuneCache.machine_fingerprint()
    assert fp == AutotuneCache.machine_fingerprint()
    assert len(fp) == 16
    # Any component change must change the fingerprint.
    fp2 = AutotuneCache.machine_fingerprint(gpu_names=["Some Other GPU"])
    assert fp2 != fp


def test_decide_requires_min_samples_and_median():
    c = AutotuneCache()
    assert decide(c, "k", "off", ["off", "on"]) is None  # nothing recorded, no measurer
    c.record("k", "on", 10.0)
    c.record("k", "off", 5.0)
    assert decide(c, "k", "off", ["off", "on"]) is None  # <3 samples each
    c.record("k", "on", 12.0); c.record("k", "on", 11.0)
    c.record("k", "off", 6.0); c.record("k", "off", 4.0)
    # medians: off=5.0, on=11.0; improvement (11-5)/5 = 120% > 3% -> adopt.
    d = decide(c, "k", "off", ["off", "on"])
    assert d and d.adopted == "on" and not d.rolled_back


def test_decide_rolls_back_below_margin():
    c = AutotuneCache()
    for v in (10.0, 10.2, 9.8):
        c.record("k", "off", v)
    for v in (10.1, 10.0, 10.3):  # +0% median -- inside the 3% margin
        c.record("k", "on", v)
    d = decide(c, "k", "off", ["off", "on"], lambda: 0.0)
    assert d and d.adopted == "off" and d.rolled_back and d.median["on"] > 0


def test_lower_is_better_metric():
    c = AutotuneCache()
    for v in (10.0, 10.0, 10.0):
        c.record("lat", "a", v)
    for v in (5.0, 5.1, 4.9):  # half the latency
        c.record("lat", "b", v)
    d = decide(c, "lat", "a", ["a", "b"], lambda: 0.0, higher_is_better=False)
    assert d and d.adopted == "b"


def test_config_sha_pins_measurement_context():
    c = AutotuneCache()
    c.record("k", "on", 1.0, config={"window": 4})
    d = decide(c, "k", "off", ["on"], lambda: 0.0, config={"window": 8})
    assert d is None  # different config sha -> measurement not applicable


def test_persistence_roundtrip(tmp_path):
    p = tmp_path / "auto.json"
    c = AutotuneCache(p)
    c.record("k", "on", 3.0, config={"w": 1})
    c2 = AutotuneCache(p)
    assert c2.config_sha({"w": 1}) in c2.entries.get("k:on", {})
    assert c2.entries["k:on"][c2.config_sha({"w": 1})] == [3.0]


def test_live_measure_path(tmp_path):
    c = AutotuneCache(tmp_path / "m.json")
    calls = {"n": 0}
    def measure():
        calls["n"] += 1
        return 100.0 if calls["n"] <= 3 else 50.0  # off=100 x3, then on=50 x3
    d = decide(c, "live", "off", ["off", "on"], measure, higher_is_better=False)
    assert calls["n"] == 6 and d.adopted == "on"
    # Cached decision short-circuits further measurement.
    d2 = decide(c, "live", "off", ["off", "on"], measure)
    assert calls["n"] == 6 and d2.adopted == "on"
