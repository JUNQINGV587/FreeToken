"""Measured-placement autotune safety gate for disk-tier knobs (colibri §7.5 port).

Colibri's methodology, reduced to its transferable shape:

- knobs are *tuned, never edited*: the gate measures candidate values and
  adopts one; it never mutates configuration in place;
- a measurement is the **median of >=3 samples** under a pinned config
  (sha256 of the config dict), so a stale measurement under a different
  config can never leak into a decision;
- **regression = rollback**: a candidate is adopted only if it beats the
  baseline by more than a margin; otherwise the baseline stays;
- decisions are cached by a **machine fingerprint** (CPU + GPU + block
  device + engine version), so a moved model dir or swapped GPU re-measures
  instead of trusting a foreign machine's numbers.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

MIN_SAMPLES = 3
DEFAULT_MARGIN = 0.03  # adopt only on >3% improvement; else roll back


@dataclass
class Decision:
    adopted: str
    rolled_back: bool
    median: dict[str, float]


class AutotuneCache:
    """Machine-fingerprint-keyed measurement/decision store (JSON file)."""

    def __init__(self, path: str | Path | None = None, fingerprint: str | None = None):
        self.path = Path(path) if path else None
        self.fingerprint = fingerprint or self.machine_fingerprint()
        self.entries: dict[str, dict[str, list[float]]] = {}
        self.decisions: dict[str, str] = {}
        if self.path and self.path.exists():
            blob = json.loads(self.path.read_text())
            if blob.get("fingerprint") == self.fingerprint:
                self.entries = blob.get("entries", {})
                self.decisions = blob.get("decisions", {})

    @staticmethod
    def machine_fingerprint(cpu: str | None = None, gpu_names: list[str] | None = None,
                            blockdev: str | None = None, version: str | None = None) -> str:
        cpu = cpu if cpu is not None else platform.processor() or platform.machine()
        if gpu_names is None:
            gpu_names = []
            try:
                import torch

                if torch.cuda.is_available():
                    gpu_names = sorted(torch.cuda.get_device_name(i)
                                       for i in range(torch.cuda.device_count()))
            except Exception:
                gpu_names = []
        if blockdev is None:
            try:
                blockdev = Path(
                    subprocess.run(["findmnt", "-n", "-o", "SOURCE", "/"],
                                   capture_output=True, text=True, timeout=5).stdout.strip()
                ).name
            except Exception:
                blockdev = "unknown"
        if version is None:
            try:
                import freetoken

                version = getattr(freetoken, "__version__", "dev")
            except Exception:
                version = "dev"
        raw = json.dumps({"cpu": cpu, "gpus": gpu_names, "blockdev": blockdev,
                          "version": version}, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    @staticmethod
    def config_sha(config: dict | None) -> str:
        return hashlib.sha256(
            json.dumps(config or {}, sort_keys=True, default=str).encode()).hexdigest()[:12]

    def _key(self, knob: str, value: str) -> str:
        return f"{knob}:{value}"

    def record(self, knob: str, value: str, sample: float, config: dict | None = None) -> None:
        sha = self.config_sha(config)
        self.entries.setdefault(self._key(knob, value), {}).setdefault(sha, []).append(sample)
        self._save()

    def samples(self, knob: str, value: str, config: dict | None = None) -> list[float]:
        return self.entries.get(self._key(knob, value), {}).get(self.config_sha(config), [])

    def _save(self) -> None:
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(
                {"fingerprint": self.fingerprint, "entries": self.entries,
                 "decisions": self.decisions}))


def decide(cache: AutotuneCache, knob: str, baseline: str, candidates: list[str],
           measure: Callable[[], float] | None = None, config: dict | None = None,
           higher_is_better: bool = True, margin: float = DEFAULT_MARGIN,
           n_samples: int = MIN_SAMPLES) -> Decision | None:
    """Pick a knob value by measurement. None when evidence is insufficient.

    Every candidate must reach ``n_samples`` recorded samples for the current
    config sha; missing ones are measured live through ``measure()`` -- but
    only when it is the candidate currently in effect, so the caller drives
    the knob externally and calls ``decide`` once per live setting, or passes
    a ``measure`` that itself sets the knob (see tests for both styles).
    A cached decision for this (knob, config) short-circuits measurement.
    """
    dkey = f"{knob}|{cache.config_sha(config)}"
    if dkey in cache.decisions:
        return Decision(adopted=cache.decisions[dkey], rolled_back=False, median={})

    def better(a: float, b: float) -> float:
        return (a - b) / abs(b) if higher_is_better else (b - a) / abs(b)

    values = [baseline] + [c for c in candidates if c != baseline]
    medians: dict[str, float] = {}
    # Deterministic protocol: candidates are visited in order; measure() must
    # return the metric for the candidate currently being visited (the caller
    # engages it before/while we poll, as in the live-measure test).
    for v in values:
        if len(cache.samples(knob, v, config)) < n_samples and measure is None:
            return None  # evidence insufficient and no live measurer supplied
        while len(cache.samples(knob, v, config)) < n_samples:
            cache.record(knob, v, measure(), config)
        s = sorted(cache.samples(knob, v, config))
        medians[v] = s[len(s) // 2] if len(s) % 2 else (s[len(s) // 2 - 1] + s[len(s) // 2]) / 2

    base = medians[baseline]
    if base == 0:
        return None
    best, best_gain = baseline, 0.0
    for v in candidates:
        if v == baseline:
            continue
        g = better(medians[v], base)
        if g > best_gain:
            best, best_gain = v, g
    rolled_back = best_gain <= margin
    adopted = baseline if rolled_back else best
    cache.decisions[dkey] = adopted
    cache._save()
    return Decision(adopted=adopted, rolled_back=rolled_back, median=medians)
