"""VRAM cache elastic release/rewarm + route-count online admission (dsv41 port item 5).

Reference semantics: dsv41-flash-offload ct/ct_vllm.py -- ec_release/ec_rewarm/
ec_summary (lines 379-433) and the replan loop (route counts + 0.9 decay + admit
cap + DSV41_EC_HYST hysteresis, lines ~290-376).

Both classes are pure host-side policy: they plan WHICH rows a release or a swap
may touch and keep the side-profile counters. The CUDA mechanics (unmap +
empty_cache + re-add copies, and the scheduler call sites that feed step hooks)
are GPU-battery wiring on top of these plans. Everything is env-gated and
default-off; off leaves the cache byte-identical to the pre-item-5 behavior.
"""

from __future__ import annotations

import os

import torch


def _env_bool(name: str) -> bool:
    return os.getenv(name, "0").strip().lower() in {"1", "true", "yes", "on"}


# Master switches (item 5.3.4). Default off == current behavior.
VRAM_ELASTIC = _env_bool("FREETOKEN_VRAM_ELASTIC")
ONLINE_ADMISSION = _env_bool("FREETOKEN_ONLINE_ADMISSION")

# Honesty bit for /v1/stats (review 202610, m4 finding): until the GPU battery
# (review item #12) lands the engine/scheduler call sites AND the CUDA apply,
# these policies run with no production traffic -- "enabled" must not be read
# as "working". Flip to True in the battery-#12 commit that wires the
# scheduler's note_prefill_step/note_decode_step/admission_update calls and
# the release/rewarm CUDA apply (unmap + empty_cache + re-add).
WIRED = False


class ElasticConfig:
    """Knobs for the elastic guard (dsv41 DSV41_EC_* equivalents)."""

    def __init__(self) -> None:
        # Prefill steps with MORE tokens than this trigger a release (dsv41: >512).
        self.prefill_tokens = int(os.getenv("FREETOKEN_EC_PREFILL_TOKENS", "512"))
        # Consecutive decode steps before the keep set rewarms (dsv41: 4). Sooner
        # and agent-style interleaved loads bounce the rows back and forth.
        self.calm_steps = int(os.getenv("FREETOKEN_EC_CALM_STEPS", "4"))
        # Decode steps before the guard arms at all (dsv41 DSV41_EC_WARMUP=64):
        # early routing is too noisy to pick a keep set worth re-adding.
        self.warmup_steps = int(os.getenv("FREETOKEN_EC_WARMUP_STEPS", "64"))


class AdmissionConfig:
    """Knobs for online admission (dsv41 replan equivalents, conservative defaults)."""

    def __init__(self) -> None:
        # Decode steps between evaluations (dsv41 replans every 16; slower here --
        # our pin set is a host-RAM layout and a swap reprices disk-tier rows).
        self.period = int(os.getenv("FREETOKEN_ADMISSION_PERIOD", "64"))
        # Max row swaps per evaluation (dsv41 caps admits at 48 per replan).
        self.max_swaps = int(os.getenv("FREETOKEN_ADMISSION_MAX_SWAPS", "8"))
        # A pin row is a swap-out candidate only STRICTLY above this decayed miss
        # rate (item 5.3.2: >10%).
        self.miss_threshold = float(os.getenv("FREETOKEN_ADMISSION_MISS_THRESHOLD", "0.10"))
        # A challenger must beat the incumbent's decayed misses by this factor
        # (dsv41 DSV41_EC_HYST); without it borderline rows thrash every cycle.
        self.hysteresis = float(os.getenv("FREETOKEN_ADMISSION_HYSTERESIS", "1.5"))
        # Per-cycle score decay (dsv41 DSV41_EC_DECAY=0.9).
        self.decay = float(os.getenv("FREETOKEN_ADMISSION_DECAY", "0.9"))


class ElasticCacheGuard:
    """Elastic release/rewarm bookkeeping for the MoE slot cache (item 5.3.1).

    A release unmaps only slots OUTSIDE the borrowed prefill ring whose resident
    (layer, expert) is not pinned; the released residents form the keep set that
    rewarm re-adds after ``calm_steps`` consecutive decode steps. Pin rows are
    the rewarm floor -- releasing them would re-add them right back, so they are
    frozen instead.
    """

    def __init__(self, num_layers: int, num_experts: int, cache_size: int,
                 borrowed_slots: int, config: ElasticConfig | None = None) -> None:
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.cache_size = cache_size
        self.borrowed_slots = borrowed_slots
        self.config = config or ElasticConfig()
        # Flat ids (layer * num_experts + expert) a release must keep resident.
        self._pinned: set[int] = set()
        self._released = False
        self._keep: list[int] = []
        self._calm = 0
        self._steps = 0
        # Filled by set_bank_sources once the bank row shapes are known; only
        # feeds the rewarmed-bytes side profile.
        self.slot_bytes = 0
        # Side profile (item 5.3.3): every release -> re-add round is counted.
        self.releases = 0
        self.rewarms = 0
        self.released_rows_total = 0
        self.rewarmed_rows_total = 0
        self.rewarmed_bytes_total = 0

    def set_pin_rows(self, pin_rows: list[list[int]]) -> None:
        """Freeze set: per-layer local expert ids (the disk tier's pin layout)."""
        self._pinned = {
            layer * self.num_experts + int(e)
            for layer, rows in enumerate(pin_rows)
            for e in rows
        }

    @property
    def released(self) -> bool:
        return self._released

    @property
    def keep(self) -> list[int]:
        return list(self._keep)

    def plan_release(self, id_of_slot) -> list[int]:
        """Slot ids a release may unmap, given the slot -> flat-id map.

        Off-limits: the borrowed ring prefix (slots < borrowed_slots -- mainline
        ring bookkeeping; the 2a pin-source ledger on port/m3-prefill-source must
        keep this same non-borrowed-only scope when the branches merge) and every
        pinned (layer, expert) row.
        """
        return [
            slot
            for slot in range(self.borrowed_slots, self.cache_size)
            if 0 <= int(id_of_slot[slot]) and int(id_of_slot[slot]) not in self._pinned
        ]

    def note_prefill(self, tokens: int, id_of_slot=None) -> list[int] | None:
        """Prefill-step hook: the slots to release, or None.

        dsv41 releases ahead of any over-threshold step; below warmup the keep
        set would be routing noise, so the guard stays inert. Any prefill (even a
        small one) resets the calm countdown -- it is not a decode step.
        """
        self._calm = 0
        if self._released or tokens <= self.config.prefill_tokens:
            return None
        if self._steps < self.config.warmup_steps or id_of_slot is None:
            return None
        slots = self.plan_release(id_of_slot)
        if not slots:
            return None
        self._keep = [int(id_of_slot[s]) for s in slots]
        self._released = True
        self.releases += 1
        self.released_rows_total += len(slots)
        return slots

    def note_decode(self) -> list[int] | None:
        """Decode-step hook: the keep set to rewarm once calm, else None."""
        self._steps += 1
        if not self._released:
            return None
        self._calm += 1
        if self._calm < self.config.calm_steps:
            return None
        keep = self._keep
        self._keep = []
        self._released = False
        self.rewarms += 1
        self.rewarmed_rows_total += len(keep)
        self.rewarmed_bytes_total += len(keep) * self.slot_bytes
        return keep

    def summary(self) -> dict:
        return {
            "enabled": True,
            "wired": WIRED,
            "released": self._released,
            "keep_rows": len(self._keep),
            "releases": self.releases,
            "rewarms": self.rewarms,
            "released_rows": self.released_rows_total,
            "rewarmed_rows": self.rewarmed_rows_total,
            "rewarmed_bytes": self.rewarmed_bytes_total,
        }


class OnlineAdmission:
    """Route-count online admission for the RAM pin set (item 5.3.2).

    Every ``period`` decode steps the decayed per-row signals are evaluated: pin
    rows whose decayed miss rate exceeds ``miss_threshold`` swap places with the
    highest-miss non-pin rows, subject to hysteresis and a per-cycle batch cap.
    The pin layout stays in the pin_manager shape (per-layer sorted local-id
    lists -- disk_tier.pin_rows_to_row_map's input), so applying a swap is a
    layout rebuild, never a new format.
    """

    def __init__(self, num_layers: int, num_experts: int,
                 pin_rows: list[list[int]] | None = None,
                 config: AdmissionConfig | None = None) -> None:
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.config = config or AdmissionConfig()
        self._pin: list[list[int]] = [[] for _ in range(num_layers)]
        if pin_rows is not None:
            self.set_pin_rows(pin_rows)
        self._scores = torch.zeros(num_layers, num_experts, dtype=torch.float64)
        self._misses = torch.zeros(num_layers, num_experts, dtype=torch.float64)
        self._since_eval = 0
        # Side profile (item 5.3.3/5.3.5).
        self.evaluations = 0
        self.swaps = 0
        self._last_miss_rate_max = 0.0

    def set_pin_rows(self, pin_rows: list[list[int]]) -> None:
        assert len(pin_rows) == self.num_layers
        self._pin = [sorted(int(e) for e in rows) for rows in pin_rows]

    @property
    def pin_rows(self) -> list[list[int]]:
        return [list(rows) for rows in self._pin]

    def note_step(self) -> None:
        self._since_eval += 1

    @property
    def evaluation_due(self) -> bool:
        """True once ``period`` decode steps elapsed since the last evaluation."""
        return self._since_eval >= self.config.period

    def update(self, route_counts, miss_counts=None) -> list[tuple[tuple[int, int], tuple[int, int]]]:
        """Fold one cycle's per-row counts in; evaluate when the period elapses.

        ``route_counts``/``miss_counts``: [L, E] per-row routed entries / misses
        since the last update. The wiring source is the cpu tier split kernel's
        [L,E] per-expert counters (moe/cpu_tier.py, port/m4-perexpert-counts);
        offload_cache.admission_update() pulls their deltas when called without
        explicit counts. ``miss_counts=None`` remains the conservative fallback
        (no miss signal: zero misses, so no pin row crosses the threshold and
        the layout stays put). Returns the applied swaps as
        [((layer, out_e), (layer, in_e)), ...].
        """
        counts = torch.as_tensor(route_counts, dtype=torch.float64)
        assert counts.shape == (self.num_layers, self.num_experts), counts.shape
        cfg = self.config
        self._scores.mul_(cfg.decay).add_(counts)
        if miss_counts is not None:
            misses = torch.as_tensor(miss_counts, dtype=torch.float64)
            assert misses.shape == counts.shape, misses.shape
            self._misses.mul_(cfg.decay).add_(misses)
        else:
            self._misses.mul_(cfg.decay)
        if self._since_eval < cfg.period:
            return []
        self._since_eval = 0
        self.evaluations += 1
        return self._evaluate()

    def _evaluate(self) -> list[tuple[tuple[int, int], tuple[int, int]]]:
        cfg = self.config
        rates = self._misses / self._scores.clamp(min=1.0)
        self._last_miss_rate_max = float(rates.max()) if rates.numel() else 0.0
        swaps: list[tuple[int, int, int]] = []
        for layer in range(self.num_layers):
            pin_set = set(self._pin[layer])
            # Swap-out candidates: pinned rows strictly above the threshold,
            # worst miss rate first (id tie-break for determinism).
            outs = sorted(
                (e for e in pin_set if float(rates[layer, e]) > cfg.miss_threshold),
                key=lambda e: (-float(rates[layer, e]), e),
            )
            if not outs:
                continue
            # Challengers: non-pin rows by decayed misses (hottest first).
            ins = sorted(
                (e for e in range(self.num_experts) if e not in pin_set),
                key=lambda e: (-float(self._misses[layer, e]), e),
            )
            consumed: set[int] = set()
            for out_e in outs:
                if len(swaps) >= cfg.max_swaps:
                    break
                bar = cfg.hysteresis * float(self._misses[layer, out_e])
                in_e = next(
                    (e for e in ins
                     if e not in consumed and float(self._misses[layer, e]) > bar),
                    None,
                )
                if in_e is None:
                    # No challenger clears THIS incumbent's bar; later outs have
                    # different bars, so keep scanning (not break).
                    continue
                consumed.add(in_e)
                swaps.append((layer, out_e, in_e))
        for layer, out_e, in_e in swaps:
            rows = self._pin[layer]
            rows.remove(out_e)
            rows.append(in_e)
            rows.sort()
        self.swaps += len(swaps)
        return [((layer, out_e), (layer, in_e)) for layer, out_e, in_e in swaps]

    def summary(self) -> dict:
        return {
            "enabled": True,
            "wired": WIRED,
            "evaluations": self.evaluations,
            "swaps": self.swaps,
            "miss_rate_max": round(self._last_miss_rate_max, 4),
            "pin_rows": sum(len(r) for r in self._pin),
        }
