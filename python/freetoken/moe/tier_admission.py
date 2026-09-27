"""LFRU admission contract for the disk tier's pinned (RAM-resident) set.

Verbatim port of colibri's `c/tier.h` (70 lines, semantics preserved exactly):

- frequency is the primary signal, recency only breaks close calls -- a recent
  access contributes at most 255 points while one frequency count is worth 256,
  so a merely recent expert cannot displace a genuinely hotter one;
- promotion requires beating the resident by 25% + 4 frequency units
  (hysteresis against ping-pong);
- heat decays by half per round so stale heat dies off;
- a saturated uint32 heat counter must go sticky, never wrap -- wrapping would
  let a colder expert look hotter and get admitted.

This is a *decision* layer: it answers "which pinned expert should be swapped
for which hotter non-resident one". Actuating an arbitrary pin set needs the
loader row remap (RAM rows are a contiguous expert-id prefix today) and is a
separate, deliberately deferred piece.
"""
from __future__ import annotations

import torch

_U32_MAX = (1 << 32) - 1


def heat_u32(route_hist_row: torch.Tensor) -> torch.Tensor:
    """Float routing mass -> uint32 heat with sticky saturation (never wraps).

    float32 cannot represent 2**32-1 (24-bit mantissa rounds it to 2**32),
    so the clamp must happen in float64 before the integer cast.
    """
    return route_hist_row.to(torch.float64).clamp(min=0, max=_U32_MAX).to(torch.int64)


def should_promote(hot: int, cold: int) -> bool:
    """colibri tier_should_promote: hot > cold + cold/4 + 4 (uint64 math)."""
    return hot > cold + (cold >> 2) + 4


def decay_value(heat: int) -> int:
    return heat >> 1


def decay(heat: torch.Tensor) -> torch.Tensor:
    """In-place per-round decay of a uint32 heat vector (halving)."""
    heat.bitwise_right_shift_(1)
    return heat


def lfru_score(heat: int, last: int, clock: int) -> int:
    """(heat << 8) | recent, where recent = max(0, 255 - age)."""
    age = clock - last
    recent = 255 - age if age < 255 else 0
    return (heat << 8) | recent


def pick_swap(heat: torch.Tensor, pinned: list[int]) -> tuple[int, int, int] | None:
    """Pick (pinned_slot, hot_eid, gain) to swap, or None when the margin says no.

    heat: int64 (uint32-range) per-expert heat. pinned: current resident ids.
    """
    if not pinned:
        return None
    cold_pos = min(range(len(pinned)), key=lambda z: int(heat[pinned[z]]))
    pinned_set = set(pinned)
    hot, hot_heat = -1, 0
    for e in range(heat.numel()):
        if e not in pinned_set and int(heat[e]) > hot_heat:
            hot, hot_heat = e, int(heat[e])
    if hot < 0:
        return None
    cold_heat = int(heat[pinned[cold_pos]])
    if not should_promote(hot_heat, cold_heat):
        return None
    return cold_pos, hot, hot_heat - cold_heat


def pick_lfru(heat: torch.Tensor, last: torch.Tensor, clock: int,
              pinned: list[int]) -> tuple[int, int, int] | None:
    """LFRU variant: score-ordered pick with the same 25%+4 hysteresis.

    Returns (pinned_slot, hot_eid, gain_in_frequency_units) or None.
    """
    if not pinned:
        return None

    def score(e: int) -> int:
        return lfru_score(int(heat[e]), int(last[e]), clock)

    cold_pos = min(range(len(pinned)), key=lambda z: score(pinned[z]))
    pinned_set = set(pinned)
    hot, hot_score = -1, -1
    for e in range(heat.numel()):
        if e not in pinned_set:
            s = score(e)
            if hot < 0 or s > hot_score:
                hot, hot_score = e, s
    if hot < 0:
        return None
    cold_score = score(pinned[cold_pos])
    # Hysteresis in score units: 25% margin + 4 frequency units (4 << 8).
    if hot_score <= cold_score + (cold_score >> 2) + (4 << 8):
        return None
    return cold_pos, hot, (hot_score - cold_score) >> 8
