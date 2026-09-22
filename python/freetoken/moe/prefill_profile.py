"""Opt-in host-time profile of one MoE prefill chunk (``FREETOKEN_PREFILL_PROFILE=1``).

Why: a prefill batch costs ~1.2 s of scheduler-main-thread CPU before its first token, and
that cost is flat in prompt length (measured 2026-09-22: 7 -> 4343 prompt tokens moved the
rank-0 CPU total by 10 ms, and ``moe.prefill_rows`` grew by 48 * 256 = the whole local
expert set per batch). Host CPU, not GPU, is the TTFT floor, so the next optimization has
to target a measured phase. This module attributes that time to attention / PLE / MoE
staging / expert GEMM, per chunk, in one log line.

Disabled by default: with the env unset every call site is a single no-op method call and
no timestamp is read, so behavior and kernel scheduling stay bit-identical.
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from typing import Dict, Iterator, Optional

_TRUTHY = {"1", "true", "yes", "on"}

# Phases printed first, in this order; anything else is appended sorted. The first eight
# are the top-level partition of the chunk, so their sum plus "unaccounted" is the host
# wall time.
_TOP_LEVEL = (
    "entry",
    "ple",
    "attn_mix",
    "attn_core",
    "attn_combine",
    "mlp_mix",
    "moe_total",
    "mlp_combine",
)


def _truthy(value: str) -> bool:
    return value.strip().lower() in _TRUTHY


PROFILE_ENABLED = _truthy(os.getenv("FREETOKEN_PREFILL_PROFILE", "0"))


class _NullPhase:
    """Shared no-op context manager handed out while profiling is disabled."""

    __slots__ = ()

    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: object) -> bool:
        return False


_NULL_PHASE = _NullPhase()


class PrefillProfiler:
    """Per-chunk accumulators. One instance serves every module that records phases."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self._totals: Dict[str, float] = {}
        self._counts: Dict[str, int] = {}
        self._chunk_start = 0.0
        self._chunks = 0

    def begin_chunk(self) -> None:
        if not self.enabled:
            return
        self._totals.clear()
        self._counts.clear()
        self._chunk_start = time.perf_counter()

    def phase(self, name: str):
        """Context manager that accumulates wall time under ``name``."""
        if not self.enabled:
            return _NULL_PHASE
        return self._timed(name)

    @contextmanager
    def _timed(self, name: str) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            self._totals[name] = self._totals.get(name, 0.0) + dt
            self._counts[name] = self._counts.get(name, 0) + 1

    def end_chunk(self, *, layers: int, extra: str = "") -> Optional[str]:
        """Close the chunk and return its report line, or None while disabled."""
        if not self.enabled:
            return None
        host = time.perf_counter() - self._chunk_start
        self._chunks += 1
        accounted = sum(self._totals.get(name, 0.0) for name in _TOP_LEVEL)
        parts = []
        for name in _TOP_LEVEL:
            parts.append(f"{name}={self._fmt(name)}")
        rest = sorted(set(self._totals) - set(_TOP_LEVEL))
        for name in rest:
            parts.append(f"{name}={self._fmt(name)}")
        unaccounted = host - accounted
        line = (
            f"MoE prefill profile: chunk={self._chunks} layers={layers} "
            f"host={host * 1000:.1f}ms unaccounted={unaccounted * 1000:.1f}ms | "
            + " ".join(parts)
        )
        if extra:
            line += f" | {extra}"
        return line

    def _fmt(self, name: str) -> str:
        return f"{self._totals.get(name, 0.0) * 1000:.1f}/{self._counts.get(name, 0)}"


_PROFILER = PrefillProfiler(PROFILE_ENABLED)


def get_profiler() -> PrefillProfiler:
    """The process-wide profiler; every recording module shares it.

    ``enabled`` is re-read from the module flag on each call so a caller that caches the
    instance (the offload cache does) still follows the flag, and so tests can flip it.
    """
    _PROFILER.enabled = PROFILE_ENABLED
    return _PROFILER
