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


def _relaunch_enabled(profile: bool, raw: str) -> bool:
    return profile and _truthy(raw)


PROFILE_ENABLED = _truthy(os.getenv("FREETOKEN_PREFILL_PROFILE", "0"))

# Measurement-only control inside the hit compaction: launch the same kernel a second time
# back to back. The kernel is idempotent (fixed-shape writes derived from the same inputs),
# so the second launch costs the launch path alone -- which separates "this launch is slow"
# from "the host was slow around it". Off unless the profile itself is on.
RELAUNCH_ENABLED = _relaunch_enabled(
    PROFILE_ENABLED, os.getenv("FREETOKEN_PREFILL_RELAUNCH", "0")
)

# Per-layer timeline: CUDA events on the copy and compute streams plus per-layer host
# timestamps, so a chunk can be attributed to "the device was blocked waiting for this
# layer's copy" versus "the device was idle and the host was the limit". The phase totals
# above only say which host region the time showed up in, and that region moves between
# configurations (measured 2026-09-22: the same attention code costs 13.7 ms/layer in one
# staging configuration and 1.06 ms/layer in another), so it cannot name the bottleneck.
# Off unless the profile itself is on, like the relaunch control.
TIMELINE_ENABLED = _relaunch_enabled(
    PROFILE_ENABLED, os.getenv("FREETOKEN_PREFILL_TIMELINE", "0")
)

# Where the per-layer rows are appended (one row per layer per chunk). The default keeps
# the writer out of the workspace; the window passes an explicit path.
TIMELINE_OUT = os.getenv("FREETOKEN_PREFILL_TIMELINE_OUT", "")


class _NullPhase:
    """Shared no-op context manager handed out while profiling is disabled."""

    __slots__ = ()

    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: object) -> bool:
        return False


_NULL_PHASE = _NullPhase()

# Batch-entry size buckets, in bytes: the staging ships one cudaMemcpyBatchAsync whose
# entries are coalesced expert-id runs, and a batch of many sub-64KiB entries is a different
# problem (per-entry overhead) from a few large ones. Fixed edges keep the histogram
# comparable between runs.
BATCH_EDGES = (64 * 1024, 1024 * 1024)


def timeline_path(base: str, pid: int) -> str:
    """Per-process timeline file: every TP rank records its own layers.

    Two ranks share the layer ids and the chunk counter, so one shared file interleaves
    rows from both processes and cannot be attributed afterwards -- learned when a first
    run produced a file with two headers and an unusable row order.
    """
    if not base:
        return ""
    stem, dot, ext = base.rpartition(".")
    if dot and ext and "/" not in ext:
        return f"{stem}.pid{pid}.{ext}"
    return f"{base}.pid{pid}"


def batch_bucket(nbytes: int) -> int:
    """Bucket index for one batch entry: 0 = <64 KiB, 1 = <1 MiB, 2 = larger."""
    if nbytes < BATCH_EDGES[0]:
        return 0
    if nbytes < BATCH_EDGES[1]:
        return 1
    return 2


def bank_byte_split(
    feats, num_experts: int, miss_runs, small_threshold: int, small_gather: bool = False
) -> Dict[str, tuple]:
    """``(entries, bytes)`` per staging origin for one layer (pure, unit-tested).

    Mirrors the copy plan in ``OffloadMoeCache._prefetch_split``: a bank whose per-expert
    row is below ``small_threshold`` is copied whole-layer even with zero misses, every
    other bank stages only the miss runs. The two together are the PCIe batch, so telling
    them apart is what decides whether the whole-layer small banks are worth removing
    (they cost bytes even at full cache residency, where all their rows are hits).

    ``small_gather`` mirrors the prototype flag of the same name: with the gather covering
    the small banks they stage by miss run like everything else, and reporting them as
    whole-layer there would have made the account contradict the batch it describes.
    """
    small = [0, 0]
    miss = [0, 0]
    for feat in feats:
        if feat < small_threshold and not small_gather:
            small[0] += 1
            small[1] += num_experts * feat
        elif len(miss_runs):
            miss[0] += len(miss_runs)
            miss[1] += int(sum(miss_runs)) * feat
    return {"small": (small[0], small[1]), "miss": (miss[0], miss[1])}


def _percentile(values, q: float) -> float:
    """Nearest-rank percentile; 0.0 for an empty sample."""
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[idx]


def analyze_timeline(rows) -> Dict[str, float]:
    """Summarize one chunk's per-layer timeline rows (pure, unit-tested).

    ``rows`` are per-layer dicts with device timestamps in ms on a common origin
    (``copy_begin``, ``copy_end`` on the copy stream; ``wait_done``, ``gemm_end`` on the
    compute stream, in program order) plus host wall timestamps (``h_copy_begin``,
    ``h_wait_done``) in seconds. The compute stream is FIFO, so the gap between the previous
    layer's ``gemm_end`` and this layer's ``wait_done`` is the device time spent blocked on
    this layer's staging; comparing it against the host gap over the same span separates
    "the device waited" from "the host was busy and the device had nothing queued".
    """
    stalls, copy_durs, lates, host_gaps = [], [], [], []
    for i, row in enumerate(rows):
        if "copy_begin" in row and "copy_end" in row:
            copy_durs.append(row["copy_end"] - row["copy_begin"])
        if i == 0:
            if "wait_done" in row and "chunk_begin" in row:
                stalls.append(row["wait_done"] - row["chunk_begin"])
        elif "wait_done" in row and "gemm_end" in rows[i - 1]:
            prev = rows[i - 1]["gemm_end"]
            stalls.append(row["wait_done"] - prev)
            if "copy_end" in row:
                # Positive when the copy landed after the consumer was already waiting.
                lates.append(row["copy_end"] - prev)
        if "h_wait_done" in row and i and "h_gemm_end" in rows[i - 1]:
            host_gaps.append((row["h_wait_done"] - rows[i - 1]["h_gemm_end"]) * 1000.0)
    total_stall = sum(v for v in stalls if v > 0.0)
    total_host = sum(v for v in host_gaps if v > 0.0)
    return {
        "layers": len(rows),
        "stall_sum_ms": total_stall,
        "stall_p50_ms": _percentile(stalls, 0.5),
        "stall_p90_ms": _percentile(stalls, 0.9),
        "host_gap_sum_ms": total_host,
        "copy_p50_ms": _percentile(copy_durs, 0.5),
        "copy_p90_ms": _percentile(copy_durs, 0.9),
        "copy_sum_ms": sum(copy_durs),
        "late_p50_ms": _percentile(lates, 0.5),
        # Share of the host's inter-layer time in which the device was demonstrably blocked.
        "stall_frac_of_host": (total_stall / total_host) if total_host > 0 else 0.0,
    }


class LayerTimeline:
    """Reused CUDA events + host timestamps for one chunk's per-layer timeline.

    One instance per process. Events are created lazily on first use (CUDA must already be
    initialized by then; creating them at import would break the engine's "no CUDA before
    the allocator is configured" ordering) and reused across chunks by (kind, layer) key,
    which is legal: re-recording overwrites a timestamp. Every layer records every kind, so
    a stale timestamp from a previous chunk can never be read back as this chunk's.
    """

    KINDS = ("copy_begin", "copy_end", "wait_done", "gemm_end")

    def __init__(self, event_factory=None) -> None:
        self._factory = event_factory
        self._events: Dict[tuple, object] = {}
        self._rows: list = []
        self._origin = None
        self._last = None
        self._queue: list = []
        self._chunk = 0
        self.dropped = 0
        self.summaries: list = []

    def _event(self, kind: str, layer: int):
        key = (kind, layer)
        ev = self._events.get(key)
        if ev is None:
            if self._factory is None:
                import torch  # local import: keeps this module import-safe off-GPU

                self._factory = lambda: torch.cuda.Event(enable_timing=True)
            ev = self._factory()
            self._events[key] = ev
        return ev

    def begin_chunk(self) -> None:
        """Start a new chunk, flushing any parked chunk first (see ``finish_chunk``)."""
        self.flush()
        self._chunk += 1
        self._rows = []
        self._origin = self._event("chunk_begin", 0)
        self._origin.record()
        self._last = self._origin
        self._rows.append({})

    def record(self, kind: str, layer: int, stream=None) -> None:
        if kind not in self.KINDS:
            raise ValueError(f"unknown timeline kind: {kind!r}")
        ev = self._event(kind, layer)
        ev.record(stream)
        self._last = ev
        while len(self._rows) <= layer:
            self._rows.append({})
        self._rows[layer][kind] = ev
        self._rows[layer]["h_" + kind] = time.perf_counter()

    def set_origin_row(self) -> None:
        """Attach the chunk origin to row 0 so layer 0's stall has a left edge."""
        if self._rows:
            self._rows[0]["chunk_begin"] = self._origin

    def finish_chunk(self) -> None:
        """Park this chunk for a deferred dump.

        Deferred on purpose: synchronizing here would make ``host`` include the device
        drain, which would both change the pipeline and make this arm's chunk time
        incomparable with every other arm.
        """
        self.set_origin_row()
        self._queue.append((self._chunk, self._rows, self._origin, self._last))
        self._rows = []
        # Bounded: a chunk whose device work never completes before it ages out is dropped
        # rather than growing the queue forever.
        while len(self._queue) > 4:
            self._queue.pop(0)
            self.dropped += 1

    def flush(self) -> int:
        """Dump every parked chunk whose device work has completed (never blocks).

        ``query`` instead of ``synchronize``: the timeline must not serialize the pipeline,
        and a chunk that is still in flight is simply retried at the next chunk boundary.
        """
        done = 0
        keep = []
        for pending in self._queue:
            if self._dump_one(pending):
                done += 1
            else:
                keep.append(pending)
        self._queue = keep
        return done

    def _dump_one(self, pending) -> bool:
        chunk, rows, origin, last = pending
        try:
            if not last.query():
                return False
        except Exception:  # pragma: no cover - device-side failure, keep the engine alive
            return True
        out = []
        for row in rows:
            flat = {}
            for key, value in row.items():
                if hasattr(value, "elapsed_time"):
                    try:
                        flat[key] = origin.elapsed_time(value)
                    except Exception:  # pragma: no cover
                        continue
                else:
                    flat[key] = value
            out.append(flat)
        self._write_rows(chunk, out)
        self.summaries.append(analyze_timeline(out))
        return True

    def _write_rows(self, chunk: int, rows) -> None:
        if not TIMELINE_OUT:
            return
        import csv

        path = timeline_path(TIMELINE_OUT, os.getpid())
        # Every field a reader needs to recompute the stall analysis from the CSV alone:
        # the device timeline (ms from the chunk origin) plus the host wall clock over the
        # same span.
        fields = [
            "chunk", "layer", "chunk_begin", "copy_begin", "copy_end", "wait_done",
            "gemm_end", "h_copy_begin", "h_copy_end", "h_wait_done", "h_gemm_end",
        ]
        try:
            new = not os.path.exists(path) or os.path.getsize(path) == 0
            with open(path, "a", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
                if new:
                    writer.writeheader()
                for layer, row in enumerate(rows):
                    out = dict(row)
                    out["chunk"] = chunk
                    out["layer"] = layer
                    writer.writerow(out)
        except OSError:
            pass


class PrefillProfiler:
    """Per-chunk accumulators. One instance serves every module that records phases."""

    def __init__(self, enabled: bool, timeline_enabled: bool = False) -> None:
        self.enabled = enabled
        self.timeline = timeline_enabled
        self._totals: Dict[str, float] = {}
        self._counts: Dict[str, int] = {}
        self._ops: Dict[str, int] = {}
        self._series: Dict[str, list] = {}
        self._batch_entries = 0
        self._batch_bytes = 0
        self._batch_buckets = [0, 0, 0]
        self._batch_driver_ms: list = []
        self._batch_sources: dict = {}
        self._layer = -1
        self._chunk_start = 0.0
        self._chunks = 0
        self._timeline: Optional[LayerTimeline] = None

    @property
    def timeline_enabled(self) -> bool:
        """True only while the env-gated per-layer timeline is recording."""
        return self.enabled and self.timeline

    def begin_chunk(self) -> None:
        if not self.enabled:
            return
        self._totals.clear()
        self._counts.clear()
        self._ops.clear()
        self._series.clear()
        self._batch_entries = 0
        self._batch_bytes = 0
        self._batch_buckets = [0, 0, 0]
        self._batch_driver_ms = []
        self._batch_sources = {}
        if self.timeline_enabled:
            if self._timeline is None:
                self._timeline = LayerTimeline()
            self._timeline.begin_chunk()
        self._chunk_start = time.perf_counter()

    def set_layer(self, layer_id: int) -> None:
        """Name the layer that subsequent recordings belong to (timeline only)."""
        if self.timeline_enabled:
            self._layer = layer_id

    def evt(self, kind: str, layer: Optional[int] = None, stream=None) -> None:
        """Record one point of the per-layer device timeline."""
        if not self.timeline_enabled:
            return
        assert self._timeline is not None
        self._timeline.record(kind, self._layer if layer is None else layer, stream)

    def batch(self, nbytes_seq, driver_ms: Optional[float] = None, sources=None) -> None:
        """Account one staging batch: entry count, bytes, size histogram, driver time.

        ``sources`` optionally maps an origin name (see ``bank_byte_split``) to its own
        ``(entries, bytes)``, which is what separates bytes nobody can avoid (a real miss)
        from bytes a fully resident cache still pays (a whole-layer small bank).
        """
        if not self.enabled:
            return
        n = len(nbytes_seq)
        self._batch_entries += n
        for value in nbytes_seq:
            self._batch_bytes += int(value)
            self._batch_buckets[batch_bucket(int(value))] += 1
        if driver_ms is not None:
            self._batch_driver_ms.append(driver_ms)
        if sources:
            for name, (entries, nbytes) in sources.items():
                slot = self._batch_sources.setdefault(name, [0, 0])
                slot[0] += int(entries)
                slot[1] += int(nbytes)

    def phase(self, name: str, series: bool = False):
        """Context manager that accumulates wall time under ``name``.

        ``series=True`` additionally keeps the per-layer samples, which is what shows a
        within-chunk trend (e.g. the host's per-call latency growing as a device queue
        fills) that a chunk total averages away.
        """
        if not self.enabled:
            return _NULL_PHASE
        return self._timed(name, series=series and self.timeline_enabled)

    def bump(self, name: str, n: int = 1) -> None:
        """Count host operations that are too cheap to time individually.

        A phase total answers "how long", this answers "how many": a region that costs
        milliseconds for three calls and one that costs the same for three hundred are
        different problems, and only the counter tells them apart.
        """
        if not self.enabled:
            return
        self._ops[name] = self._ops.get(name, 0) + n

    @contextmanager
    def _timed(self, name: str, series: bool = False) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            self._totals[name] = self._totals.get(name, 0.0) + dt
            self._counts[name] = self._counts.get(name, 0) + 1
            if series:
                self._series.setdefault(name, []).append((self._layer, dt * 1000.0))

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
        if self._ops:
            line += " | ops=" + ",".join(
                f"{name}:{value}" for name, value in sorted(self._ops.items())
            )
        if self._batch_entries or self._batch_bytes:
            lo, mid, hi = self._batch_buckets
            drv = self._batch_driver_ms
            line += (
                f" | batch=entries:{self._batch_entries},bytes:{self._batch_bytes}"
                f",lt64k:{lo},to1m:{mid},gt1m:{hi}"
                f",drv_p50:{_percentile(drv, 0.5):.2f}ms,drv_max:{max(drv) if drv else 0.0:.2f}ms"
            )
            if self._batch_sources:
                parts = ",".join(
                    f"{name}:{entries}/{nbytes}"
                    for name, (entries, nbytes) in sorted(self._batch_sources.items())
                )
                line += f",src={parts}"
        if self.timeline_enabled and self._timeline is not None:
            self._timeline.finish_chunk()
            line += (
                f" | timeline=pid:{os.getpid()}"
                f",parked:{len(self._timeline._queue)},dropped:{self._timeline.dropped}"
            )
            if self._series:
                line += " series=" + ",".join(
                    f"{name}:{len(samples)}" for name, samples in sorted(self._series.items())
                )
        if extra:
            line += f" | {extra}"
        return line

    def _fmt(self, name: str) -> str:
        return f"{self._totals.get(name, 0.0) * 1000:.1f}/{self._counts.get(name, 0)}"


_PROFILER = PrefillProfiler(PROFILE_ENABLED, TIMELINE_ENABLED)


def get_profiler() -> PrefillProfiler:
    """The process-wide profiler; every recording module shares it.

    ``enabled`` is re-read from the module flag on each call so a caller that caches the
    instance (the offload cache does) still follows the flag, and so tests can flip it.
    """
    _PROFILER.enabled = PROFILE_ENABLED
    _PROFILER.timeline = TIMELINE_ENABLED
    return _PROFILER
