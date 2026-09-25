"""Runtime metrics for /v1/stats. The FrontendManager owns one StatsTracker and feeds it
every UserReply (the single chokepoint in listen()). kv/mamba/vram keep their last-known-value
semantics like ShellStats; throughput uses an independent sliding-window rate (NOT cumulative
average like tok_s), so idle polls decay to zero by wall clock."""

from __future__ import annotations

import time
from collections import deque
from typing import Any


class StatsTracker:
    def __init__(self, window_s: float = 5.0) -> None:
        self.window_s = window_s
        # maxlen bounds memory on the headless path: stale-sample eviction is poll-driven
        # (only _rate() trims to window_s), and clients that never hit /v1/stats (e.g.
        # codex/claude via /v1/chat/completions) would otherwise grow these unbounded.
        # 4096 is generous vs the sliding window's span at any realistic reply rate.
        self._decode: "deque[tuple[float, int]]" = deque(maxlen=4096)
        self._prefill: "deque[tuple[float, int]]" = deque(maxlen=4096)
        self._inflight: set[int] = set()
        # Requests for which an abort was dispatched but the scheduler's explicit terminal
        # acknowledgement has not arrived yet. They remain active until that barrier, while
        # any sampled tokens racing the abort continue to count toward lifetime totals.
        self._aborting: set[int] = set()
        self.completed = 0
        # Cumulative prompt/completion tokens since this process started (lifetime for THIS served
        # model). Exposed in /v1/stats so the desktop can diff consecutive polls into per-model
        # "cost saved by running locally" accounting. Monotonic; resets when the process restarts.
        self.prompt_tokens_total = 0
        self.completion_tokens_total = 0
        # Lifetime prefix-cache hits: tokens of admitted prompts served from the radix
        # cache instead of recomputed. Arrives per request on the same reply as
        # prompt_tokens_delta (UserReply.cached_tokens); hit_ratio = this /
        # prompt_tokens_total (same tokens denominator as vLLM's hits/queries pair).
        self.cached_tokens_total = 0
        # Last MoE slot-cache snapshot stamped by the scheduler (miss/residency/routing
        # concentration); None until the first sample or on non-offload models.
        self.moe_stats: dict | None = None
        # Last encoder embedding-cache snapshot (entries/bytes) stamped by the scheduler; None
        # until the first sample or when the process builds no encoder tower.
        self.mm_stats: dict | None = None
        # Last host KV-tier snapshot stamped by the scheduler (spills/restores/refusals and live
        # residency); None until the first sample or when the tier is not enabled -- None is NOT
        # "enabled but idle", which is why the field is absent rather than zeroed.
        self.host_tier_stats: dict | None = None
        # Cumulative timing, summed per request like llama.cpp's prompt/predicted seconds, so a
        # poller can diff consecutive polls into speeds; the sliding-window rates above decay
        # to zero between polls. Prefill runs from admission to a request's first output reply;
        # decode is the time between its later output replies, and decode_tokens_total counts
        # only the tokens those replies carried. cached_prompt_tokens_total mirrors
        # cached_tokens_total above under upstream #553's name so future stats.py merges align.
        self.cached_prompt_tokens_total = 0
        self.prefill_seconds_total = 0.0
        self.decode_seconds_total = 0.0
        self.decode_tokens_total = 0
        self._admitted_at: dict[int, float] = {}
        self._last_output_at: dict[int, float] = {}
        self.kv_used_pages = 0
        self.kv_total_pages = 0
        self.mamba_used_slots = 0
        self.mamba_total_slots = 0
        self.swa_used_tokens = 0
        self.swa_total_tokens = 0
        self.vram_bytes = 0

    @property
    def active(self) -> int:
        return len(self._inflight)

    @property
    def inflight_uids(self) -> tuple[int, ...]:
        """Stable snapshot used by prepare-stop to abort every still-admitted request."""
        return tuple(sorted(self._inflight))

    def on_new_user(self, uid: int, now: float | None = None) -> None:
        self._inflight.add(uid)
        self._aborting.discard(uid)
        self._admitted_at[uid] = time.monotonic() if now is None else now

    def on_abort(self, uid: int) -> None:
        if uid in self._inflight:
            self._aborting.add(uid)

    def observe(self, reply: Any, now: float | None = None) -> None:
        t = time.monotonic() if now is None else now
        uid = getattr(reply, "uid", None)
        if getattr(reply, "completion_tokens_delta", 0) > 0:
            self._decode.append((t, reply.completion_tokens_delta))
            self.completion_tokens_total += reply.completion_tokens_delta
            self._observe_output_timing(uid, reply.completion_tokens_delta, t)
        if getattr(reply, "prompt_tokens_delta", 0) > 0:
            self._prefill.append((t, reply.prompt_tokens_delta))
            self.prompt_tokens_total += reply.prompt_tokens_delta
        if getattr(reply, "cached_tokens", 0) > 0:
            self.cached_tokens_total += reply.cached_tokens
            self.cached_prompt_tokens_total += reply.cached_tokens
        if getattr(reply, "moe_stats", None) is not None:
            self.moe_stats = reply.moe_stats
        if getattr(reply, "mm_stats", None) is not None:
            self.mm_stats = reply.mm_stats
        if getattr(reply, "host_tier_stats", None) is not None:
            self.host_tier_stats = reply.host_tier_stats
        if getattr(reply, "kv_total_pages", 0) > 0:  # ignore 0/0 (prompt reply, owned-KV)
            self.kv_used_pages = reply.kv_used_pages
            self.kv_total_pages = reply.kv_total_pages
        if getattr(reply, "mamba_total_slots", 0) > 0:  # hybrid (GDN) only
            self.mamba_used_slots = reply.mamba_used_slots
            self.mamba_total_slots = reply.mamba_total_slots
        if getattr(reply, "swa_total_tokens", 0) > 0:  # SWA (window pool) only
            self.swa_used_tokens = reply.swa_used_tokens
            self.swa_total_tokens = reply.swa_total_tokens
        if getattr(reply, "gpu_mem_bytes", 0) > 0:
            self.vram_bytes = reply.gpu_mem_bytes
        if getattr(reply, "finished", False):
            self._admitted_at.pop(uid, None)
            self._last_output_at.pop(uid, None)
            if uid in self._inflight:
                self._inflight.discard(uid)
                if uid in self._aborting:
                    self._aborting.discard(uid)
                else:
                    self.completed += 1

    def _observe_output_timing(self, uid: Any, tokens: int, t: float) -> None:
        last = self._last_output_at.get(uid)
        if last is None:
            # The first output reply ends prefill. Tokens riding on it (overlap can deliver
            # several) were measured by no decode interval, so they are not decode work.
            admitted = self._admitted_at.pop(uid, None)
            if admitted is not None:
                self.prefill_seconds_total += max(0.0, t - admitted)
        else:
            self.decode_seconds_total += max(0.0, t - last)
            self.decode_tokens_total += tokens
        self._last_output_at[uid] = t

    def _rate(self, window: "deque[tuple[float, int]]", now: float | None) -> float:
        t = time.monotonic() if now is None else now
        cutoff = t - self.window_s
        while window and window[0][0] < cutoff:
            window.popleft()
        if not window:
            return 0.0
        total = sum(n for _ts, n in window)
        span = max(t - window[0][0], 1e-9)
        return total / span

    def decode_tps(self, now: float | None = None) -> float:
        return self._rate(self._decode, now)

    def prefill_tps(self, now: float | None = None) -> float:
        return self._rate(self._prefill, now)


def derive_model_card(config: Any) -> dict:
    """attn enum + moe bool + ctx from the model config; input_modalities is what the API accepts right now."""
    mc = config.model_config
    if getattr(mc, "has_linear_attention", False):
        attn = "hybrid_linear"
    elif getattr(mc, "has_swa_attention", False):
        attn = "hybrid_swa"
    else:
        attn = "mha"
    return {
        "id": config.served_model_name,
        "ctx": config.max_seq_len,
        "attn": attn,
        "moe": bool(getattr(mc, "is_moe", False)),
        "input_modalities": ["text", *sorted(config.served_modalities)],
    }


def _resolved_page_size(state: Any, config: Any) -> int:
    """The engine's REAL KV page size (tokens per page).

    ``_adjust_config`` runs inside the scheduler process, so the frontend's ``config.page_size``
    can still hold the CLI default while the engine actually pages at, say, 64 (qsa_sparse).
    Reporting that default made ``total_pages`` look like a token count (4096 "tokens" for a
    262144-token pool). Prefer the value the engine published in its readiness meta.
    """
    pools = getattr(state, "cache_pools", None) or {}
    try:
        value = int(pools.get("page_size") or 0)
    except (TypeError, ValueError):
        value = 0
    if value <= 0:
        value = int(getattr(config, "page_size", 1) or 1)
    return max(1, value)


def _swa_page_size(config: Any) -> int:
    """The window pool's own page unit: P (window_size) for DSV4, 1 token for radix-SWA.
    Mirrors compute_cache_pools' swa_page_size."""
    dsv4 = getattr(getattr(config, "model_config", None), "dsv4_args", None)
    if dsv4 is not None:
        return int(getattr(dsv4, "window_size", 0) or 1)
    return 1


def build_stats(state: Any, p95_ms: int, ttft_mean_ms: int) -> dict:
    """Full /v1/stats doc. throughput is 0 when idle; kv/mamba/swa are null
    when their total is 0 (owned-KV / non-hybrid / non-SWA). kv and swa share one shape:
    pages + the pool's own page_size (tokens = pages x page_size). gpus: the engine's GPU as
    [{index, name, uuid, total_bytes}] (the primary rank's; a list so TP can extend it), []
    until the readiness meta arrives."""
    tr: StatsTracker = state.stats
    config = state.config
    ready_at = getattr(state, "ready_at", None)
    uptime_s = max(0, int(time.monotonic() - ready_at)) if ready_at is not None else 0
    kv = (
        {"used_pages": tr.kv_used_pages, "total_pages": tr.kv_total_pages,
         "page_size": _resolved_page_size(state, config)}
        if tr.kv_total_pages > 0 else None
    )
    mamba = (
        {"used_slots": tr.mamba_used_slots, "total_slots": tr.mamba_total_slots}
        if tr.mamba_total_slots > 0 else None
    )
    sps = _swa_page_size(config)
    try:
        model_max_seq_len = int(getattr(config, "max_seq_len", 0) or 0)
    except (TypeError, ValueError):
        model_max_seq_len = 0
    # Same expression the scheduler's admission check uses; fall back to the model ceiling
    # while the readiness meta is still in flight.
    try:
        effective_max_seq_len = int(getattr(state, "max_seq_len", 0) or 0)
    except (TypeError, ValueError):
        effective_max_seq_len = 0
    if effective_max_seq_len <= 0:
        effective_max_seq_len = model_max_seq_len
    swa = (
        {"used_pages": tr.swa_used_tokens // sps, "total_pages": tr.swa_total_tokens // sps,
         "page_size": sps}
        if tr.swa_total_tokens > 0 else None
    )
    model_card = derive_model_card(config)
    # ``model.ctx`` must be the limit the scheduler ENFORCES, not the checkpoint ceiling:
    # ``launch._stats_context_length`` reads exactly this field as its fallback when sizing a
    # client's context window, so leaving the raw ceiling here would still let a client send
    # prompts the scheduler rejects whenever the KV pool is smaller than max_position. The
    # raw ceiling stays available in ``limits.model_max_seq_len``.
    model_card["ctx"] = effective_max_seq_len
    # Multimodal section: only for a process that actually accepts images. image_tokens.min/max
    # are the per-image token budget; None means "whatever the checkpoint's processor defaults
    # to" (for this checkpoint that is 64..16384 tokens), NOT zero. encoder_cache is the live
    # encoder embedding cache sampled by the worker at ~1/s.
    served_modalities = tuple(getattr(config, "served_modalities", None) or ())
    mm_config = getattr(config, "mm", None)
    mm = (
        {
            "served_modalities": sorted(served_modalities),
            "image_tokens": {
                "min": getattr(mm_config, "image_min_tokens", None),
                "max": getattr(mm_config, "image_max_tokens", None),
            },
            "embed_cache_device": getattr(mm_config, "embed_cache_device", None),
            "encoder_cache": tr.mm_stats,
        }
        if "image" in served_modalities
        else None
    )
    return {
        "instance_id": getattr(state, "instance_id", None),
        "model": model_card,
        "uptime_s": uptime_s,
        "kv": kv,
        "mamba": mamba,
        "swa": swa,
        # What the scheduler actually ADMITS. prompt_tokens must stay under max_seq_len (the
        # output budget is clamped to the remainder), and it is min(model max_position, KV
        # pool tokens) -- so it can be smaller than the model's own ceiling when the KV pool
        # was configured below it. model_max_seq_len is that ceiling, for reference.
        "limits": {
            "max_seq_len": effective_max_seq_len,
            "model_max_seq_len": model_max_seq_len,
        },
        "vram_bytes": tr.vram_bytes,
        "gpus": list(getattr(state, "gpus", None) or []),
        "throughput": {
            "decode_tps": round(tr.decode_tps(), 1),
            "prefill_tps": round(tr.prefill_tps(), 1),
        },
        "requests": {
            "active": tr.active,
            "completed": tr.completed,
            "p95_ms": p95_ms,
            "ttft_mean_ms": ttft_mean_ms,
            "prompt_tokens_total": tr.prompt_tokens_total,
            "completion_tokens_total": tr.completion_tokens_total,
            "cached_prompt_tokens_total": tr.cached_prompt_tokens_total,
            "decode_tokens_total": tr.decode_tokens_total,
            "prefill_seconds_total": round(tr.prefill_seconds_total, 6),
            "decode_seconds_total": round(tr.decode_seconds_total, 6),
        },
        "prefix_cache": {
            "cached_tokens_total": tr.cached_tokens_total,
            "prompt_tokens_total": tr.prompt_tokens_total,
            "hit_ratio": (
                round(tr.cached_tokens_total / tr.prompt_tokens_total, 4)
                if tr.prompt_tokens_total
                else 0.0
            ),
        },
        "moe": tr.moe_stats,
        "mm": mm,
        # The host KV tier is a prefix-cache extension, so it sits beside that section rather
        # than under moe/mm. None when the deployment did not enable it.
        "kv_host_tier": tr.host_tier_stats,
    }
