"""Read-only control-plane endpoints consumed by the desktop app: /health (lifecycle),
/ready (readiness), /v1/stats (runtime metrics, Task 6), /v1/requests (request log ring, Task 5).

All handlers read a shared FrontendManager snapshot via ``get_state``; nothing here touches
the scheduler or blocks. Registered on the app alongside the OpenAI/Anthropic/Responses routes.
"""

from __future__ import annotations

import time
from typing import Any, Callable

from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse


def build_health(state: Any, version: str) -> dict:
    """Full-lifecycle health doc: loading -> ok -> error."""
    instance_id = getattr(state, "instance_id", None)
    fatal = getattr(state, "fatal_error", None)
    if fatal:
        return {"status": "error", "message": fatal, "instance_id": instance_id}

    mstate = getattr(state, "maintenance_state", "serving")
    config = getattr(state, "config", None)
    model = getattr(config, "served_model_name", None)

    if mstate == "loading":
        lp = getattr(state, "load_progress", None)
        return {
            "status": "loading",
            "phase": lp.phase if lp is not None else "other",
            "progress": {
                "done_bytes": lp.done_bytes if lp is not None else 0,
                "total_bytes": lp.total_bytes if lp is not None else 0,
            },
            "model": model,
            "instance_id": instance_id,
        }

    ready_at = getattr(state, "ready_at", None)
    uptime_s = max(0, int(time.monotonic() - ready_at)) if ready_at is not None else 0
    return {
        "status": "ok",
        "model": model,
        "instance_id": instance_id,
        "uptime_s": uptime_s,
        "maintenance": mstate,
        "version": version,
    }


def is_ready(doc: dict) -> bool:
    """Whether a ``build_health`` document describes an engine that can take work.

    Readiness is not liveness: ``/health`` answers 200 for the whole lifecycle so the client can
    render load progress, while ``/ready`` must fail closed -- ``loading``, ``failed`` and
    ``stopping`` all mean a caller has to keep waiting (or give up), and only ``serving`` counts.
    """
    return doc.get("status") == "ok" and doc.get("maintenance", "serving") == "serving"


def register_control_routes(
    app: FastAPI,
    get_state: Callable[[], Any],
    get_model_sampling: Callable[[], dict] | None = None,
) -> None:
    @app.get("/health")
    async def health():
        return build_health(get_state(), app.version)

    @app.get("/ready")
    async def ready(response: Response):
        doc = build_health(get_state(), app.version)
        if not is_ready(doc):
            response.status_code = 503
        return doc

    # The k8s-style probe split. /health and /ready stay exactly as they are (the desktop
    # reads /health for load progress and ops gates on /ready), and readiness here goes
    # through the same is_ready(doc) predicate instead of a second copy of the state checks.
    @app.get("/healthz")
    async def healthz():
        # Liveness is deliberately independent of engine startup/rebuild/failure.
        # Restarting an HTTP process just because weights are loading only repeats
        # that loading forever. Readiness below carries the admission signal.
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz():
        # Same lifecycle/phase/progress document as /health: the status code, not a
        # replacement body, tells an orchestrator when to route.
        doc = build_health(get_state(), app.version)
        return JSONResponse(doc, status_code=200 if is_ready(doc) else 503)

    from . import request_ring

    @app.get("/v1/requests")
    async def list_requests(since: int = 0, limit: int = 100):
        limit = max(1, min(limit, 512))
        entries, next_cursor = request_ring.requests_since(since, limit)
        return {"entries": entries, "next_cursor": next_cursor}

    from .stats import build_stats

    @app.get("/v1/stats")
    async def stats():
        doc = build_stats(
            get_state(), request_ring.requests_p95_ms(), request_ring.requests_ttft_mean_ms()
        )
        # Surface the model's recommended sampling (from its generation_config.json / GGUF
        # metadata) so clients can seed their sampling controls per-model instead of guessing.
        if get_model_sampling is not None:
            doc["model"]["sampling"] = get_model_sampling() or {}
        return doc
