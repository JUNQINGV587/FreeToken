#!/usr/bin/env python3
"""Small same-origin browser playground for a running FreeToken server.

The browser only talks to this process.  Generation and read-only telemetry are
proxied to a fixed loopback upstream, so the FreeToken API does not need broad
CORS permissions and the UI cannot be turned into a generic HTTP proxy.
"""

from __future__ import annotations

import argparse
import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse


ROOT = Path(__file__).resolve().parent
INDEX_HTML = ROOT / "index.html"
MAX_REQUEST_BYTES = 8 * 1024 * 1024
UPSTREAM = "http://127.0.0.1:1919"


def _upstream_url(path: str, query: str = "") -> str:
    url = f"{UPSTREAM}{path}"
    return f"{url}?{query}" if query else url


@asynccontextmanager
async def lifespan(app: FastAPI):
    timeout = httpx.Timeout(connect=5.0, read=None, write=30.0, pool=5.0)
    app.state.client = httpx.AsyncClient(timeout=timeout, follow_redirects=False)
    try:
        yield
    finally:
        await app.state.client.aclose()


app = FastAPI(
    title="FreeToken Playground",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
    lifespan=lifespan,
)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; img-src 'self' data: blob:; "
        "style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; "
        "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
    )
    return response


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(INDEX_HTML, media_type="text/html; charset=utf-8")


@app.get("/favicon.ico")
async def favicon() -> Response:
    return Response(status_code=204)


@app.get("/robots.txt")
async def robots() -> Response:
    return Response("User-agent: *\nDisallow: /\n", media_type="text/plain")


async def _proxy_get(request: Request, path: str) -> Response:
    client: httpx.AsyncClient = request.app.state.client
    try:
        upstream = await client.get(
            _upstream_url(path, request.url.query),
            headers={"Accept": "application/json"},
        )
    except httpx.HTTPError as exc:
        return JSONResponse(
            {"error": {"message": f"FreeToken upstream unavailable: {exc}"}},
            status_code=502,
        )
    content_type = upstream.headers.get("content-type", "application/json")
    return Response(upstream.content, status_code=upstream.status_code, media_type=content_type)


@app.get("/api/health")
async def api_health(request: Request) -> Response:
    return await _proxy_get(request, "/health")


@app.get("/api/v1/models")
async def api_models(request: Request) -> Response:
    return await _proxy_get(request, "/v1/models")


@app.get("/api/v1/stats")
async def api_stats(request: Request) -> Response:
    return await _proxy_get(request, "/v1/stats")


@app.get("/api/v1/requests")
async def api_requests(request: Request) -> Response:
    return await _proxy_get(request, "/v1/requests")


@app.get("/api/v1/cache/status")
async def api_cache_status(request: Request) -> Response:
    return await _proxy_get(request, "/v1/cache/status")


async def _stream_response(response: httpx.Response) -> AsyncIterator[bytes]:
    try:
        async for chunk in response.aiter_raw():
            yield chunk
    finally:
        await response.aclose()


@app.post("/api/v1/chat/completions")
async def api_chat(request: Request) -> Response:
    body = await request.body()
    if len(body) > MAX_REQUEST_BYTES:
        return JSONResponse(
            {"error": {"message": "request body exceeds the 8 MiB playground limit"}},
            status_code=413,
        )

    client: httpx.AsyncClient = request.app.state.client
    upstream_request = client.build_request(
        "POST",
        _upstream_url("/v1/chat/completions"),
        content=body,
        headers={
            "Content-Type": request.headers.get("content-type", "application/json"),
            "Accept": "text/event-stream, application/json",
        },
    )
    try:
        upstream = await client.send(upstream_request, stream=True)
    except httpx.HTTPError as exc:
        return JSONResponse(
            {"error": {"message": f"FreeToken upstream unavailable: {exc}"}},
            status_code=502,
        )

    content_type = upstream.headers.get("content-type", "application/octet-stream")
    if upstream.status_code >= 400 or "text/event-stream" not in content_type:
        content = await upstream.aread()
        await upstream.aclose()
        return Response(content, status_code=upstream.status_code, media_type=content_type)

    return StreamingResponse(
        _stream_response(upstream),
        status_code=upstream.status_code,
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


def _read_meminfo() -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, raw = line.split(":", 1)
            values[key] = int(raw.strip().split()[0]) * 1024
    except (OSError, ValueError, IndexError):
        return {"total_bytes": 0, "used_bytes": 0, "available_bytes": 0}
    total = values.get("MemTotal", 0)
    available = values.get("MemAvailable", 0)
    return {
        "total_bytes": total,
        "used_bytes": max(0, total - available),
        "available_bytes": available,
    }


def _gpu_snapshot() -> list[dict[str, int | str | float | None]]:
    try:
        import pynvml

        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
    except Exception:
        return []

    cards: list[dict[str, int | str | float | None]] = []
    for index in range(count):
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8", "replace")
            try:
                temperature: int | None = pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                )
            except Exception:
                temperature = None
            try:
                power_w: float | None = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            except Exception:
                power_w = None
            cards.append(
                {
                    "index": index,
                    "name": str(name),
                    "total_bytes": int(memory.total),
                    "used_bytes": int(memory.used),
                    "free_bytes": int(memory.free),
                    "utilization_percent": int(utilization.gpu),
                    "temperature_c": temperature,
                    "power_w": round(power_w, 1) if power_w is not None else None,
                }
            )
        except Exception:
            continue
    return cards


@app.get("/api/ui/system")
async def ui_system() -> dict:
    memory, gpus = await asyncio.gather(
        asyncio.to_thread(_read_meminfo), asyncio.to_thread(_gpu_snapshot)
    )
    return {"memory": memory, "gpus": gpus}


def main() -> None:
    parser = argparse.ArgumentParser(description="FreeToken browser playground")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30002)
    parser.add_argument("--upstream", default="http://127.0.0.1:1919")
    args = parser.parse_args()

    global UPSTREAM
    UPSTREAM = args.upstream.rstrip("/")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info", access_log=False)


if __name__ == "__main__":
    main()
