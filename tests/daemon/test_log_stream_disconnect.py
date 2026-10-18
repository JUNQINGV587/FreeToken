"""The daemon SSE log stream must notice a gone client on its own.

``Request.is_disconnected()`` is unusable on uvicorn 0.54 (see
freetoken.server.disconnect), so the stream would otherwise only learn about a
disconnect when the next heartbeat write fails — up to 15 s late, and never for
a stream that produces no frames. These tests pin the direct-probe behaviour.
"""

from __future__ import annotations

import asyncio

from freetoken.daemon.app import _log_stream


class _Ring:
    """Minimal stand-in for the daemon's log ring."""

    def __init__(self, backlog=None):
        self._backlog = list(backlog or [])
        self.subs = []
        self.unsubs = []

    def subscribe(self, push):
        self.subs.append(push)

    def since(self, since):
        cursor = self._backlog[-1]["seq"] + 1 if self._backlog else since
        return list(self._backlog), cursor

    def unsubscribe(self, push):
        self.unsubs.append(push)


class _GoneRequest:
    """A request whose transport reports http.disconnect on every probe."""

    headers: dict = {}

    def __init__(self):
        self.probes = 0

    async def _receive(self):
        self.probes += 1
        return {"type": "http.disconnect"}


async def _drain(response):
    frames = []
    async for chunk in response.body_iterator:
        frames.append(chunk)
    return frames


def test_sse_stops_immediately_when_client_already_gone():
    async def run():
        req = _GoneRequest()
        ring = _Ring()
        frames = await asyncio.wait_for(_drain(_log_stream(req, ring, 0)), timeout=2)
        assert frames == []
        assert req.probes >= 1
        assert ring.unsubs == ring.subs  # subscriber always removed

    asyncio.run(run())


def test_sse_replays_backlog_then_stops_when_client_goes():
    async def run():
        req = _GoneRequest()
        ring = _Ring([{"seq": 7, "level": "INFO", "msg": "hello"}])
        frames = await asyncio.wait_for(_drain(_log_stream(req, ring, 7)), timeout=2)
        assert len(frames) == 1 and "hello" in frames[0]
        assert ring.unsubs == ring.subs

    asyncio.run(run())
