"""Tests for the non-streaming disconnect watchdog.

`Request.is_disconnected()` cannot be used here: it runs receive() inside an
already-cancelled anyio scope, so the cancellation lands on receive()'s first
await and a disconnect that is not queued yet is never observed (uvicorn 0.54
reports a live peer forever). The watchdog probes receive() directly with a
short timeout instead.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from freetoken.server import openai_api
from freetoken.server.openai_api import _await_watching_disconnect, _client_is_gone


class _ScriptedRequest:
    """Minimal Starlette-Request stand-in with a scripted receive channel."""

    def __init__(self, *, silent_probes=0, then_disconnect=False):
        self.silent_probes = silent_probes
        self.then_disconnect = then_disconnect
        self.receive_calls = 0

    async def _receive(self):
        self.receive_calls += 1
        if self.receive_calls > self.silent_probes and self.then_disconnect:
            return {"type": "http.disconnect"}
        await asyncio.sleep(3600)

    async def is_disconnected(self):  # the broken probe must not be trusted
        return False


async def _abort_recorder():
    aborts: list[int] = []

    async def abort_user(uid):
        aborts.append(uid)

    return SimpleNamespace(abort_user=abort_user, aborts=aborts)


def test_probe_sees_queued_disconnect():
    async def run():
        req = _ScriptedRequest(silent_probes=0, then_disconnect=True)
        assert await _client_is_gone(req) is True

    asyncio.run(run())


def test_probe_reports_connected_while_receive_blocks():
    async def run():
        req = _ScriptedRequest(silent_probes=100)
        assert await _client_is_gone(req) is False

    asyncio.run(run())


def test_probe_reports_connected_for_plain_request_messages():
    class _RequestMessage(_ScriptedRequest):
        async def _receive(self):
            return {"type": "http.request", "body": b"", "more_body": False}

    async def run():
        assert await _client_is_gone(_RequestMessage()) is False

    asyncio.run(run())


def test_watchdog_aborts_when_receive_returns_disconnect(monkeypatch):
    monkeypatch.setattr(openai_api, "_DISCONNECT_POLL_SECONDS", 0.05)

    async def run():
        state = await _abort_recorder()
        started = asyncio.Event()

        async def generation():
            started.set()
            await asyncio.sleep(3600)

        req = _ScriptedRequest(silent_probes=1, then_disconnect=True)
        result = await _await_watching_disconnect(generation(), req, state, 5)
        assert started.is_set()
        assert result is None
        assert state.aborts == [5]

    asyncio.run(run())


def test_watchdog_returns_result_when_generation_finishes(monkeypatch):
    monkeypatch.setattr(openai_api, "_DISCONNECT_POLL_SECONDS", 0.05)

    async def run():
        state = await _abort_recorder()

        async def generation():
            await asyncio.sleep(0.01)
            return "done"

        req = _ScriptedRequest(silent_probes=100)
        assert await _await_watching_disconnect(generation(), req, state, 9) == "done"
        assert state.aborts == []

    asyncio.run(run())
