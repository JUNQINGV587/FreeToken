"""Transport-level client-disconnect detection shared by the HTTP front ends.

Starlette's ``Request.is_disconnected()`` cannot be trusted on uvicorn 0.54: it
runs ``receive()`` inside an anyio scope that is already cancelled, so the
cancellation lands on the first ``await`` inside ``RequestResponseCycle.receive``
(which is parked on ``message_event``) and a queued ``http.disconnect`` message
is never observed — the call reports a live peer forever. Probing ``receive()``
directly with a short timeout returns that queued message instead.

Both the non-streaming chat/completions watchdog and the daemon's SSE log
stream need this, so it lives here rather than in either handler module.
"""

from __future__ import annotations

import asyncio

from starlette.requests import Request

# How long a single transport probe may block before the peer counts as
# connected. Short enough to run once per second (or per SSE loop iteration)
# without measurable overhead, long enough for a queued http.disconnect to be
# returned immediately.
DISCONNECT_PROBE_SECONDS = 0.05


async def client_is_gone(request: Request) -> bool:
    """True once the peer closed the connection, without blocking the caller.

    Returns False when the probe times out (peer still connected), when the
    request object has no receive channel (non-Starlette doubles), or when the
    channel yields an ordinary ``http.request`` message.
    """
    receive = getattr(request, "_receive", None)
    if receive is None:  # pragma: no cover - only for non-Starlette doubles
        return await request.is_disconnected()
    try:
        message = await asyncio.wait_for(receive(), timeout=DISCONNECT_PROBE_SECONDS)
    except asyncio.TimeoutError:
        return False
    return isinstance(message, dict) and message.get("type") == "http.disconnect"
