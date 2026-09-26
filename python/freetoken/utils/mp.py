from __future__ import annotations

import os
import sys
import time
from typing import Any, Callable, Dict, Generic, TypeVar

import msgpack
import zmq
import zmq.asyncio

T = TypeVar("T")

# A 262144-token prompt with logprobs packs to ~107 MiB (measured ~407 B/token) and
# top_logprobs multiplies that, so the msgpack default (100 MiB) sits below the legal
# worst case; env-overridable for tighter deployments.
_DEFAULT_MAX_BUFFER = 1 << 30


def _max_buffer_size() -> int:
    raw = os.environ.get("FREETOKEN_MSGPACK_MAX_BUFFER")
    if raw is None:
        return _DEFAULT_MAX_BUFFER
    try:
        value = int(raw)
    except ValueError:
        return _DEFAULT_MAX_BUFFER
    return value if value > 0 else _DEFAULT_MAX_BUFFER


class _CoalescedUnpacker:
    """
    Buffered msgpack decoder tolerant of coalesced frames.

    A single ZMQ frame should carry exactly one msgpack object, but on some platforms
    (issue #452: Windows, offload MoE) a read can surface a frame containing two packed
    objects back to back. ``msgpack.unpackb`` raises ``ExtraData`` in that case and kills
    the worker process. Feeding every received frame through one ``Unpacker`` instead
    decodes the first object and buffers the remainder, so the next ``get()`` returns it
    in order instead of crashing.
    """

    def __init__(self, max_buffer_size: int | None = None) -> None:
        self._max_buffer_size = max_buffer_size if max_buffer_size is not None else _max_buffer_size()
        self._unpacker = self._new_unpacker()
        self._pending: list[Any] = []
        self.dropped_frames = 0

    def _new_unpacker(self) -> msgpack.Unpacker:
        return msgpack.Unpacker(raw=False, max_buffer_size=self._max_buffer_size)

    def feed(self, frame: bytes) -> None:
        """Queue up every msgpack object contained in ``frame`` (usually exactly one)."""
        try:
            self._unpacker.feed(frame)
            for obj in self._unpacker:
                self._pending.append(obj)
        except msgpack.exceptions.BufferFull:
            # One oversized message must not kill the worker: drop that frame and keep
            # serving later ones; the owning request times out instead of the process.
            self.dropped_frames += 1
            print(
                f"freetoken.mp: dropped a {len(frame)}-byte frame above the "
                f"{self._max_buffer_size}-byte msgpack buffer (FREETOKEN_MSGPACK_MAX_BUFFER)",
                file=sys.stderr,
            )
            self._unpacker = self._new_unpacker()

    def take(self) -> Any:
        """Pop the oldest undelivered object; raises StopIteration when the buffer is empty."""
        return self._pending.pop(0)

    def __len__(self) -> int:
        return len(self._pending)


class ZmqPushQueue(Generic[T]):
    def __init__(
        self,
        addr: str,
        create: bool,
        encoder: Callable[[T], Dict],
    ):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUSH)
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.encoder = encoder

    def put(self, obj: T):
        event = msgpack.packb(self.encoder(obj), use_bin_type=True)
        self.socket.send(event, copy=False)

    def stop(self):
        self.socket.close()
        self.context.term()


class ZmqAsyncPushQueue(Generic[T]):
    def __init__(
        self,
        addr: str,
        create: bool,
        encoder: Callable[[T], Dict],
    ):
        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.PUSH)
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.encoder = encoder

    async def put(self, obj: T):
        event = msgpack.packb(self.encoder(obj), use_bin_type=True)
        await self.socket.send(event, copy=False)

    def stop(self):
        self.socket.close()
        self.context.term()


class ZmqPullQueue(Generic[T]):
    def __init__(
        self,
        addr: str,
        create: bool,
        decoder: Callable[[Dict], T],
    ):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PULL)
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.decoder = decoder
        self._unpacker = _CoalescedUnpacker()

    def get(self) -> T:
        if not len(self._unpacker):
            event = self.socket.recv()
            self._unpacker.feed(event)
        obj = self._unpacker.take()
        return self.decoder(obj)

    def get_raw(self) -> bytes:
        # Raw path (multi-rank broadcast) must stay unbuffered: callers pair empty()/get_raw()
        # with a rank-wide count, so a buffered remainder would desynchronize the loop.
        if len(self._unpacker):
            raise RuntimeError(
                "get_raw() cannot be mixed with buffered get() on the same queue: "
                "the unpacker holds undelivered objects that would be skipped."
            )
        return self.socket.recv()

    def decode(self, raw: bytes) -> T:
        return self.decoder(msgpack.unpackb(raw, raw=False))

    def empty(self) -> bool:
        return len(self._unpacker) == 0 and self.socket.poll(timeout=0) == 0

    def stop(self):
        self.socket.close()
        self.context.term()


class ZmqAsyncPullQueue(Generic[T]):
    def __init__(
        self,
        addr: str,
        create: bool,
        decoder: Callable[[Dict], T],
    ):
        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.PULL)
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.decoder = decoder
        self._unpacker = _CoalescedUnpacker()

    async def get(self) -> T:
        if not len(self._unpacker):
            event = await self.socket.recv()
            self._unpacker.feed(event)
        obj = self._unpacker.take()
        return self.decoder(obj)

    def stop(self):
        self.socket.close()
        self.context.term()


class ZmqPubQueue(Generic[T]):
    def __init__(
        self,
        addr: str,
        create: bool,
        encoder: Callable[[T], Dict],
    ):
        self.context = zmq.Context()
        # XPUB sends like a PUB and surfaces inbound subscription events, so
        # wait_for_subscribers can close the slow-joiner gap
        self.socket = self.context.socket(zmq.XPUB)
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.encoder = encoder

    def wait_for_subscribers(self, count: int, timeout_s: float = 30.0) -> None:
        """Block until ``count`` subscribers have registered.

        A SUB's subscribe only reaches the publisher asynchronously; anything
        broadcast before it lands is silently dropped.
        """
        if count <= 0:
            return
        deadline = time.monotonic() + timeout_s
        seen = 0
        while seen < count:
            remaining_ms = int(max(0, deadline - time.monotonic()) * 1000)
            if not self.socket.poll(remaining_ms):
                raise TimeoutError(f"only {seen}/{count} subscribers registered within {timeout_s}s")
            event = self.socket.recv(copy=False)
            if event.bytes and event.bytes[0] == 1:
                seen += 1

    def put_raw(self, raw: bytes):
        self.socket.send(raw, copy=False)

    def put(self, obj: T):
        event = msgpack.packb(self.encoder(obj), use_bin_type=True)
        self.socket.send(event, copy=False)

    def stop(self):
        self.socket.close()
        self.context.term()


class ZmqSubQueue(Generic[T]):
    def __init__(
        self,
        addr: str,
        create: bool,
        decoder: Callable[[Dict], T],
    ):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.decoder = decoder
        self._unpacker = _CoalescedUnpacker()

    def get(self) -> T:
        if not len(self._unpacker):
            event = self.socket.recv()
            self._unpacker.feed(event)
        obj = self._unpacker.take()
        return self.decoder(obj)

    def empty(self) -> bool:
        return len(self._unpacker) == 0 and self.socket.poll(timeout=0) == 0

    def stop(self):
        self.socket.close()
        self.context.term()