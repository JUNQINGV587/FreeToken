# SPDX-License-Identifier: Apache-2.0
"""Regression tests for coalesced msgpack frames on ZMQ queues (issue #452).

A single ZMQ frame should carry exactly one msgpack object, but a read can surface
a frame containing two packed objects back to back. ``msgpack.unpackb`` raises
``ExtraData`` there and the worker process dies. The buffered unpacker decodes the
first object and holds the remainder for the next get().
"""

import asyncio

import msgpack
import pytest

from freetoken.utils.mp import (
    ZmqAsyncPullQueue,
    ZmqAsyncPushQueue,
    ZmqPubQueue,
    ZmqPullQueue,
    ZmqPushQueue,
    ZmqSubQueue,
)


def _identity_decoder(obj):
    return obj


def _addr(port):
    return f"tcp://127.0.0.1:{port}"


class TestCoalescedFrames:
    """A frame carrying two packed objects must yield two get() calls, not a crash."""

    def test_sync_pull_coalesced_frame(self):
        push = ZmqPushQueue(_addr(5601), create=True, encoder=lambda o: o)
        pull = ZmqPullQueue(_addr(5601), create=False, decoder=_identity_decoder)
        try:
            # Simulate the coalesced frame: two packb outputs concatenated.
            frame = msgpack.packb({"i": 1}, use_bin_type=True) + msgpack.packb({"i": 2}, use_bin_type=True)
            pull._unpacker.feed(frame)

            assert pull.get() == {"i": 1}
            assert pull.get() == {"i": 2}
            assert len(pull._unpacker) == 0
        finally:
            push.stop()
            pull.stop()

    def test_decode_coalesced_raw(self):
        from freetoken.utils.mp import _CoalescedUnpacker

        unpacker = _CoalescedUnpacker()
        frame = msgpack.packb("first", use_bin_type=True) + msgpack.packb("second", use_bin_type=True)
        unpacker.feed(frame)
        assert unpacker.take() == "first"
        assert unpacker.take() == "second"

    def test_empty_still_true_with_buffered_only(self):
        push = ZmqPushQueue(_addr(5602), create=True, encoder=lambda o: o)
        pull = ZmqPullQueue(_addr(5602), create=False, decoder=_identity_decoder)
        try:
            # nothing on socket and no buffer -> empty
            assert pull.empty()
            # a coalesced frame arrives and is fully buffered -> socket empty but buffer has 1 left
            push.put({"x": 1})
            pull._unpacker.feed(pull.get_raw())
            assert pull.get() == {"x": 1}
            assert pull.empty()
        finally:
            push.stop()
            pull.stop()

    def test_sub_queue_coalesced(self):
        import msgpack

        pub = ZmqPubQueue(_addr(5603), create=True, encoder=lambda o: o)
        sub = ZmqSubQueue(_addr(5603), create=False, decoder=_identity_decoder)
        try:
            import time

            time.sleep(0.1)  # SUB subscription propagation
            pub.put({"a": 1})
            pub.put({"a": 2})
            time.sleep(0.1)
            # Deliver both objects even if they arrive in one frame.
            assert sub.get() == {"a": 1}
            assert sub.get() == {"a": 2}
        finally:
            pub.stop()
            sub.stop()


def test_async_pull_coalesced():
    """The async queue goes through the same buffered unpacker.

    Written as a sync test that drives the coroutine with ``asyncio.run``: a bare
    ``async def`` test needs pytest-asyncio (plus ``asyncio_mode = auto``), and this
    repo declares only ``pytest`` in its dev extras, so the plugin form would fail
    collection here rather than exercise the queue.
    """
    import sys

    # zmq.asyncio needs a Selector loop on Windows; the project runs on Linux CI.
    if sys.platform == "win32":
        pytest.skip("zmq.asyncio requires SelectorEventLoop on Windows (prod runs on Linux)")

    async def _exercise() -> None:
        push = ZmqAsyncPushQueue(_addr(5604), create=True, encoder=lambda o: o)
        pull = ZmqAsyncPullQueue(_addr(5604), create=False, decoder=_identity_decoder)
        try:
            await push.put({"n": 1})
            await push.put({"n": 2})
            await asyncio.sleep(0.1)

            # Even if both messages arrive in a single frame, two gets must succeed.
            assert await pull.get() == {"n": 1}
            assert await pull.get() == {"n": 2}
        finally:
            push.stop()
            pull.stop()

    asyncio.run(_exercise())

class TestOversizedFrames:
    """A frame above the msgpack buffer cap is dropped, not fatal (2026-09-26 incident:
    a 258k-token prompt with logprobs packed past the 100 MiB msgpack default and the
    uncaught BufferFull killed the tokenizer worker)."""

    def test_default_cap_is_one_gib(self):
        from freetoken.utils.mp import _CoalescedUnpacker, _DEFAULT_MAX_BUFFER

        assert _DEFAULT_MAX_BUFFER == 1 << 30
        assert _CoalescedUnpacker()._max_buffer_size == _DEFAULT_MAX_BUFFER

    def test_env_override_and_invalid_fallback(self, monkeypatch):
        from freetoken.utils.mp import _CoalescedUnpacker

        monkeypatch.setenv("FREETOKEN_MSGPACK_MAX_BUFFER", str(64 << 20))
        assert _CoalescedUnpacker()._max_buffer_size == 64 << 20
        monkeypatch.setenv("FREETOKEN_MSGPACK_MAX_BUFFER", "not-a-number")
        assert _CoalescedUnpacker()._max_buffer_size == 1 << 30
        monkeypatch.setenv("FREETOKEN_MSGPACK_MAX_BUFFER", "0")
        assert _CoalescedUnpacker()._max_buffer_size == 1 << 30

    def test_oversized_frame_is_dropped_and_service_continues(self, capsys):
        from freetoken.utils.mp import _CoalescedUnpacker

        unpacker = _CoalescedUnpacker(max_buffer_size=1 << 20)
        big = msgpack.packb({"blob": b"x" * (2 << 20)}, use_bin_type=True)
        unpacker.feed(big)  # must not raise
        assert unpacker.dropped_frames == 1
        assert "dropped" in capsys.readouterr().err
        # Later frames still decode: the worker survives the oversized request.
        unpacker.feed(msgpack.packb({"i": 1}, use_bin_type=True))
        assert unpacker.take() == {"i": 1}

    def test_pending_objects_survive_a_later_oversized_frame(self):
        from freetoken.utils.mp import _CoalescedUnpacker

        unpacker = _CoalescedUnpacker(max_buffer_size=1 << 20)
        unpacker.feed(msgpack.packb("kept", use_bin_type=True))
        unpacker.feed(msgpack.packb({"blob": b"y" * (2 << 20)}, use_bin_type=True))
        assert unpacker.dropped_frames == 1
        assert unpacker.take() == "kept"
