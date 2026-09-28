"""Unit tests for the disk-tier graph-doorbell fetch (moe/graph_fetch.py).

These run on GPU: the request/spin/install kernels are what get recorded into
the decode CUDA graphs, so they are exercised directly here rather than only
through the e2e consistency gate. Uses a tiny fake tier/banks so no model
weights are needed.

DMA transport: the request block lives in DEVICE memory and reaches the host
through the captured D2H memcpy (stage_fetch runs eagerly here, so the copy
just executes); the ack and staging reach the device through the service
thread's H2D copies on its private stream.
"""
from __future__ import annotations

import os
import threading
import time
from types import SimpleNamespace

import pytest
import torch

from freetoken.moe.graph_fetch import GraphFetchBridge, _gf_request_kernel

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

ROW = 4096         # bytes per expert row (keeps fast_index_copy alignment happy)
N_SLOTS = 64
N_LAYERS = 2
PLAN = 16
K_MAX = 8


def _make_disk_file(tmp_path, n_rows=16):
    """Row r filled with byte value r, so staging contents are identifiable."""
    path = tmp_path / "fake_disk.bin"
    buf = bytearray(n_rows * ROW)
    for r in range(n_rows):
        buf[r * ROW:(r + 1) * ROW] = bytes([r % 256]) * ROW
    path.write_bytes(bytes(buf))
    return str(path)


class _FakeTier:
    """Minimal stand-in for DiskTier's internals used by GraphFetchBridge."""

    def __init__(self, tmp_path, ram: int):
        self._ram = ram
        self.ownership = None
        self._preadv_calls = 0
        self._index = SimpleNamespace(scalar_banks=frozenset())
        self._path = _make_disk_file(tmp_path)
        self._staging = threading.local()
        self._staging_size = 4 * ROW  # bridge bounce buffer size
        # cache.banks entries are (per-layer [host tensors], gpu_cache) pairs
        host_layers = [torch.zeros(16, ROW, dtype=torch.uint8) for _ in range(N_LAYERS)]
        gpu = torch.zeros(N_SLOTS, ROW, dtype=torch.uint8, device="cuda")
        self._banks = [(host_layers, gpu)]

    def _staging_ring(self):
        ring = getattr(self._staging, "ring", None)
        if ring is None:
            buf = torch.zeros(4 * ROW, dtype=torch.uint8)
            ring = [(SimpleNamespace(tensor=buf, addr=buf.data_ptr()), None)]
            self._staging.ring = ring
        return ring

    def _fd(self, shard_idx):
        return os.open(self._path, os.O_RDONLY), False

    def _group_runs(self, bank_idx, layer, expert):
        a0 = expert * ROW
        # (shard_idx, a0, a1, members[(d0, d1, off, nbytes)], end)
        yield 0, a0, a0 + ROW, [(0, ROW, a0, ROW)], a0 + ROW

    def _fill_scalar_row(self, *a, **k):  # pragma: no cover - no scalar banks
        raise AssertionError("unexpected scalar bank")


class _FakeCache:
    def __init__(self, tier):
        self.banks = tier._banks
        self.num_indices = torch.zeros(1, dtype=torch.int64, device="cuda")
        self.evict_slots = torch.zeros(PLAN, dtype=torch.int32, device="cuda")
        self.src_indices = torch.zeros(PLAN, dtype=torch.int32, device="cuda")


def _set_plan(cache, srcs, slots):
    n = len(srcs)
    cache.num_indices[0] = n
    cache.evict_slots[:n] = torch.tensor(slots, dtype=torch.int32)
    cache.src_indices[:n] = torch.tensor(srcs, dtype=torch.int32)


def _launch_request(bridge, cache, layer=0):
    """Run one full stage_fetch eagerly: request kernel + D2H memcpy + spin
    (doorbell off unless enable() ran) + install."""
    bridge.stage_fetch(cache, layer)
    torch.cuda.synchronize()


@pytest.fixture()
def bridge(tmp_path):
    tier = _FakeTier(tmp_path, ram=3)
    cache = _FakeCache(tier)
    b = GraphFetchBridge(tier, cache, k_max=K_MAX)  # __init__ compile-warms
    yield b
    b.shutdown()


def test_request_kernel_splits_ram_and_disk(bridge):
    cache = _FakeCache(bridge.tier)
    _set_plan(cache, srcs=[0, 5, 2, 7], slots=[10, 11, 12, 13])
    _launch_request(bridge, cache)
    # RAM rows compacted in place, in order.
    assert int(cache.num_indices.item()) == 2
    assert cache.evict_slots[:2].tolist() == [10, 12]
    assert cache.src_indices[:2].tolist() == [0, 2]
    # Disk rows recorded in the request block (local ids, order kept); the D2H
    # memcpy delivered it to the pinned mirror.
    blk = bridge.req_host
    assert int(blk[0]) == 2                       # count
    assert int(blk[1]) == 0                       # layer
    assert blk[2:4].tolist() == [5, 7]            # rows
    assert int(blk[bridge.seq_off]) == 1          # seq (end of block)
    assert bridge.req_slots_dev[:2].tolist() == [11, 13]


def test_request_kernel_no_disk_rows_does_not_ring(bridge):
    cache = _FakeCache(bridge.tier)
    _set_plan(cache, srcs=[0, 1, 2], slots=[10, 11, 12])  # all < ram=3
    _launch_request(bridge, cache)
    assert int(cache.num_indices.item()) == 3  # everything stays on the RAM plan
    assert int(bridge.req_host[0]) == 0
    assert int(bridge.req_host[bridge.seq_off]) == 0  # doorbell never rung


def test_serve_cycle_fills_staging_and_installs(bridge):
    from freetoken.kernel.fast_index_copy import fast_index_copy_jit

    cache = _FakeCache(bridge.tier)
    bridge.enable()  # enable first: it snapshots req seq and ignores stale seqs
    _set_plan(cache, srcs=[5, 1, 7], slots=[20, 21, 22])
    # stage_fetch launches the spin with the doorbell on: the launch itself
    # blocks the stream until the service thread acks -- the full doorbell
    # round-trip is exercised by this call completing.
    _launch_request(bridge, cache)
    assert int(bridge.req_host[0]) == 2  # rows 5 and 7 are disk-resident
    assert int(bridge.req_host[bridge.seq_off]) == 1

    deadline = time.time() + 10
    while int(bridge.resp_host[0]) < 1 and time.time() < deadline:
        time.sleep(0.001)
    bridge.disable()
    assert int(bridge.resp_host[0]) == 1, "service thread never acked"

    # Staging (pinned host, filled by the thread before acking) holds the
    # on-disk bytes for rows 5 and 7.
    staging = bridge.staging_host[0]
    assert staging[0].unique().tolist() == [5]
    assert staging[1].unique().tolist() == [7]

    # Install lands the staged rows in the recorded slots.
    gpu_cache = torch.zeros(N_SLOTS, ROW, dtype=torch.uint8, device="cuda")
    fast_index_copy_jit(gpu_cache, bridge.req_slots_dev,
                        staging, bridge.staging_idx_dev, bridge.req_count_dev)
    torch.cuda.synchronize()
    slots = bridge.req_slots_dev.tolist()
    assert slots[:2] == [20, 22]  # only the two disk rows were recorded
    assert gpu_cache[20].cpu().unique().tolist() == [5]
    assert gpu_cache[22].cpu().unique().tolist() == [7]
    assert gpu_cache[21].cpu().unique().tolist() == [0]  # RAM row: not ours


def test_overflow_refuses_to_ack(tmp_path):
    tier = _FakeTier(tmp_path, ram=0)
    cache = _FakeCache(tier)
    b = GraphFetchBridge(tier, cache, k_max=2)
    try:
        b.enable()  # first, so the snapshot doesn't swallow the request
        _set_plan(cache, srcs=[1, 2, 3], slots=[10, 11, 12])
        # Launch the request kernel + memcpy directly: stage_fetch would also
        # launch the spin, which (correctly) never returns for this request.
        _gf_request_kernel[(1,)](
            cache.num_indices, cache.evict_slots, cache.src_indices,
            b.req_dev, b.req_slots_dev, b.req_count_dev,
            RAM_LOCAL=0, K_MAX=2, LAYER=0, TRACE=False)
        b.req_host.copy_(b.req_dev, non_blocking=True)
        torch.cuda.synchronize()
        assert int(b.req_host[0]) == 3  # true count: over k_max=2, must refuse
        time.sleep(0.3)
        assert int(b.resp_host[0]) == 0, "over-capacity request must hang"
    finally:
        b.disable()
        b.shutdown()


def test_spin_noop_when_doorbell_off(bridge):
    from freetoken.moe.graph_fetch import _gf_spin_kernel
    # req seq ahead of resp but the doorbell is off: spin must not block.
    bridge.req_dev[bridge.seq_off] = 5
    _gf_spin_kernel[(1,)](bridge.resp_host, bridge.req_dev,
                          bridge.doorbell_dev, SEQ_OFF=bridge.seq_off,
                          TRACE=False)
    torch.cuda.synchronize()  # returns at all == pass
