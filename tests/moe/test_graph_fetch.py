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

    def __init__(self, tmp_path, ram: int, row_map: torch.Tensor | None = None):
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
        # Pinned-row layout (local id -> bank row); identity without a pin set.
        if row_map is None:
            row_map = (
                torch.arange(16, dtype=torch.int32, device="cuda")
                .expand(N_LAYERS, -1).contiguous()
            )
        assert row_map.shape == (N_LAYERS, 16)
        self._row_map_dev = row_map.to(device="cuda", dtype=torch.int32).contiguous()

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

    # Doorbell ledger stubs (DiskTier.record_doorbell*): the bridge calls them
    # off the ack path; counting them here also feeds the refusal test.
    _db_timeouts = 0

    def record_doorbell(self, rows, host_us):
        pass

    def record_doorbell_timeout(self):
        self._db_timeouts += 1


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
        # Launch the request kernel + memcpy directly: the spin would be
        # released by the poisoned ack immediately, so test the host path
        # without it.
        _gf_request_kernel[(1,)](
            cache.num_indices, cache.evict_slots, cache.src_indices,
            b.req_dev, b.req_slots_dev, b.req_count_dev,
            tier._row_map_dev[0],
            RAM_LOCAL=0, K_MAX=2, LAYER=0, TRACE=False)
        b.req_host.copy_(b.req_dev, non_blocking=True)
        torch.cuda.synchronize()
        assert int(b.req_host[0]) == 3  # true count: over k_max=2, must refuse
        # Fail-loud semantics: the refusal sets the health flag, counts a
        # doorbell_timeout, and poisons the ack so the replay's spin releases
        # into the engine's raise_if_unhealthy (instead of hanging, the
        # pre-2026-10-06 behaviour).
        deadline = time.time() + 5
        while not b._err and time.time() < deadline:
            time.sleep(0.01)
        assert b._err, "over-capacity request must trip the health flag"
        assert int(b.resp_host[0]) == 1, "refusal must poison the ack with the seq"
        assert tier._db_timeouts == 1
        with pytest.raises(RuntimeError, match="k_max"):
            b.raise_if_unhealthy()
    finally:
        b.disable()
        b.shutdown()


def test_spin_noop_when_doorbell_off(bridge):
    from freetoken.moe.graph_fetch import _gf_spin_kernel
    # req seq ahead of resp but the doorbell is off: spin must not block.
    bridge.req_dev[bridge.seq_off] = 5
    _gf_spin_kernel[(1,)](bridge.resp_host, bridge.req_dev,
                          bridge.doorbell_dev, bridge.spin_stats_dev,
                          SEQ_OFF=bridge.seq_off,
                          TRACE=False)
    torch.cuda.synchronize()  # returns at all == pass
    # ...and it must not record a wait it never performed.
    assert int(bridge.spin_stats_dev[1]) == 0


def test_request_kernel_remaps_through_row_map(tmp_path):
    """Learned pin set: the split compares BANK ROWS, the request block keeps
    LOCAL IDS, and the compacted RAM plan carries bank rows for the PCIe copy.

    Pin set {2, 5, 9} (ram=3): row_map maps those experts to rows 0/1/2 and
    every other expert to a row >= 3. srcs [0, 5, 2, 7] -> bank rows
    [>=3, 1, 0, >=3], so 0 and 7 are disk-resident (requested by local id)
    while 5 and 2 compact as bank rows 1 and 0."""
    row_map = torch.full((N_LAYERS, 16), 3, dtype=torch.int32)
    rest = [e for e in range(16) if e not in (2, 5, 9)]
    for r, e in enumerate([2, 5, 9] + rest):
        row_map[:, e] = r
    tier = _FakeTier(tmp_path, ram=3, row_map=row_map)
    cache = _FakeCache(tier)
    b = GraphFetchBridge(tier, cache, k_max=K_MAX)
    try:
        _set_plan(cache, srcs=[0, 5, 2, 7], slots=[10, 11, 12, 13])
        _launch_request(b, cache)
        # RAM rows compacted in place as BANK ROWS (order kept).
        assert int(cache.num_indices.item()) == 2
        assert cache.evict_slots[:2].tolist() == [11, 12]
        assert cache.src_indices[:2].tolist() == [1, 0]
        # Disk rows recorded as LOCAL EXPERT IDS (the doorbell resolves the
        # checkpoint row from the local id).
        blk = b.req_host
        assert int(blk[0]) == 2
        assert blk[2:4].tolist() == [0, 7]
        assert int(blk[b.seq_off]) == 1
        assert b.req_slots_dev[:2].tolist() == [10, 13]
    finally:
        b.shutdown()


class _FakeRowGroupsTier(_FakeTier):
    """_FakeTier + the native cross-bank merged plan: every expert row is
    served as two half-row groups on bank 0, so the intra-row fan-out path
    (parallel_rows) has multiple groups to split across the row pool."""

    def _row_groups(self, layer, expert):
        a0 = expert * ROW
        half = ROW // 2
        return [
            [0, a0, a0 + half, [(0, 0, half, a0, half)], a0 + half],
            [0, a0 + half, a0 + ROW, [(0, half, ROW, a0 + half, half)],
             a0 + ROW],
        ]


@pytest.mark.parametrize("serial", [False, True])
def test_single_row_group_fanout(tmp_path, monkeypatch, serial):
    """count==1 doorbell request: with parallel_rows the row's merged groups
    are read concurrently on the row pool; both modes must produce the exact
    on-disk bytes (the drive's ~3.8x concurrency headroom is what the fan-out
    exists to use)."""
    monkeypatch.setenv("FT_GRAPH_FETCH_ROW_SERIAL", "1" if serial else "0")
    tier = _FakeRowGroupsTier(tmp_path, ram=0)
    cache = _FakeCache(tier)
    b = GraphFetchBridge(tier, cache, k_max=K_MAX)
    try:
        b.enable()
        _set_plan(cache, srcs=[5], slots=[20])
        _launch_request(b, cache)
        assert int(b.req_host[0]) == 1
        deadline = time.time() + 10
        while int(b.resp_host[0]) < 1 and time.time() < deadline:
            time.sleep(0.001)
        b.disable()
        assert int(b.resp_host[0]) == 1, "service thread never acked"
        # Row 5's byte pattern fills the whole staged row regardless of mode.
        assert b.staging_host[0][0].unique().tolist() == [5]
        assert tier._preadv_calls == 2  # both half-row groups were read
    finally:
        b.shutdown()


# ---------------------------------------------------------------------------
# DS-FP4 conversion mode (triton_dsfp4): the doorbell stages NATIVE NVFP4 scale
# bytes and installs them through the device-side convert kernel.


class _FakeConvertTier:
    """Fake tier in conversion mode: bank0 packed (identical on disk), bank1
    scale (cache row [4,4] e8m0; native NVFP4 row [4,8] e4m3, gate|up halves
    with different globals)."""

    PACK_ROWS, PACK_COLS = 2, 2048        # 4096 B (fast_index_copy min geometry)
    SC_ROWS, SC_COLS = 4, 4               # cache scale row 16 B; native 32 B
    EXPERT_BYTES = PACK_ROWS * PACK_COLS + SC_ROWS * SC_COLS * 2

    def __init__(self, tmp_path):
        import struct as st
        self._ram = 0
        self.ownership = None
        self._preadv_calls = 0
        self._convert = True
        self._disk_bank = (0, 1, 3, 4)
        self._scale_convert = {1: (1, 2)}
        self._dst_slices = [[(0, self.PACK_ROWS)],
                            [(0, 2), (2, 4)]]
        self._index = SimpleNamespace(scalar_banks=frozenset({2, 5}),
                                      num_layers=N_LAYERS)
        self._telemetry_path = None
        self._staging_size = 4096
        # Disk: per expert [packed 64B = byte expert][scale 32B = byte 64]
        path = tmp_path / "fake_disk_conv.bin"
        buf = bytearray()
        for e in range(16):
            buf += bytes([e % 256]) * 4096
            buf += bytes([64]) * 32      # e4m3 0x40 = 2^1, positive pow2
        path.write_bytes(bytes(buf))
        self._path = str(path)
        # Scalar preload blob: fp32, local-expert-major, (gate, up) = (2^-13, 2^-5)
        self._blob = b"".join(st.pack("<2f", 2.0 ** -13, 2.0 ** -5)
                              for _ in range(16))
        host_packed = [torch.zeros(16, self.PACK_ROWS, self.PACK_COLS,
                                   dtype=torch.uint8) for _ in range(N_LAYERS)]
        gpu_packed = torch.zeros(N_SLOTS, self.PACK_ROWS, self.PACK_COLS,
                                 dtype=torch.uint8, device="cuda")
        host_scale = [torch.zeros(16, self.SC_ROWS, self.SC_COLS,
                                  dtype=torch.uint8) for _ in range(N_LAYERS)]
        gpu_scale = torch.zeros(N_SLOTS, self.SC_ROWS, self.SC_COLS,
                                dtype=torch.uint8, device="cuda")
        self._banks = [(host_packed, gpu_packed), (host_scale, gpu_scale)]
        self._row_map_dev = (
            torch.arange(16, dtype=torch.int32, device="cuda")
            .expand(N_LAYERS, -1).contiguous())

    def _fd(self, shard_idx):
        return os.open(self._path, os.O_RDONLY), False

    def _scalar_blob(self, layer, bank):
        assert (layer, bank) == (0, 2) or bank == 2
        return self._blob

    def _group_runs(self, bank_idx, layer, expert, disk_bank=None):
        base = expert * self.EXPERT_BYTES
        if bank_idx == 0:
            assert disk_bank == 0
            yield 0, base, base + 4096, [(0, 2, base, 4096)], base + 4096
        else:
            assert disk_bank == 1
            yield 0, base + 4096, base + 4112, [(0, 2, base + 4096, 16)], base + 4112
            yield 0, base + 4112, base + 4128, [(2, 4, base + 4112, 16)], base + 4128

    def _fill_scalar_row(self, *a, **k):  # pragma: no cover
        raise AssertionError("scalar banks must not be filled in convert mode")

    def mark_turn_src(self, *a, **k):
        pass

    def end_turn(self):
        pass


def test_convert_mode_doorbell_stages_and_installs(tmp_path):
    """End-to-end doorbell round-trip in conversion mode: request kernel ->
    thread stages native scale bytes + per-half global exponents -> device
    convert-install folds the global and halves the scale width."""
    tier = _FakeConvertTier(tmp_path)
    cache = _FakeCache(tier)
    b = GraphFetchBridge(tier, cache, k_max=K_MAX)  # __init__ compile-warms
    try:
        # Convert-mode staging: scale bank stages the DOUBLE-width native row.
        assert tuple(b.staging_host[1].shape) == (K_MAX, 4, 8)
        assert 1 in b._gexp_host
        b.enable()
        _set_plan(cache, srcs=[5], slots=[20])
        _launch_request(b, cache)
        assert int(b.req_host[0]) == 1
        deadline = time.time() + 10
        while int(b.resp_host[0]) < 1 and time.time() < deadline:
            time.sleep(0.001)
        b.disable()
        assert int(b.resp_host[0]) == 1, "service thread never acked"

        # Staging holds the NATIVE bytes; gexp the two per-half exponents.
        assert b.staging_host[1][0].unique().tolist() == [64]
        assert b._gexp_host[1][0].tolist() == [-13, -5]

        # The eager stage_fetch already installed (spin ordered before install):
        # bank0 verbatim, bank1 converted per half: E=8 -> 8+120+gexp.
        gpu_packed, gpu_scale = tier._banks[0][1], tier._banks[1][1]
        assert gpu_packed[20].cpu().unique().tolist() == [5]
        got = gpu_scale[20].cpu()
        assert got[0].unique().tolist() == [115]   # gate half: 8+120-13
        assert got[1].unique().tolist() == [115]
        assert got[2].unique().tolist() == [123]   # up half: 8+120-5
        assert got[3].unique().tolist() == [123]
        # Native byte 64 is NOT what the cache holds (no verbatim install).
        assert 64 not in got.unique().tolist()
    finally:
        b.disable()
        b.shutdown()
