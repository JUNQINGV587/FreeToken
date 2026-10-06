"""CPU tests for GraphFetchBridge._read_into_staging after the W19 host-fetch
optimisation: ctypes.memmove member copies + per-thread bounce slabs + the
row pool. No CUDA anywhere (the bridge is built with object.__new__)."""
from __future__ import annotations

import os
import queue
import threading
import time
import types

import pytest
import torch

from freetoken.moe.graph_fetch import GraphFetchBridge

N_LAYERS = 2
N_EXPERTS = 8
ROW = 4096          # staging row elements (uint8 here)
HALF = ROW // 2
SLAB = ROW * 2      # tier._staging_size (bounce must hold one aligned run + pad)


def _pattern(n: int, seed: int) -> bytes:
    return bytes(((i * 7 + seed * 13) % 251) for i in range(n))


class _FakeIndex:
    scalar_banks: tuple = ()


class _FakeTier:
    """One uint8 bank whose expert row is split across TWO non-adjacent file
    ranges (a gate|up style 2-member layout with junk in between), so the
    offset math of the memmove path is actually exercised."""

    def __init__(self, path: str, file_bytes: bytes, off_a: int, off_b: int):
        self._staging_size = SLAB
        self._index = _FakeIndex()
        self._preadv_calls = 0
        self._prefetch_window = 0
        self._fd_no = os.open(path, os.O_RDONLY)
        self._file = file_bytes
        self.off_a = off_a  # file offset of member 0 (dst [0, HALF))
        self.off_b = off_b  # file offset of member 1 (dst [HALF, ROW))
        host = [torch.zeros(N_EXPERTS, ROW, dtype=torch.uint8)
                for _ in range(N_LAYERS)]
        self._banks = [(host, torch.zeros(1, 1))]

    def _fd(self, shard_idx: int):
        return self._fd_no, False

    def _group_runs(self, bank_idx: int, layer: int, expert: int):
        # expert e's row lives at off_a + e*ROW (member 0) and
        # off_b + e*ROW (member 1): two runs, never mergeable (gap of junk).
        a0 = self.off_a + expert * ROW
        b0 = self.off_b + expert * ROW
        return [
            (0, a0, a0 + HALF, [(0, HALF, a0, HALF)], a0 + HALF),
            (0, b0, b0 + HALF, [(HALF, ROW, b0, HALF)], b0 + HALF),
        ]

    def _fill_scalar_row(self, bank_idx, layer, expert, row):  # pragma: no cover
        raise AssertionError("no scalar banks in this fake")


def _make_bridge(tier: "_FakeTier", n_slabs: int = 8) -> GraphFetchBridge:
    br = object.__new__(GraphFetchBridge)
    br.tier = tier
    br.k_max = 8
    br._bounce_tls = threading.local()
    br._bounce_slabs = queue.SimpleQueue()
    for _ in range(n_slabs):
        br._bounce_slabs.put(torch.zeros(SLAB, dtype=torch.uint8))
    br.staging_host = [torch.full((8, ROW), 0xFF, dtype=torch.uint8)]
    br._last_pf_check = 0.0
    return br


def _expected_row(tier: _FakeTier, expert: int) -> torch.Tensor:
    a0 = tier.off_a + expert * ROW
    b0 = tier.off_b + expert * ROW
    exp = torch.empty(ROW, dtype=torch.uint8)
    exp[:HALF] = torch.tensor(list(tier._file[a0:a0 + HALF]), dtype=torch.uint8)
    exp[HALF:] = torch.tensor(list(tier._file[b0:b0 + HALF]), dtype=torch.uint8)
    return exp


def _fake_tier(tmp_path) -> _FakeTier:
    off_a, off_b = 1000, 100000  # non-adjacent, unaligned, junk around
    n = off_b + N_EXPERTS * ROW + 500
    buf = bytearray(_pattern(n, seed=5))
    p = tmp_path / "shard0.bin"
    p.write_bytes(bytes(buf))
    return _FakeTier(str(p), bytes(buf), off_a, off_b)


def test_read_into_staging_byte_exact(tmp_path):
    tier = _fake_tier(tmp_path)
    br = _make_bridge(tier)
    for expert in (0, 3, N_EXPERTS - 1):
        br.staging_host[0].fill_(0xFF)
        br._read_into_staging(1, expert, 5)
        got = br.staging_host[0][5]
        assert torch.equal(got, _expected_row(tier, expert)), (
            f"expert {expert}: staged row mismatch")
        # untouched staging rows keep their 0xFF sentinel
        assert int(br.staging_host[0][4, 0]) == 0xFF
    assert tier._preadv_calls == 2 * 3  # two runs per row


def test_read_into_staging_concurrent_rows(tmp_path):
    tier = _fake_tier(tmp_path)
    br = _make_bridge(tier, n_slabs=8)
    experts = [1, 2, 5, 7]
    errors: list = []

    def work(j: int, expert: int) -> None:
        try:
            br._read_into_staging(0, expert, j)
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=work, args=(j, e))
               for j, e in enumerate(experts)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, f"worker errors: {errors}"
    for j, e in enumerate(experts):
        assert torch.equal(br.staging_host[0][j], _expected_row(tier, e)), (
            f"row {j} (expert {e}) corrupted under concurrency")


def test_bounce_slab_exhaustion_is_loud(tmp_path):
    tier = _fake_tier(tmp_path)
    br = _make_bridge(tier, n_slabs=1)
    br._bounce_slabs.get_nowait()  # drain the only slab
    try:
        br._read_into_staging(0, 0, 0)
    except RuntimeError as e:
        assert "bounce slab" in str(e)
    else:  # pragma: no cover
        raise AssertionError("expected loud slab-exhaustion error")


# ---------------------------------------------------------------------------
# M1 hardening: watchdog timeout / over-k_max refusal / spin-stats readout.
# Host-side logic only -- no CUDA (same object.__new__ shell as above).


class _WdTier:
    """Just enough tier for the watchdog paths: a timeout counter."""

    def __init__(self):
        self.timeouts = 0

    def record_doorbell_timeout(self):
        self.timeouts += 1


def _wd_bridge(k_max: int = 4, timeout_s: float = 0.2) -> GraphFetchBridge:
    b = object.__new__(GraphFetchBridge)
    b.k_max = k_max
    b.seq_off = 2 + k_max
    # Plain CPU tensors stand in for the pinned pair: the watchdog and refusal
    # paths are plain CPU stores/reads, pinning is irrelevant to them.
    b.req_host = torch.zeros(3 + k_max, dtype=torch.int64)
    b.resp_host = torch.zeros(1, dtype=torch.int64)
    b.tier = _WdTier()
    b._served = 0
    b._err = False
    b._stop = threading.Event()
    b._timeout_s = timeout_s
    b._device = "cpu"  # only interpolated into log lines
    return b


def test_watchdog_timeout_poisons_and_fails_loud():
    b = _wd_bridge()
    b.req_host[b.seq_off] = 1  # request seq 1 arrives, is never acked
    t = threading.Thread(target=b._watchdog, daemon=True)
    t.start()
    try:
        deadline = time.time() + 5
        while not b._err and time.time() < deadline:
            time.sleep(0.01)
        assert b._err, "watchdog never fired on a wedged request"
        assert int(b.resp_host[0]) == 1, "watchdog must poison the ack to release the spin"
        assert b.tier.timeouts == 1
        with pytest.raises(RuntimeError, match="graph-doorbell"):
            b.raise_if_unhealthy()
        # The poisoned sequence must not refire...
        time.sleep(0.5)
        assert b.tier.timeouts == 1
        # ...but a NEW wedged request fires again.
        b._err = False
        b.req_host[b.seq_off] = 2
        deadline = time.time() + 5
        while b.tier.timeouts < 2 and time.time() < deadline:
            time.sleep(0.01)
        assert b.tier.timeouts == 2
        assert int(b.resp_host[0]) == 2
    finally:
        b._stop.set()
        t.join(timeout=2)


def test_watchdog_ignores_acked_and_legacy_sequences():
    b = _wd_bridge()
    b.req_host[b.seq_off] = 1
    b.resp_host[0] = 1  # already acked: nothing is pending
    t = threading.Thread(target=b._watchdog, daemon=True)
    t.start()
    try:
        time.sleep(0.5)
        assert not b._err
        assert b.tier.timeouts == 0
    finally:
        b._stop.set()
        t.join(timeout=2)


def test_refuse_overcount_fails_loud():
    b = _wd_bridge(k_max=2)
    b.req_host[b.seq_off] = 7
    b._refuse_overcount(count=3, seq=7)
    assert b._err, "over-k_max refusal must trip the health flag"
    assert b.tier.timeouts == 1, "refusal counts as a doorbell timeout"
    assert int(b.resp_host[0]) == 7, "refusal must poison the ack with the seq"
    with pytest.raises(RuntimeError, match="k_max"):
        b.raise_if_unhealthy()


def test_spin_stats_readout():
    b = object.__new__(GraphFetchBridge)
    b.spin_stats_host = torch.tensor([150_000_000, 3, 100_000_000],
                                     dtype=torch.int64)
    s = b.spin_stats()
    assert s["doorbell_spins"] == 3
    assert s["doorbell_wait_ms"] == 50.0  # 150ms total / 3 waits
    assert s["doorbell_wait_ms_peak"] == 100.0
    # No waits yet -> 0.0, never a division error.
    b.spin_stats_host = torch.zeros(3, dtype=torch.int64)
    assert b.spin_stats()["doorbell_wait_ms"] == 0.0
