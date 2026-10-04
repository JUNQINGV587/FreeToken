"""CPU tests for GraphFetchBridge._read_into_staging after the W19 host-fetch
optimisation: ctypes.memmove member copies + per-thread bounce slabs + the
row pool. No CUDA anywhere (the bridge is built with object.__new__)."""
from __future__ import annotations

import os
import queue
import threading
import types

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
