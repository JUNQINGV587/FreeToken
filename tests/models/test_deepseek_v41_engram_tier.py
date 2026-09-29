"""The engram tier: rows of the 94 GiB tables read off disk must be the rows the resident table holds.

Two halves. A hand-built shard (real safetensors framing, real F8_E4M3/F8_E8M0 dtypes, unaligned
offsets) proves the gather, the out-of-range masking and the row geometry -- offline, so it runs
everywhere. Then, when ``FT_V41_CHECKPOINT`` points at the real checkpoint, the same gather is
checked against the 94 GiB tensors themselves: the geometry the tier derives from the index json and
shard headers has to match, and a handful of rows read through ``row_store`` have to be the bytes
``safetensors`` returns for those same rows.
"""

from __future__ import annotations

import json
import os
import struct

import pytest
import torch

pytest.importorskip("freetoken.kernel._row_store")

from freetoken.models.deepseek_v41.engram import ResidentEngramTable
from freetoken.models.deepseek_v41.engram_tier import EngramTier, locate_engram_tables

BLOCK = 32


def _write_shard(path, tensors) -> None:
    """A safetensors file with the frames the reader walks: 8-byte length + JSON, then the data.

    ``tensors`` is a list of ``(name, dtype, shape, bytes)``; each tensor starts 8-byte aligned
    inside the data section, exactly as the real writer leaves them.
    """
    header: dict[str, dict] = {}
    blob = bytearray()
    for name, dtype, shape, data in tensors:
        pad = (8 - len(blob) % 8) % 8
        blob += b"\0" * pad
        start = len(blob)
        blob += data
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [start, start + len(data)]}
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * ((8 - len(raw) % 8) % 8)  # the header is padded so the data section is 8-aligned
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(raw)))
        fh.write(raw)
        fh.write(bytes(blob))


def _fp8_bytes(rows: int, dim: int, seed: int) -> torch.Tensor:
    """Valid e4m3 bytes: the all-ones exponent (0x7F/0xFF) would be a NaN, and NaN != NaN."""
    gen = torch.Generator().manual_seed(seed)
    raw = torch.randint(0, 256, (rows, dim), dtype=torch.int64, generator=gen).to(torch.uint8)
    return raw.masked_fill((raw & 0x7F) == 0x7F, 0)


def _e8m0_bytes(rows: int, cols: int, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(120, 134, (rows, cols), dtype=torch.int64, generator=gen).to(torch.uint8)


def _as_fp8(raw: torch.Tensor):
    return raw.contiguous().view(torch.float8_e4m3fn)


def _as_e8m0(raw: torch.Tensor):
    return raw.contiguous().view(torch.float8_e8m0fnu)


def _bytes(t: torch.Tensor) -> bytes:
    return t.contiguous().view(torch.uint8).numpy().tobytes()


def _no_host_read(self, loc, flat):
    raise AssertionError("a captured gather must not take the eager D2H path")


class _FakeBridge:
    """``EngramGraphFetch`` minus the CUDA: stages the requested rows into private device buffers.

    The real bridge hands the ids to a host thread and serves rows out of pinned memory; here the
    rows come straight from the shard's bytes, so what is checked is the routing, the row count the
    request carries and the dequantization -- not the transport.
    """

    def __init__(self, weights: torch.Tensor, scales: torch.Tensor) -> None:
        self._w = weights
        self._s = scales
        self.calls: list[tuple[int, list[list[int]], int]] = []
        self.out_w = torch.zeros(weights.numel(), dtype=torch.uint8)
        self.out_s = torch.zeros(scales.numel(), dtype=torch.uint8)

    def stage_gather(self, layer_index: int, ids: torch.Tensor, n: int) -> None:
        self.calls.append((layer_index, ids[:n].tolist(), n))
        rows = ids[:n]
        bad = (rows < 0) | (rows >= self._w.shape[0])
        safe = rows.clamp(0, self._w.shape[0] - 1)
        w = self._w[safe].masked_fill(bad[:, None], 0)
        s = self._s[safe].masked_fill(bad[:, None], 0)
        self.out_w[: n * self._w.shape[1]] = w.reshape(-1)
        self.out_s[: n * self._s.shape[1]] = s.reshape(-1)


@pytest.fixture(scope="module")
def fake_checkpoint(tmp_path_factory):
    """Two engram layers, 40 and 24 rows, one shard -- the two layers do NOT share a row count."""
    folder = tmp_path_factory.mktemp("engram-ckpt")
    dim = 64
    scales = {1: _e8m0_bytes(40, dim // BLOCK, 11), 14: _e8m0_bytes(24, dim // BLOCK, 12)}
    weights = {1: _fp8_bytes(40, dim, 1), 14: _fp8_bytes(24, dim, 2)}
    tensors = []
    for layer_id in (1, 14):
        tensors.append(
            (f"layers.{layer_id}.engram.embed.weight", "F8_E4M3", weights[layer_id].shape, _bytes(weights[layer_id]))
        )
    # weights first, then the scales: the scale base is not adjacent to the weight it belongs to
    for layer_id in (1, 14):
        tensors.append(
            (f"layers.{layer_id}.engram.embed.scale", "F8_E8M0", scales[layer_id].shape, _bytes(scales[layer_id]))
        )
    shard = "model-00001-of-00001.safetensors"
    _write_shard(os.path.join(folder, shard), tensors)
    index = {
        "metadata": {"total_size": sum(len(t[3]) for t in tensors)},
        "weight_map": {t[0]: shard for t in tensors},
    }
    with open(os.path.join(folder, "model.safetensors.index.json"), "w") as fh:
        json.dump(index, fh)
    return folder, dim, weights, scales


@pytest.mark.parametrize("use_io_uring", [True, False])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_a_gather_returns_the_resident_tables_rows_bitwise(fake_checkpoint, use_io_uring, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    folder, dim, weights, scales = fake_checkpoint
    tier = EngramTier(folder, [1, 14], device=device, max_rows=64, use_io_uring=use_io_uring)
    assert [loc.rows for loc in tier.locations] == [40, 24]
    assert (tier.dim, tier.block) == (dim, BLOCK)

    for index, layer_id in enumerate((1, 14)):
        resident = ResidentEngramTable(_as_fp8(weights[layer_id]), _as_e8m0(scales[layer_id]))
        rows = torch.tensor([[0, 3, 7], [39 if layer_id == 1 else 23, 1, 1]])
        got = tier.gather(index, rows)
        want = resident.gather(rows).to(device)
        assert got.shape == (2, 3, dim)
        assert got.dtype == torch.bfloat16
        assert torch.equal(got, want), layer_id


def test_out_of_range_and_negative_rows_read_as_zero(fake_checkpoint):
    folder, dim, weights, scales = fake_checkpoint
    tier = EngramTier(folder, [1], device="cpu", max_rows=8)
    resident = ResidentEngramTable(_as_fp8(weights[1]), _as_e8m0(scales[1]))
    rows = torch.tensor([-1, 0, 39, 40, 1000, 5])
    got = tier.gather(0, rows)
    want = resident.gather(rows)
    assert got.device == tier.device
    assert torch.equal(got, want)
    assert torch.equal(got[0], torch.zeros(dim, dtype=torch.bfloat16))
    assert torch.equal(got[3], torch.zeros(dim, dtype=torch.bfloat16))
    assert torch.equal(got[4], torch.zeros(dim, dtype=torch.bfloat16))
    assert not torch.equal(got[2], torch.zeros(dim, dtype=torch.bfloat16))


def test_a_gather_larger_than_the_staging_buffer_is_chunked(fake_checkpoint):
    """A 4096-token prefill gathers 24576 rows; the staging buffer is sized for one round trip, not
    for a whole forward, so the over-long gather has to be split rather than refused."""
    folder, dim, weights, scales = fake_checkpoint
    tier = EngramTier(folder, [1], device="cpu", max_rows=6, use_io_uring=False)
    resident = ResidentEngramTable(_as_fp8(weights[1]), _as_e8m0(scales[1]))

    sizes: list[int] = []
    eager = EngramTier._gather_eager
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        EngramTier, "_gather_eager", lambda self, loc, flat: (sizes.append(int(flat.numel())), eager(self, loc, flat))[1]
    )
    try:
        rows = torch.arange(13) % 40  # 13 rows through a 6-row buffer: 6 + 6 + 1
        got = tier.gather(0, rows)
    finally:
        monkeypatch.undo()
    assert sizes == [6, 6, 1]
    assert torch.equal(got, resident.gather(rows))
    # a chunk boundary must not drop or duplicate a row
    assert torch.equal(got[6], resident.gather(rows[6:7])[0])


def test_a_captured_gather_goes_through_the_doorbell_and_never_reads_the_host(fake_checkpoint):
    """Inside a capture the ids cannot leave the device, so ``gather`` must hand them to the bridge.

    The bridge is faked out (the real one needs CUDA); what is under test is the routing and the
    dequant of the staged rows, and that the eager D2H path is not reachable from a capture.
    """
    folder, dim, weights, scales = fake_checkpoint
    tier = EngramTier(folder, [1], device="cpu", max_rows=64, use_io_uring=False)
    resident = ResidentEngramTable(_as_fp8(weights[1]), _as_e8m0(scales[1]))
    bridge = _FakeBridge(weights[1], scales[1])
    tier.device = torch.device("cuda")  # the branch keys off the tier's device, not a real one
    tier._graph_bridge = bridge

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
        patch.setattr(EngramTier, "_gather_eager", _no_host_read)
        # an id the table does not have: the host side zeroes it, so the graph needs no mask
        rows = torch.tensor([[-1, 0, 3], [39, 40, 7]])
        got = tier.gather(0, rows)
    assert bridge.calls == [(0, [-1, 0, 3, 39, 40, 7], 6)]  # the request carries the flat ids
    assert got.shape == (2, 3, dim)
    assert torch.equal(got, resident.gather(rows))


def test_a_capture_without_the_bridge_says_so_instead_of_hitting_the_driver(fake_checkpoint):
    """FREETOKEN_ENGRAM_FETCH=0 plus graphs on must fail loudly, not inside the D2H."""
    folder, _, _, _ = fake_checkpoint
    tier = EngramTier(folder, [1], device="cpu", max_rows=64, use_io_uring=False)
    tier.device = torch.device("cuda")
    tier._graph_bridge = None

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
        with pytest.raises(RuntimeError, match="needs the doorbell bridge"):
            tier.gather(0, torch.zeros(2, 3, dtype=torch.int64))


def test_an_empty_gather_does_not_touch_the_disk(fake_checkpoint):
    folder, dim, _, _ = fake_checkpoint
    tier = EngramTier(folder, [14], device="cpu", max_rows=4)
    got = tier.gather(0, torch.zeros(0, 2, dtype=torch.int64))
    assert got.shape == (0, 2, dim)


def test_a_separate_two_layer_request_keeps_its_own_row_space(fake_checkpoint):
    """Layer 14's rows must not be read out of layer 1's extent (they have different row counts)."""
    folder, _, weights, scales = fake_checkpoint
    tier = EngramTier(folder, [1, 14], device="cpu", max_rows=8)
    resident = ResidentEngramTable(_as_fp8(weights[14]), _as_e8m0(scales[14]))
    rows = torch.tensor([0, 23])
    assert torch.equal(tier.gather(1, rows), resident.gather(rows))  # index 1 == layer 14
    resident1 = ResidentEngramTable(_as_fp8(weights[1]), _as_e8m0(scales[1]))
    assert torch.equal(tier.gather(0, rows), resident1.gather(rows))


def test_the_tier_hands_the_layer_the_shape_it_asks_for(fake_checkpoint):
    """``Engram`` gathers its own ``[B, L, n_cols]`` hash grid and then flattens the last two dims."""
    folder, dim, _, _ = fake_checkpoint
    tier = EngramTier(folder, [1], device="cpu", max_rows=16)
    grid = torch.tensor([[[0, 5, 7, 7], [1, 2, 3, 4]]])  # (max_ngram - 1) * n_heads columns
    got = tier.gather(0, grid)
    assert got.shape == (1, 2, 4, dim)  # what wkv sees is gather(...).flatten(-2)
    assert got.flatten(-2).shape == (1, 2, 4 * dim)


def test_the_geometry_check_refuses_a_table_whose_scale_does_not_cover_it(fake_checkpoint, tmp_path):
    folder, dim, weights, _ = fake_checkpoint
    _write_shard(
        os.path.join(tmp_path, "bad.safetensors"),
        [
            ("layers.2.engram.embed.weight", "F8_E4M3", (40, dim), _bytes(weights[1])),
            ("layers.2.engram.embed.scale", "F8_E8M0", (40, dim // 64), _e8m0_bytes(40, dim // 64, 3).numpy().tobytes()),
        ],
    )
    with open(os.path.join(tmp_path, "model.safetensors.index.json"), "w") as fh:
        json.dump(
            {
                "weight_map": {
                    "layers.2.engram.embed.weight": "bad.safetensors",
                    "layers.2.engram.embed.scale": "bad.safetensors",
                }
            },
            fh,
        )
    with pytest.raises(ValueError):
        locate_engram_tables(str(tmp_path), [2])


def test_the_index_must_name_the_table(fake_checkpoint, tmp_path):
    with open(os.path.join(fake_checkpoint[0], "model.safetensors.index.json")) as fh:
        index = json.load(fh)
    with open(os.path.join(tmp_path, "model.safetensors.index.json"), "w") as fh:
        json.dump(index, fh)
    with pytest.raises(KeyError, match="layers.7.engram.embed.weight"):
        locate_engram_tables(str(tmp_path), [7])


# --------------------------------------------------------------------------- the real checkpoint

CHECKPOINT = os.getenv("FT_V41_CHECKPOINT")
LAYERS = (1, 14)
ROWS = (384006168, 384016682)
DIM = 256


@pytest.mark.skipif(not CHECKPOINT, reason="FT_V41_CHECKPOINT is not set")
def test_the_real_tables_are_where_the_reference_says_and_gather_the_same_rows():
    import safetensors

    from freetoken.models.deepseek_v41.engram_tier import _WEIGHT_SUFFIX

    locations = locate_engram_tables(CHECKPOINT, LAYERS)
    assert [loc.rows for loc in locations] == list(ROWS)
    assert [loc.dim for loc in locations] == [DIM, DIM]
    assert [loc.block for loc in locations] == [32, 32]
    assert locations[0].weight_path != "" and os.path.exists(locations[0].weight_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tier = EngramTier(CHECKPOINT, LAYERS, device=device, max_rows=512)
    for index, layer_id in enumerate(LAYERS):
        loc = locations[index]
        rows = torch.tensor([0, 1, 17, loc.rows // 2, loc.rows - 1, loc.rows, -1])
        got = tier.gather(index, rows)

        with safetensors.safe_open(loc.weight_path, framework="pt", device="cpu") as fh:
            weight = fh.get_slice(f"layers.{layer_id}{_WEIGHT_SUFFIX}")[
                rows.clamp(0, loc.rows - 1).tolist()
            ]
        with safetensors.safe_open(loc.scale_path, framework="pt", device="cpu") as fh:
            scale = fh.get_slice(f"layers.{layer_id}.engram.embed.scale")[rows.clamp(0, loc.rows - 1).tolist()]
        # get_slice keeps a leading batch dim when handed a list of rows; the tier flattens
        values = weight.reshape(-1, DIM).float()
        scales = scale.reshape(-1, DIM // 32).float()
        want = (values.unflatten(-1, (-1, 32)) * scales.unsqueeze(-1)).flatten(-2).to(torch.bfloat16)
        want = want.masked_fill(((rows < 0) | (rows >= loc.rows)).unsqueeze(-1), 0)
        assert torch.equal(got.cpu(), want), (layer_id, (got.cpu().float() - want.float()).abs().max())
