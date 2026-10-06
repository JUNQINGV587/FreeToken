"""DS-FP4 conversion mode for the NVFP4 disk tier (moe.nvfp4_to_dsfp4): the cache
holds 4 DS-FP4 banks while the checkpoint rows stay native NVFP4 -- packed banks
copy 1:1, scale extents are repacked (per-16 e4m3 + per-expert fp32 global ->
per-32 e8m0) on the host between preadv and the H2D copy."""

import json
import re
import struct
import threading
import types

import numpy as np
import pytest
import torch

from freetoken.moe.disk_tier import DiskTier, Nvfp4DiskIndex
from freetoken.moe.host_banks import HostBank
from freetoken.moe.nvfp4_to_dsfp4 import (
    E4M3_EXP_LUT,
    _E8M0_NAN,
    convert_pieces,
    convert_scale_rows,
    global_exponent,
)
from freetoken.models.nvfp4_banks import Nvfp4ExpertSourceSpec

# Geometry that supports BOTH per-16 and per-32 groups.
H, I, E, L = 64, 64, 4, 2
SHARDS = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")

SPEC = Nvfp4ExpertSourceSpec(
    key_pattern=re.compile(
        r"^model\.language_model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
        r"(?P<proj>gate_proj|up_proj|down_proj)\."
        r"(?P<kind>weight|weight_scale|weight_scale_2)$"
    ),
    proj_to_role={"gate_proj": "gate", "up_proj": "up", "down_proj": "down"},
    layer_to_bank=lambda layer, config: layer,
    desc="dsfp4-convert test",
)

E8M0 = torch.float8_e8m0fnu

# Legal V4.1 scale content: e4m3 power-of-two bytes, adjacent 16-groups identical.
_POW2_BYTES = np.array([0x08, 0x10, 0x20, 0x30, 0x38, 0x40, 0x48, 0x50], dtype=np.uint8)


def _legal_scales(rows: int, k16: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pair = _POW2_BYTES[rng.integers(0, len(_POW2_BYTES), size=(rows, k16 // 2))]
    return np.repeat(pair, 2, axis=1)


def _pow2_global(seed: int) -> float:
    return float(2.0 ** (int(np.random.default_rng(seed).integers(-6, 2))))


def _name(layer, expert, proj, kind):
    return f"model.language_model.layers.{layer}.mlp.experts.{expert}.{proj}.{kind}"


@pytest.fixture()
def checkpoint(tmp_path):
    """Native NVFP4 checkpoint whose scales satisfy the V4.1 per-32 pow2 invariant."""
    import safetensors.torch

    by_shard = {s: {} for s in SHARDS}
    weight_map = {}
    for layer in range(L):
        for expert in range(E):
            shard = SHARDS[(layer * E + expert) % 2]
            seed = layer * 1000 + expert
            tensors = {
                _name(layer, expert, "gate_proj", "weight"):
                    torch.randint(0, 256, (I, H // 2), dtype=torch.uint8),
                _name(layer, expert, "up_proj", "weight"):
                    torch.randint(0, 256, (I, H // 2), dtype=torch.uint8),
                _name(layer, expert, "down_proj", "weight"):
                    torch.randint(0, 256, (H, I // 2), dtype=torch.uint8),
                _name(layer, expert, "gate_proj", "weight_scale"):
                    torch.from_numpy(_legal_scales(I, H // 16, seed + 1)),
                _name(layer, expert, "up_proj", "weight_scale"):
                    torch.from_numpy(_legal_scales(I, H // 16, seed + 2)),
                _name(layer, expert, "down_proj", "weight_scale"):
                    torch.from_numpy(_legal_scales(H, I // 16, seed + 3)),
                _name(layer, expert, "gate_proj", "weight_scale_2"):
                    torch.tensor(_pow2_global(seed + 4), dtype=torch.float32),
                _name(layer, expert, "up_proj", "weight_scale_2"):
                    torch.tensor(_pow2_global(seed + 5), dtype=torch.float32),
                _name(layer, expert, "down_proj", "weight_scale_2"):
                    torch.tensor(_pow2_global(seed + 6), dtype=torch.float32),
            }
            for name, t in tensors.items():
                by_shard[shard][name] = t
                weight_map[name] = shard
    for shard, tensors in by_shard.items():
        safetensors.torch.save_file(tensors, str(tmp_path / shard), metadata={"format": "pt"})
    with open(tmp_path / "model.safetensors.index.json", "w", encoding="utf-8") as f:
        json.dump({"weight_map": weight_map, "metadata": None}, f)
    config = types.SimpleNamespace(num_experts=E, hidden_size=H, moe_intermediate_size=I,
                                   num_layers=L, first_k_dense_replace=0)
    return tmp_path, config


# DS-FP4 (cache-side) bank shapes/dtypes: packed u8 + scales e8m0.
DS_BANK_SHAPES = ((2 * I, H // 2), (2 * I, H // 32), (H, I // 2), (H, I // 32))
DS_BANK_DTYPES = (torch.uint8, E8M0, torch.uint8, E8M0)


def _ds_cache(num_experts=E):
    banks = [
        (
            [torch.zeros(num_experts, *shape, dtype=dtype) for _ in range(L)],
            torch.full((8, *shape), 0x7F, dtype=dtype),
        )
        for shape, dtype in zip(DS_BANK_SHAPES, DS_BANK_DTYPES)
    ]
    cache = type("FakeCache", (), {})()
    cache.banks = banks
    cache.num_experts = num_experts
    cache.num_layers = L
    cache.num_indices = torch.tensor([0], dtype=torch.int64)
    cache.src_indices = torch.zeros(64, dtype=torch.int32)
    cache.evict_slots = torch.zeros(64, dtype=torch.int32)
    cache.quant_format = "ds_fp4"
    return cache


def _tier(checkpoint, cache, ram_experts=1):
    path, config = checkpoint
    index = Nvfp4DiskIndex(str(path), config, SPEC)
    tier = DiskTier(index, cache, ram_experts=ram_experts, workers=2)
    local = threading.local()

    def _staging_ring():
        ring = getattr(local, "ring", None)
        if ring is None:
            ring = [[HostBank((tier._staging_size,), torch.uint8), None]
                    for _ in range(tier._STAGING_RING)]
            local.ring = ring
        return ring

    tier._staging_ring = _staging_ring
    return tier


def _reference_rows(checkpoint, layer: int, expert: int):
    """Convert the checkpoint's expert row through convert_pieces = the expected
    DS-FP4 cache row content (the same math the pack path uses at load time)."""
    import safetensors

    def get(shard_path, name):
        with safetensors.safe_open(shard_path, framework="pt", device="cpu") as sf:
            return sf.get_tensor(name)

    pieces = {}
    path, _config = checkpoint
    with open(path / "model.safetensors.index.json", encoding="utf-8") as f:
        wmap = json.load(f)["weight_map"]
    for proj in ("gate_proj", "up_proj", "down_proj"):
        role = {"gate_proj": "gate", "up_proj": "up", "down_proj": "down"}[proj]
        for kind, suffix in (("weight", ""), ("weight_scale", "_scale"),
                             ("weight_scale_2", "_global")):
            name = _name(layer, expert, proj, kind)
            t = get(path / wmap[name], name)
            if kind == "weight_scale":
                t = t.view(torch.uint8)
            pieces[role + suffix] = t.unsqueeze(0)
    out = convert_pieces(pieces, verify=True)
    return {k: v[0] for k, v in out.items()}


def test_convert_fetch_places_all_banks(checkpoint):
    """End to end: tier fetch into a DS-FP4 cache matches the pack-path reference."""
    cache = _ds_cache()
    tier = _tier(checkpoint, cache, ram_experts=1)
    assert tier._convert and tier._disk_bank == (0, 1, 3, 4)
    layer, expert, slot = 1, 3, 5
    tier._preload_scalars()
    tier._fetch_expert(layer, expert, slot)
    ref = _reference_rows(checkpoint, layer, expert)
    for bank_idx, role in enumerate(("gate_up", "gate_up_scale", "down", "down_scale")):
        got = cache.banks[bank_idx][1][slot].contiguous().view(torch.uint8)
        want = ref[role].contiguous().view(torch.uint8)
        assert torch.equal(got, want), f"bank {bank_idx} ({role}) mismatch"
    # Byte accounting: disk-side NVFP4 bytes > cache-side DS-FP4 bytes.
    cache_bytes = sum(
        int(np.prod(s)) for s in DS_BANK_SHAPES)
    assert tier._fetch_bytes > cache_bytes
    assert tier._fetches == 1


def test_convert_group_runs_use_disk_bank_segments(checkpoint):
    """_group_runs(disk_bank=...) must resolve the NVFP4 disk segments while the
    destination slices stay the DS-FP4 cache row's."""
    cache = _ds_cache()
    tier = _tier(checkpoint, cache, ram_experts=1)
    # cache bank 2 (down_packed) reads disk bank 3 (down packed, one segment)
    groups = tier._group_runs(2, 0, 1, disk_bank=tier._disk_bank[2])
    assert len(groups) == 1
    shard_idx, a0, a1, members, _ = groups[0]
    assert members[0][3] == H * (I // 2)  # down packed bytes
    # cache bank 1 (gate_up_scale) -> disk bank 1: two NVFP4 segments
    index = tier._index
    segs = index.row_segments(1, 0, 1)
    assert len(segs) == 2


def test_convert_ref_row_matches_fetch(checkpoint):
    """The debug verify path (_ref_row) must reconstruct the same DS-FP4 row."""
    cache = _ds_cache()
    tier = _tier(checkpoint, cache, ram_experts=1)
    tier._preload_scalars()
    layer, expert = 0, 2
    ref = _reference_rows(checkpoint, layer, expert)
    for bank_idx, role in enumerate(("gate_up", "gate_up_scale", "down", "down_scale")):
        host_row = cache.banks[bank_idx][0][0][0]
        row_bytes = host_row.numel() * host_row.element_size()
        got = tier._ref_row(bank_idx, layer, expert, row_bytes,
                            host_row.element_size(),
                            host_row.numel() // host_row.shape[0])
        want = ref[role].contiguous().view(torch.uint8).reshape(-1)
        assert torch.equal(got, want), f"_ref_row bank {bank_idx} ({role}) mismatch"


def test_convert_scale_rows_rejects_inconsistent_pairs():
    """The per-32 invariant is the conversion's precondition: verify=True (the
    audit tool / offline gate) must fail loudly when adjacent 16-groups disagree.
    The hot fetch path runs verify=False and trusts that gate (see the module
    docstring): there the odd columns are simply dropped."""
    pair = _POW2_BYTES[[1, 3, 5, 2]]
    sc = np.tile(np.repeat(pair, 2), (4, 1)).copy()  # (4, 8), adjacent pairs equal
    legal = convert_scale_rows(sc[:1], 0, verify=True)
    assert legal.shape == (1, 4)
    sc[1, 1] = (int(sc[1, 0]) + 8) % 256  # adjacent pair disagrees
    with pytest.raises(ValueError, match="adjacent"):
        convert_scale_rows(sc, 0, verify=True)
    # verify=False: no raise, even column wins.
    out = convert_scale_rows(sc, 0, verify=False)
    assert out[1, 0] == E4M3_EXP_LUT[sc[1, 0]] + 127


def test_convert_scale_rows_out_of_e8m0_range_becomes_nan():
    """The V4.1 audit found e8m0-range violations at |gexp| >= 141; verify=False
    keeps converting (an illegal byte passes through as the NaN marker)."""
    sc = np.full((2, 4), 0x7F, dtype=np.uint8)  # 0x7F is e4m3 NaN (invalid byte)
    out = convert_scale_rows(sc, 141, verify=False)
    assert (out == _E8M0_NAN).all()
    with pytest.raises(ValueError, match="powers of two"):
        convert_scale_rows(sc, 0, verify=True)


def test_e4m3_exp_lut():
    """Every positive exact power of two (incl. subnormals) has a LUT entry; the
    sign bit, non-pow2 mantissas and NaN map to the invalid sentinel."""
    assert E4M3_EXP_LUT[0x38] == 0   # 1.0
    assert E4M3_EXP_LUT[0x08] == -6  # 2^-6 (smallest normal pow2)
    assert E4M3_EXP_LUT[0x04] == -7  # subnormal 0.5 x 2^-6
    assert E4M3_EXP_LUT[0x02] == -8
    assert E4M3_EXP_LUT[0x01] == -9
    assert E4M3_EXP_LUT[0x40] == 1   # 2.0
    assert E4M3_EXP_LUT[0x78] == 8   # 256 = 2^8 (largest e4m3 pow2: exp 15, man 0)
    assert E4M3_EXP_LUT[0x7E] == -32768  # exp 15 man 6 -> not pow2
    assert E4M3_EXP_LUT[0x7F] == -32768  # NaN
    assert E4M3_EXP_LUT[0x80] == -32768  # sign set
    assert E4M3_EXP_LUT[0x3C] == -32768  # 1.5 -> not pow2
