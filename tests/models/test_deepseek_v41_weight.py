"""DeepSeek-V4.1 loader against a synthetic checkpoint shaped like the release.

Tiny tensors, real key names and dtypes (fp8 + e8m0 block-32 scales, bf16
compressor/indexer pieces, NVFP4 expert dialect, 94 GiB-class engram table
dialect). Covers: ``load_args`` (inference/config.json -> DeepseekV41Args),
``parse_config`` (-> ModelConfig with the DSV41 attention group), ``iter_weights``
(name set / dtypes / wo_a dequant / exclusions) and ``nvfp4_expert_spec``.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from freetoken.attention.base import AttnType
from freetoken.models.config import DSV41AttentionGroupConfig
from freetoken.models.deepseek_v41.args import DeepseekV41Args, load_args
from freetoken.models.deepseek_v41.config import parse_config
from freetoken.models.deepseek_v41.weight import iter_weights, nvfp4_expert_spec

FP8 = torch.float8_e4m3fn

# tiny dims (every fp8 block scale is 32x32)
V, D, MI, QR, HD, RD = 64, 64, 32, 32, 32, 16  # vocab, dim, moe_inter, q_lora, head_dim, rope dim
NH, OG, OL, E = 2, 2, 16, 4  # heads, o_groups, o_lora, routed experts
INH, IHD = 2, 16  # index heads / head dim
HCM = 2  # hc_mult

_ARGS = {
    "vocab_size": V,
    "dim": D,
    "moe_inter_dim": MI,
    "n_layers": 4,
    "n_mtp_layers": 1,
    "n_heads": NH,
    "n_routed_experts": E,
    "n_shared_experts": 1,
    "n_activated_experts": 2,
    "score_func": "sqrtsoftplus",
    "route_scale": 1.5,
    "swiglu_limit": 10.0,
    "q_lora_rank": QR,
    "head_dim": HD,
    "rope_head_dim": RD,
    "norm_eps": 1e-20,
    "o_groups": OG,
    "o_lora_rank": OL,
    "window_size": 8,
    "compress_ratios": [0, 2, 0, 1, 0],  # 4 backbone + 1 mtp; L1 ratio 2, L3 ratio 1
    "kv_source_layers": [1, 3],
    "index_source_layers": [1, 2, 3],
    "candidate_source_layer": 3,
    "candidate_topk_blocks": 8,
    "candidate_block_size": 2,
    "compress_rope_theta": 160000.0,
    "original_seq_len": 4096,
    "rope_theta": 10000.0,
    "rope_factor": 16,
    "beta_fast": 32,
    "beta_slow": 1,
    "index_n_heads": INH,
    "index_head_dim": IHD,
    "index_topk": 16,
    "hc_mult": HCM,
    "hc_sinkhorn_iters": 20,
    "hc_eps": 1e-6,
    "engram_layer_ids": [1],
    "engram_vocab_size": 1000,
    "engram_num_embeddings": [101],
    "engram_max_ngram_size": 4,
    "engram_pad_id": 2,
    "engram_compressed_vocab_size": 500,
    "engram_n_heads": 2,
    "engram_head_dim": 32,
    "dspark_block_size": 5,
    "dspark_noise_token_id": 63,
    "dspark_target_layer_ids": [2, 3],
    "dspark_markov_rank": 8,
    "dspark_n_routed_experts": 4,
    "dspark_n_activated_experts": 2,
    "image_token_id": 62,
    "max_seq_len": 256,
}


def _bf16(*shape):
    return torch.randn(*shape).to(torch.bfloat16)


def _fp8(*shape):
    return (torch.randn(*shape) * 0.1).to(FP8)


def _scale(*shape):
    # e8m0 code around 127 (scale ~1)
    return torch.full(shape, 127, dtype=torch.uint8)


def _checkpoint(folder):
    """Write inference/config.json + a single safetensors shard + its index."""
    inf = folder / "inference"
    inf.mkdir(parents=True)
    (inf / "config.json").write_text(json.dumps(_ARGS))

    t: dict[str, torch.Tensor] = {
        "embed.weight": _bf16(V, D),
        "norm.weight": _bf16(D),
        "head.weight": _bf16(V, D),
        # decoys that must never be read by the resident-weight loader
        "vision.0.weight": _bf16(4, 4),
        "mtp.0.weight": _bf16(4, 4),
    }

    def lin(name, out, inn):
        t[f"{name}.weight"] = _fp8(out, inn)
        t[f"{name}.scale"] = _scale((out + 31) // 32, (inn + 31) // 32)

    for L in range(4):
        a = f"layers.{L}.attn"
        lin(f"{a}.wq_a", QR, D)
        t[f"{a}.q_norm.weight"] = _bf16(QR)
        lin(f"{a}.wq_b", NH * HD, QR)
        lin(f"{a}.wkv", HD, D)
        t[f"{a}.kv_norm.weight"] = _bf16(HD)
        lin(f"{a}.wo_a", OG * OL, OG * OL)
        lin(f"{a}.wo_b", D, OG * OL)
        t[f"{a}.attn_sink"] = torch.randn(NH, dtype=torch.float32)
        if L in (1, 3):  # kv sources
            t[f"{a}.compressor.wkv.weight"] = _bf16(HD, D)
            t[f"{a}.compressor.norm.weight"] = _bf16(HD)
            if L == 1:  # ratio 2 layers carry the gate
                t[f"{a}.compressor.wgate.weight"] = _bf16(HD, D)
        if L in (1, 2, 3):  # index sources
            lin(f"{a}.indexer.wq_b", INH * IHD, QR)
            t[f"{a}.indexer.weights_proj.weight"] = _bf16(INH, D)
            if L in (1, 3):  # only kv sources project index K
                t[f"{a}.indexer.wk.weight"] = _bf16(IHD, HD)
                t[f"{a}.indexer.k_norm.weight"] = _bf16(IHD)
        if L == 1:  # engram layer
            lin(f"layers.{L}.engram.wkv", 16, D)
            t[f"layers.{L}.engram.q_weight"] = _bf16(HCM, D)
            t[f"layers.{L}.engram.k_weight"] = _bf16(HCM, D)
            # the table itself: fp8 rows + e8m0 scales, NOT a resident weight
            t[f"layers.{L}.engram.embed.weight"] = _fp8(101, 32)
            t[f"layers.{L}.engram.embed.scale"] = _scale(101, 1)
        t[f"layers.{L}.attn_norm.weight"] = _bf16(D)
        t[f"layers.{L}.ffn_norm.weight"] = _bf16(D)
        t[f"layers.{L}.ffn.gate.weight"] = _bf16(E, D)
        t[f"layers.{L}.ffn.gate.bias"] = torch.randn(E, dtype=torch.float32)
        t[f"layers.{L}.ffn.gate.bias_vl"] = torch.randn(E, dtype=torch.float32)
        lin(f"layers.{L}.ffn.shared_experts.w1", MI, D)
        lin(f"layers.{L}.ffn.shared_experts.w2", D, MI)
        lin(f"layers.{L}.ffn.shared_experts.w3", MI, D)
        # routed NVFP4 expert (one is enough: the loader must skip them all)
        for proj in ("w1", "w2", "w3"):
            base = f"layers.{L}.ffn.experts.0.{proj}"
            t[f"{base}.weight"] = torch.zeros(MI, D // 2, dtype=torch.uint8)
            t[f"{base}.weight_scale"] = _fp8(MI, D // 16)
            t[f"{base}.weight_scale_2"] = torch.randn((), dtype=torch.float32)
            t[f"{base}.input_scale"] = torch.randn((), dtype=torch.float32)
        mix = (2 + HCM) * HCM
        t[f"layers.{L}.hc_attn_fn"] = torch.randn(mix, HCM * D, dtype=torch.float32)
        t[f"layers.{L}.hc_ffn_fn"] = torch.randn(mix, HCM * D, dtype=torch.float32)
        t[f"layers.{L}.hc_attn_base"] = torch.randn(mix, dtype=torch.float32)
        t[f"layers.{L}.hc_ffn_base"] = torch.randn(mix, dtype=torch.float32)
        t[f"layers.{L}.hc_attn_scale"] = torch.randn(3, dtype=torch.float32)
        t[f"layers.{L}.hc_ffn_scale"] = torch.randn(3, dtype=torch.float32)

    save_file(t, str(folder / "model-00001-of-00001.safetensors"))
    index = {"metadata": {}, "weight_map": {k: "model-00001-of-00001.safetensors" for k in t}}
    (folder / "model.safetensors.index.json").write_text(json.dumps(index))
    return t


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    folder = tmp_path_factory.mktemp("dsv41")
    return folder, _checkpoint(folder)


# --------------------------------------------------------------------------- args


def test_load_args_reads_inference_config_and_overrides(checkpoint):
    folder, _ = checkpoint
    args = load_args(str(folder), max_batch_size=7)
    assert args.n_layers == 4 and args.dim == D and args.vocab_size == V
    assert args.compress_ratios == (0, 2, 0, 1, 0)  # JSON list -> tuple
    assert args.kv_source_layers == (1, 3)
    assert args.index_source_layers == (1, 2, 3)
    assert args.max_batch_size == 7  # override wins
    assert args.max_seq_len == 256  # from the file
    assert args.layer_ratio(1) == 2 and args.layer_ratio(3) == 1 and args.layer_ratio(0) == 0
    assert args.is_kv_source(1) and not args.is_kv_source(2)
    assert args.is_index_source(2) and not args.is_index_source(0)
    assert args.is_engram_layer(1) and not args.is_engram_layer(2)
    # shared compressed cache: most recent source at or before the layer
    assert args.kv_source_for(0) is None
    assert args.kv_source_for(1) == 1 and args.kv_source_for(2) == 1 and args.kv_source_for(3) == 3
    assert args.engram_table_rows(1) == 101


def test_load_args_ignores_unknown_keys(checkpoint):
    folder, _ = checkpoint
    raw = json.loads((folder / "inference" / "config.json").read_text())
    raw["some_future_key"] = 1
    (folder / "inference" / "config.json").write_text(json.dumps(raw))
    args = load_args(str(folder))
    assert not hasattr(args, "some_future_key")


# --------------------------------------------------------------------------- config


def test_parse_config_builds_dsv41_model_config(checkpoint):
    folder, _ = checkpoint
    hf = SimpleNamespace(
        _name_or_path=str(folder),
        max_position_embeddings=1048576,
        model_type="deepseek_v41",
        architectures=["DeepseekV41ForCausalLM"],
    )
    cfg = parse_config(hf)
    assert cfg.model_type == "deepseek_v41"
    assert cfg.num_layers == 4 and cfg.hidden_size == D and cfg.vocab_size == V
    assert cfg.num_experts == E and cfg.num_experts_per_tok == 2
    assert cfg.expert_quant == "nvfp4"
    assert cfg.rms_norm_eps == 1e-20
    assert isinstance(cfg.dsv41_args, DeepseekV41Args)
    (group,) = cfg.attention_groups
    assert isinstance(group, DSV41AttentionGroupConfig)
    assert group.layer_ids == (0, 1, 2, 3) and group.sliding_window == 8
    assert cfg.attn_type_for_layer(0) is AttnType.DSV41
    (spec,) = cfg.kv_cache_group_specs()
    assert spec.attn_type is AttnType.DSV41 and not spec.is_swa and not spec.mla
    assert cfg.rotary_config.scaling["rope_type"] == "yarn"
    assert cfg.rotary_config.scaling["original_max_position_embeddings"] == 4096


# --------------------------------------------------------------------------- weights


def test_iter_weights_names_dtypes_and_exclusions(checkpoint):
    folder, raw = checkpoint
    got = dict(iter_weights(str(folder), "cpu", include_moe_experts=False))

    # top-level + no model.-root rename beyond the ``model.`` prefix
    assert got["model.embed.weight"].dtype == torch.bfloat16
    assert got["model.head.weight"].dtype == torch.bfloat16

    # fp8 linears: weight (fp8) + weight_scale_inv (e8m0)
    assert got["model.layers.0.attn.wq_a.weight"].dtype == FP8
    assert torch.equal(got["model.layers.0.attn.wq_a.weight"], raw["layers.0.attn.wq_a.weight"])
    assert torch.equal(
        got["model.layers.0.attn.wq_a.weight_scale_inv"], raw["layers.0.attn.wq_a.scale"]
    )

    # wo_a dequantized to bf16: value = fp8 * 2^(code-127) over 32x32 blocks
    code = raw["layers.2.attn.wo_a.scale"].view(torch.uint8).to(torch.float32)
    expected = (raw["layers.2.attn.wo_a.weight"].to(torch.float32) * torch.exp2(code - 127.0)).to(torch.bfloat16)
    assert got["model.layers.2.attn.wo_a"].dtype == torch.bfloat16
    assert torch.equal(got["model.layers.2.attn.wo_a"], expected)

    # compressor: wgate only on the ratio-2 layer (L1), not on the ratio-1 layer (L3)
    assert "model.layers.1.attn.compressor.wgate.weight" in got
    assert "model.layers.3.attn.compressor.wgate.weight" not in got
    assert "model.layers.3.attn.compressor.wkv.weight" in got
    assert "model.layers.0.attn.compressor.wkv.weight" not in got  # not a kv source

    # indexer: every index source scores; only kv sources carry K
    assert "model.layers.2.attn.indexer.wq_b.weight" in got  # index-only layer
    assert "model.layers.2.attn.indexer.weights_proj.weight" in got
    assert "model.layers.2.attn.indexer.wk.weight" not in got
    assert "model.layers.1.attn.indexer.wk.weight" in got
    assert "model.layers.1.attn.indexer.k_norm.weight" in got

    # engram: small tensors in, the 384M-row table OUT
    assert "model.layers.1.engram.wkv.weight" in got
    assert "model.layers.1.engram.q_weight" in got
    assert "model.layers.1.engram.embed.weight" not in got
    assert "model.layers.1.engram.embed.scale" not in got

    # router + hc + norms
    assert got["model.layers.0.ffn.gate.bias"].dtype == torch.float32
    assert "model.layers.0.hc_attn_fn" in got and "model.layers.0.hc_ffn_scale" in got

    # routed experts / mtp / vision never surface
    assert not any(".experts." in k for k in got)
    assert not any(k.startswith(("vision", "mtp", "model.vision", "model.mtp")) for k in got)

    # exact name-set equality: nothing dropped, nothing extra.
    # base per layer = 29 (attn 12 + norms 2 + gate 3 + shared 6 + hc 6);
    # L1 adds compressor 3 + indexer 5 + engram 4; L2 adds indexer 3 (no K);
    # L3 adds compressor 2 (no wgate) + indexer 5.
    assert sum(1 for k in got if k.startswith("model.layers.0.")) == 29
    assert len(got) == 3 + 29 + 41 + 32 + 36  # top + L0 + L1 + L2 + L3


def test_iter_weights_rejects_resident_experts(checkpoint):
    folder, _ = checkpoint
    with pytest.raises(ValueError, match="offload"):
        next(iter(iter_weights(str(folder), "cpu", include_moe_experts=True)))


@pytest.fixture
def tp(monkeypatch):
    """Point the loader's TP lookup at a rank for one test.

    ``set_tp_info`` is a process-wide one-shot (it raises once set), so a test that needs two
    topologies patches the name the loader reads instead of the global.
    """
    from freetoken.models.deepseek_v41 import weight as W

    def _set(rank: int, size: int):
        monkeypatch.setattr(W, "try_get_tp_info", lambda: SimpleNamespace(rank=rank, size=size))

    yield _set


def test_iter_weights_slices_the_tp_partitioned_tensors(checkpoint, tp):
    """TP>1: the loader cuts exactly what the model partitions, and nothing else.

    The model declares VocabParallelEmbedding/ParallelLMHead + column/row-parallel linears,
    so handing it the full checkpoint tensor trips the shape assert in layers/base.py -- which
    is how this was found (a TP=2 boot died on ``model.embed.weight`` (129280,5120) vs
    (64640,5120)). The names here are V4.1's, so models.loader.shard_tensor's llama/qwen
    patterns never match.
    """
    folder, full = checkpoint
    tp(0, 2)
    items = dict(iter_weights(str(folder), "cpu", include_moe_experts=False))

    def shape(name):
        return tuple(items[name].shape)

    # vocab: half the rows, and the RIGHT half for this rank
    assert shape("model.embed.weight") == (V // 2, D)
    assert shape("model.head.weight") == (V // 2, D)
    assert torch.equal(items["model.embed.weight"], full["embed.weight"][: V // 2])
    # dim 0 (column parallel): wq_b, the indexer's wq_b, shared w1/w3
    assert shape("model.layers.0.attn.wq_b.weight") == (NH * HD // 2, QR)
    assert shape("model.layers.1.attn.indexer.wq_b.weight") == (INH * IHD // 2, QR)
    assert shape("model.layers.0.ffn.shared_experts.w1.weight") == (MI // 2, D)
    assert shape("model.layers.0.ffn.shared_experts.w3.weight") == (MI // 2, D)
    # dim 0 too, one row per head: the attention sink follows the query heads
    assert shape("model.layers.0.attn.attn_sink") == (NH // 2,)
    assert torch.equal(
        items["model.layers.0.attn.attn_sink"], full["layers.0.attn.attn_sink"][: NH // 2]
    )
    # dim 1 (row parallel): wo_b, shared w2
    assert shape("model.layers.0.attn.wo_b.weight") == (D, OG * OL // 2)
    assert shape("model.layers.0.ffn.shared_experts.w2.weight") == (D, MI // 2)
    # the derived e8m0 companion follows its host tensor's axis (wo_b's own block grid is a
    # single 32-wide column at these tiny dims, so it cannot halve -- the real checkpoint's
    # 160x256 -> 160x128 companion is asserted by the header shape audit, not here)
    assert shape("model.layers.0.attn.wq_b.weight_scale_inv") == ((NH * HD // 2 + 31) // 32, 1)
    # replicated: the low-rank projections, norms, single-head wkv, indexer K
    for name in (
        "model.layers.0.attn.wq_a.weight",
        "model.layers.0.attn.wkv.weight",
        "model.layers.0.attn.q_norm.weight",
        "model.layers.1.attn.indexer.wk.weight",
        "model.layers.1.attn.indexer.k_norm.weight",
        "model.layers.1.attn.indexer.weights_proj.weight",
        "model.layers.0.attn_norm.weight",
    ):
        assert items[name].shape == full[name[len("model.") :]].shape, name
    # wo_a is DERIVED (dequantized from wo_a.weight + its e8m0 scale), so it has no same-named
    # checkpoint key. Both raw sources are cut on dim 0 (whole output groups per rank), so the
    # derived einsum operand keeps every column and only this rank's stacked group rows.
    assert items["model.layers.0.attn.wo_a"].shape == (OG * OL // 2, OG * OL)

    # rank 1 gets the complementary halves
    tp(1, 2)
    rank1 = dict(iter_weights(str(folder), "cpu", include_moe_experts=False))
    assert rank1["model.embed.weight"].shape == (V // 2, D)
    assert torch.equal(rank1["model.embed.weight"], full["embed.weight"][V // 2 :])
    assert torch.equal(
        rank1["model.layers.0.attn.wo_b.weight"],
        full["layers.0.attn.wo_b.weight"][:, OG * OL // 2 :],
    )
    assert torch.equal(
        rank1["model.layers.0.ffn.shared_experts.w2.weight"],
        full["layers.0.ffn.shared_experts.w2.weight"][:, MI // 2 :],
    )
    assert torch.equal(
        rank1["model.layers.0.attn.attn_sink"], full["layers.0.attn.attn_sink"][NH // 2 :]
    )


def test_tp_shard_is_a_no_op_without_tp_info(checkpoint):
    """A standalone ``iter_weights`` (conversion, unit tests) must see full tensors.

    ``get_tp_info`` RAISES when nothing set it, so the loader asks ``try_get_tp_info``.
    """
    from freetoken.models.deepseek_v41 import weight as W

    folder, full = checkpoint
    items = dict(iter_weights(str(folder), "cpu", include_moe_experts=False))
    assert items["model.embed.weight"].shape == (V, D)
    assert torch.equal(items["model.embed.weight"], full["embed.weight"])
    assert items["model.layers.0.attn.wo_b.weight"].shape == (D, OG * OL)
    # and directly: no rank information at all
    assert W._tp_shard("embed.weight", full["embed.weight"]) is full["embed.weight"]


# --------------------------------------------------------------------------- expert spec


def test_nvfp4_expert_spec_matches_checkpoint_keys(checkpoint):
    folder, _ = checkpoint
    spec = nvfp4_expert_spec(str(folder), None)
    m = spec.key_pattern.match("layers.12.ffn.experts.5.w2.weight_scale_2")
    assert m and m.group("layer") == "12" and m.group("expert") == "5"
    assert m.group("proj") == "w2" and m.group("kind") == "weight_scale_2"
    assert spec.proj_to_role == {"w1": "gate", "w3": "up", "w2": "down"}
    assert spec.layer_to_bank(12, None) == 12
    assert spec.kind_map["weight_scale_2"] == "weight_scale_2"
    assert spec.global_reciprocal is False
    assert not spec.key_pattern.match("layers.0.ffn.shared_experts.w1.weight")
    assert not spec.key_pattern.match("mtp.0.experts.0.w1.weight")


# --------------------------------------------------------------------- the real checkpoint

REAL_CHECKPOINT = os.environ.get("FT_V41_CHECKPOINT", "/mnt/nvme/models/DeepSeek-V4.1-Flash-NVFP4")


@pytest.mark.skipif(os.path.isdir(REAL_CHECKPOINT) is False, reason="needs the V4.1 checkpoint")
def test_the_loader_and_the_model_agree_with_the_real_checkpoint(monkeypatch):
    """The synthetic checkpoint proves the naming rules; only the real one proves they cover it.

    Everything here runs off the shard HEADERS plus a stubbed reader, so it needs neither 527 GB
    of RAM nor a GPU: it answers "does the loader ask for every key the release ships, are all its
    names real, and does the model declare exactly what it fills?".
    """
    from dataclasses import replace

    import glob

    from freetoken.distributed.info import get_tp_info, set_tp_info
    from freetoken.layers.quantization import set_quant_config
    from freetoken.models.deepseek_v41 import weight as W
    from freetoken.models.deepseek_v41.model import DeepseekV41ForCausalLM
    from freetoken.models.register import checkpoint_quant_config, get_model_spec
    from freetoken.utils import cached_load_hf_config

    shipped: set[str] = set()
    for shard in sorted(glob.glob(os.path.join(REAL_CHECKPOINT, "*.safetensors"))):
        with open(shard, "rb") as fh:
            n = int.from_bytes(fh.read(8), "little")
            shipped |= {k for k in json.loads(fh.read(n)) if k != "__metadata__"}

    asked: dict[str, int] = {}

    class _StubReader:
        def __init__(self, folder, weight_map, device):
            self._weight_map = weight_map

        def has(self, name):
            return name in self._weight_map

        def get(self, name):
            asked[name] = asked.get(name, 0) + 1
            return torch.zeros(1)

        def close(self):
            pass

    monkeypatch.setattr(W, "_ShardReader", _StubReader)
    monkeypatch.setattr(W, "_dequant_fp8_block", lambda *a, **k: torch.zeros(1))
    filled = [name for name, _ in iter_weights(REAL_CHECKPOINT, "cpu", include_moe_experts=False)]

    assert not [n for n in asked if n not in shipped], "the loader asks for keys the release lacks"
    # everything else the release ships is out of this port's scope: MTP/DSpark and the vision
    # tower are deferred, routed experts come from the offload banks, and the two 94 GiB engram
    # tables are served off disk by EngramTier.
    unread = shipped - set(asked) - {"image_start", "image_end", "image_newline"}
    assert not [k for k in unread if not _deferred(k)], f"unconsumed keys: {sorted(unread)[:10]}"

    # now the other direction: build the model exactly as the engine does (quant included) and
    # require the loader's names to be precisely its parameters
    try:
        get_tp_info()
    except RuntimeError:
        set_tp_info(rank=0, size=1)
    hf_config = cached_load_hf_config(REAL_CHECKPOINT)
    spec = get_model_spec(hf_config.architectures[0])
    quant = checkpoint_quant_config(REAL_CHECKPOINT, hf_config, spec)
    set_quant_config(quant)
    config = replace(parse_config(hf_config), quant=quant)
    with torch.device("meta"):
        model = DeepseekV41ForCausalLM(config)

    declared = set(model.state_dict())
    assert set(filled) - declared == set(), "the loader names parameters the model does not declare"
    assert declared - set(filled) == set(), "the model declares parameters the loader never fills"


def _deferred(key: str) -> bool:
    """Release keys this port deliberately does not consume: MTP/DSpark and the vision tower are
    deferred, routed experts come from the offload banks, and the two 94 GiB engram tables are
    served off disk by EngramTier."""
    return key.startswith(("mtp.", "vision.", "aligner.")) or (
        ".ffn.experts." in key or ".engram.embed." in key
    )

