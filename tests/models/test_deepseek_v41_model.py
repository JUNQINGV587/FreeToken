"""The V4.1 transformer wiring: embed -> hc copies -> blocks -> collapse -> logits.

This is the M2 level: not one number against the reference (that is what the layer tests do) but
the plumbing that ties the layers together -- the deferred pre-mix (a block's attention consumes the
PREVIOUS block's FFN mix), the collapse onto the head, the engram injection into the hc residual
before its layer, and the per-sequence caches (window ring, compressor carry, indexer keys, n-gram
history) advancing in lockstep with ``start_pos``.

The routed MoE is stubbed: it has its own tests, and the engine's expert path is the offload cache
that arrives with M5. What is under test is that a pass split into "prefill then one token at a
time" lands where a single pass lands, which only holds if every cache above is threaded correctly.
"""

from __future__ import annotations

import json
import os
import struct
from types import SimpleNamespace

import dataclasses

import pytest
import torch

from freetoken.models.deepseek_v41.args import DeepseekV41Args
from freetoken.models.deepseek_v41 import model as model_mod
from freetoken.utils.torch_utils import torch_dtype

DIM = 64
VOCAB = 256
N_LAYERS = 4
COMPRESS_RATIOS = (0, 0, 2, 1)
KV_SOURCES = (2, 3)
INDEX_SOURCES = (2, 3)


class _FakeTokenizer:
    """Enough of a transformers tokenizer for ``build_compressed_token_map``: every id decodes to
    its own unique key, so the compressed map is exactly ``size`` entries long."""

    def __init__(self, size: int):
        self._size = size
        self.backend_tokenizer = self

    def __len__(self) -> int:
        return self._size

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        return f"tok{ids[0]}"

    def id_to_token(self, i: int) -> str:
        return f"tok{i}"


class _StubFFN:
    """Deterministic stand-in for the routed MoE (its own tests cover the experts)."""

    def __init__(self, dim: int):
        self.scale = torch.full((dim,), 0.5, dtype=torch.bfloat16)

    def forward(self, x, image_mask=None):
        return x * self.scale


def _args() -> DeepseekV41Args:
    return DeepseekV41Args(
        dim=DIM,
        n_layers=N_LAYERS,
        vocab_size=VOCAB,
        n_heads=4,
        head_dim=64,
        rope_head_dim=16,
        q_lora_rank=32,
        o_lora_rank=32,
        o_groups=2,
        n_routed_experts=4,
        n_activated_experts=2,
        moe_inter_dim=32,
        n_shared_experts=1,
        hc_mult=2,
        compress_ratios=COMPRESS_RATIOS,
        kv_source_layers=KV_SOURCES,
        index_source_layers=INDEX_SOURCES,
        candidate_source_layer=3,
        index_n_heads=2,
        index_head_dim=32,
        index_topk=8,
        candidate_topk_blocks=4,
        candidate_block_size=4,
        window_size=8,
        max_batch_size=2,
        max_seq_len=64,
        norm_eps=1e-6,
        hc_eps=1e-6,
        rope_theta=10000.0,
        compress_rope_theta=160000.0,
        original_seq_len=0,
        rope_factor=1.0,
        beta_fast=32.0,
        beta_slow=1.0,
        engram_layer_ids=(1,),
        engram_num_embeddings=(VOCAB,),
        engram_max_ngram_size=4,
        engram_vocab_size=64,
        engram_n_heads=2,
        engram_head_dim=32,
        engram_pad_id=2,
        engram_compressed_vocab_size=VOCAB,  # every fake token normalizes to its own key
    )


def _band_args() -> DeepseekV41Args:
    """A stack whose layer 2 is a CONSUMER: layer 1 is the band's only kv/index source, so
    layers 2 (ratio 2) and 3 (ratio 1) have no compressor and no indexer of their own."""
    return dataclasses.replace(
        _args(),
        compress_ratios=(0, 2, 2, 1),
        kv_source_layers=(1,),
        index_source_layers=(1,),
        candidate_source_layer=1,
    )


def _model(monkeypatch, device, args: DeepseekV41Args | None = None) -> model_mod.Transformer:
    from freetoken.distributed.info import get_tp_info, set_tp_info
    from freetoken.models.deepseek_v41.engram import ResidentEngramTable

    try:  # the tp info is process-global: the first test may already have set it
        get_tp_info()
    except RuntimeError:
        set_tp_info(rank=0, size=1)
    monkeypatch.setattr(model_mod, "MoE", lambda *a, **k: _StubFFN(DIM))
    if args is None:
        args = _args()
    rows = sum(args.engram_num_embeddings)
    table = ResidentEngramTable(
        (torch.randn(rows, args.engram_head_dim) * 0.1).to(torch.float8_e4m3fn),
        torch.full((rows, args.engram_head_dim // 32), 127, dtype=torch.uint8).view(
            torch.float8_e8m0fnu
        ),
    ).to(device)
    # `torch.device` as a context sets the default device, which is what puts the embedding's
    # vocab range on the GPU (the engine does this for every model it builds)
    with torch_dtype(torch.bfloat16), torch.device(device):
        model = model_mod.Transformer(
            args, None, prefix="model", tokenizer=_FakeTokenizer(VOCAB), engram_table=table
        )
    # random but deterministic weights: nothing here is compared against the reference, only the
    # model against ITSELF under a different pass split
    torch.manual_seed(1234)
    for name, tensor in model.state_dict().items():
        if tensor.dtype.is_floating_point:
            tensor.copy_(torch.randn_like(tensor.float()).to(tensor.dtype) * 0.1)
    model.bind(device)
    return model


def _logits(model, ids, start_pos=0):
    out = model.forward(ids, start_pos)
    return out.float()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_the_head_returns_last_position_logits_and_the_full_grid_on_request(monkeypatch):
    device = torch.device("cuda")
    model = _model(monkeypatch, device)
    ids = torch.randint(0, VOCAB, (2, 6), device=device)
    last = model.forward(ids)
    assert last.shape == (2, VOCAB)
    full = model.forward(ids, full_logits=True)
    assert full.shape == (2, 6, VOCAB)
    # the default is exactly the last row of the full grid (the head slices, it does not re-run)
    assert torch.equal(last, full[:, -1])


def test_the_head_projects_the_rows_the_sampler_reads():
    """A ragged prefill names its rows; decode and the logits oracle keep their own defaults.

    ``h`` arrives as [1, T, dim]; the head is the vocab-sized GEMM (all-gathered at TP > 1), so the
    paged prefill must project each request's last row rather than the whole chunk.
    """
    h = torch.arange(12, dtype=torch.float32).reshape(1, 4, 3)
    flat = h.reshape(-1, 3)

    only_last = model_mod.Transformer._head_rows(h, None, False)
    assert only_last.shape == (1, 3)
    assert torch.equal(only_last[0], h[0, -1])

    grid = model_mod.Transformer._head_rows(h, None, True)
    assert grid.shape == (4, 3)
    assert torch.equal(grid, flat)

    keep = torch.tensor([3, 1], dtype=torch.long)
    picked = model_mod.Transformer._head_rows(h, keep, True)
    assert torch.equal(picked, flat.index_select(0, keep))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_a_split_pass_lands_where_a_single_pass_lands(monkeypatch):
    """Prefill then token-at-a-time must equal one pass, and ``reset`` must restore the start."""
    device = torch.device("cuda")
    model = _model(monkeypatch, device)
    ids = torch.randint(0, VOCAB, (2, 8), device=device)

    one_shot = _logits(model, ids)
    model.reset()
    _logits(model, ids[:, :5])
    for step in range(5, 8):
        piece = _logits(model, ids[:, step : step + 1], start_pos=step)
    # the decode step runs M=1 kernels while the prefill ran M=32 ones, so the accumulation order
    # differs: bf16, not bit-exact
    assert torch.allclose(piece, one_shot, rtol=2e-2, atol=2e-2), (
        piece - one_shot
    ).abs().max().item()

    model.reset()
    assert torch.equal(_logits(model, ids), one_shot), "reset did not restore the initial state"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_the_engram_layer_is_the_only_one_and_reaches_the_output(monkeypatch):
    device = torch.device("cuda")
    model = _model(monkeypatch, device)
    engram_layers = [
        i for i, layer in enumerate(model.layers.op_list) if layer.engram is not None
    ]
    assert engram_layers == [1] and model._engram_hash is not None
    assert model._engram_hash.token_map.numel() == VOCAB
    assert model.layers.op_list[1].engram.layer_hash_index == 0

    ids = torch.randint(0, VOCAB, (2, 6), device=device)
    with_engram = _logits(model, ids)
    model.reset()
    table = model._engram_table
    saved = table.weight.clone()
    with torch.no_grad():
        table.weight.zero_()
    try:
        without = _logits(model, ids)
    finally:
        with torch.no_grad():
            table.weight.copy_(saved)
    assert not torch.allclose(with_engram, without, rtol=1e-3, atol=1e-3), (
        "the engram contribution never reached the logits"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_one_band_runtime_is_shared_by_a_source_and_the_consumers_that_read_it(monkeypatch):
    """The reference's ``shared_attn`` is module-global, so the whole stack shares one runtime.

    Per-layer instances silently break the band: a consumer has no compressor and no indexer, so
    its compressed half *is* whatever its source published -- with a private runtime that is None.
    """
    device = torch.device("cuda")
    model = _model(monkeypatch, device, args=_band_args())
    attns = [layer.attn for layer in model.layers.op_list]

    assert len({id(attn._runtime) for attn in attns}) == 1
    source, consumer, ratio1_consumer = attns[1], attns[2], attns[3]
    assert source.is_kv_source and source.compressor is not None and source.indexer is not None
    assert consumer.kv_source_layer == 1 and not consumer.is_kv_source
    assert consumer.compressor is None and consumer.indexer is None
    assert ratio1_consumer.kv_source_layer == 1 and ratio1_consumer.indexer is None
    assert consumer._runtime is source._runtime and ratio1_consumer._runtime is source._runtime

    ids = torch.randint(0, VOCAB, (1, 6), device=device)
    logits = _logits(model, ids)
    assert torch.isfinite(logits).all()
    # The source published the compressed half its consumers read (the consumers' own is derived,
    # not stored): without the shared runtime this is None and the forward above raises.
    assert source._runtime.topk_idxs is not None
    assert source._runtime.compress_kv is not None

    # Positive control: the same stack built with a runtime per layer (what a missing ``runtime=``
    # used to give) cannot run at all -- a consumer has no compressor and no indexer, so its
    # private runtime leaves both the compressed pool and the index list at None.
    original_block = model_mod.Block
    monkeypatch.setattr(
        model_mod, "Block", lambda *a, **kw: original_block(*a, **{**kw, "runtime": None})
    )
    unshared = _model(monkeypatch, device, args=_band_args())
    with pytest.raises(TypeError, match="NoneType"):
        _logits(unshared, ids)


def _set_tp_info() -> None:
    from freetoken.distributed.info import get_tp_info, set_tp_info

    try:  # the tp info is process-global: another test may already have set it
        get_tp_info()
    except RuntimeError:
        set_tp_info(rank=0, size=1)


def _write_engram_checkpoint(folder: str, args: DeepseekV41Args) -> str:
    """A checkpoint directory holding only what the adapter reads OFF DISK: ``tokenizer.json`` and
    the engram shards its index points at.

    The tier's own file-format cases -- unaligned offsets, two layers with different row counts,
    out-of-range rows -- live in ``test_deepseek_v41_engram_tier.py``; all this one needs is shards
    the same reader can walk, with row-dependent values so a misread row would show up.
    """
    from tokenizers import Tokenizer, models

    tokenizer = Tokenizer(models.WordLevel({f"tok{i}": i for i in range(VOCAB)}, unk_token=None))
    tokenizer.save(os.path.join(folder, "tokenizer.json"))

    rows, dim = sum(args.engram_num_embeddings), args.engram_head_dim
    # e4m3 bit patterns around 0.5 (exponent 0110): valid, finite, and different per row. The
    # all-ones exponent would be a NaN.
    raw = torch.arange(rows * dim).reshape(rows, dim) % 8
    weight = (raw + 0x30).to(torch.uint8)
    scale = torch.full((rows, dim // 32), 127, dtype=torch.uint8)  # E8M0 127 is 2**0

    layer_id = args.engram_layer_ids[0]
    tensors = [
        (f"layers.{layer_id}.engram.embed.weight", "F8_E4M3", weight, weight.numpy().tobytes()),
        (f"layers.{layer_id}.engram.embed.scale", "F8_E8M0", scale, scale.numpy().tobytes()),
    ]
    header: dict[str, dict] = {}
    blob = bytearray()
    for name, dtype, tensor, data in tensors:
        blob += b"\0" * ((8 - len(blob) % 8) % 8)
        header[name] = {
            "dtype": dtype,
            "shape": list(tensor.shape),
            "data_offsets": [len(blob), len(blob) + len(data)],
        }
        blob += data
    raw_header = json.dumps(header, separators=(",", ":")).encode()
    raw_header += b" " * ((8 - len(raw_header) % 8) % 8)
    shard = "model-00001-of-00001.safetensors"
    with open(os.path.join(folder, shard), "wb") as handle:
        handle.write(struct.pack("<Q", len(raw_header)) + raw_header + bytes(blob))
    with open(os.path.join(folder, "model.safetensors.index.json"), "w") as handle:
        json.dump(
            {
                "metadata": {"total_size": len(blob)},
                "weight_map": {tensor[0]: shard for tensor in tensors},
            },
            handle,
        )
    return folder


def _adapter_config(folder: str | None, args: DeepseekV41Args) -> SimpleNamespace:
    return SimpleNamespace(
        dsv41_args=args,
        quant=None,
        moe_strategy="offload",
        decode_target="gpu",
        checkpoint_path=folder,
        engram_tokenizer_path=None,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_the_adapter_reads_its_token_map_and_engram_rows_off_the_checkpoint_dir(
    monkeypatch, tmp_path
):
    """The engine hands this class a config, not a tokenizer or a table: both have to be found in
    the checkpoint directory, and the 94 GiB per layer table has to stay on disk while it serves."""
    from freetoken.models.deepseek_v41.engram import ResidentEngramTable
    from freetoken.models.deepseek_v41.engram_tier import EngramTable

    device = torch.device("cuda")
    _set_tp_info()
    args = _args()
    folder = _write_engram_checkpoint(str(tmp_path), args)
    monkeypatch.setattr(model_mod, "MoE", lambda *a, **k: _StubFFN(DIM))
    with torch_dtype(torch.bfloat16), torch.device(device):
        adapter = model_mod.DeepseekV41ForCausalLM(_adapter_config(folder, args))
    model = adapter.model

    # the token map came out of tokenizer.json (this fake vocabulary has 256 distinct keys)
    assert model._engram_hash is not None
    assert model._engram_hash.token_map.numel() == VOCAB
    assert [block.engram.layer_id for block in adapter.engram_layers()] == [1]

    torch.manual_seed(7)
    for tensor in model.state_dict().values():
        if tensor.dtype.is_floating_point:
            tensor.copy_(torch.randn_like(tensor.float()).to(tensor.dtype) * 0.1)
    model.bind(device)

    tier = adapter.bind_engram_tier(device, use_io_uring=False)
    assert tier is not None and tier.device == device
    assert tier.rows_of(0) == VOCAB and tier.dim == args.engram_head_dim
    # each layer gets a view of the shared tier, and only the layer that has an engram block does
    table = model.layers.op_list[1].engram.table
    assert isinstance(table, EngramTable) and table.tier is tier and table.layer_id == 1
    assert table.num_embeddings == VOCAB
    assert model.layers.op_list[0].engram is None

    ids = torch.randint(0, VOCAB, (2, 5), device=device)
    served = model.forward(ids).float()
    assert torch.isfinite(served).all()

    # and the rows read off disk are what reaches the residual: an all-zero table has to move the
    # logits, or the lookup never made it through `gather`
    blank = ResidentEngramTable(
        torch.zeros(VOCAB, args.engram_head_dim, dtype=torch.float8_e4m3fn),
        torch.full((VOCAB, args.engram_head_dim // 32), 127, dtype=torch.uint8).view(
            torch.float8_e8m0fnu
        ),
    ).to(device)
    model.layers.op_list[1].engram.bind(table=blank, device=device)
    model.reset()
    without = model.forward(ids).float()
    assert not torch.allclose(served, without, rtol=1e-3, atol=1e-3), (
        "the rows served off NVMe never reached the logits"
    )


def test_a_model_with_engram_layers_needs_the_tokenizer_its_hash_is_built_from():
    """Silently hashing with a different tokenizer would renumber every token, so a config that
    names engram layers without a tokenizer is an error -- and one without them needs none."""
    config = _adapter_config(None, _args())
    with pytest.raises(RuntimeError, match="tokenizer.json"):
        model_mod.resolve_engram_tokenizer(config)

    plain = SimpleNamespace(dsv41_args=SimpleNamespace(engram_layer_ids=()), checkpoint_path=None)
    assert model_mod.resolve_engram_tokenizer(plain) is None


@pytest.mark.skipif(
    not os.path.isdir(os.environ.get("FT_V41_CHECKPOINT", "/mnt/nvme/models/DeepSeek-V4.1-Flash-NVFP4")),
    reason="needs the V4.1 checkpoint",
)
def test_parse_config_hands_the_loader_the_checkpoint_directory():
    from freetoken.models.deepseek_v41.config import parse_config
    from freetoken.models.deepseek_v41.engram import build_compressed_token_map

    folder = os.environ.get("FT_V41_CHECKPOINT", "/mnt/nvme/models/DeepSeek-V4.1-Flash-NVFP4")
    config = parse_config(SimpleNamespace(_name_or_path=folder))
    assert config.checkpoint_path == folder
    # owner-local EP (--tensor-parallel-size 2 --moe-ep-size 2): the engine's gate reads this
    # declaration instead of whitelisting the model_type (engine._check_owner_ep_model), and the
    # experts are the NVFP4 whole-expert banks that gate also requires.
    assert config.owner_ep is True
    assert config.expert_quant == "nvfp4"
    # the real tokenizer.json is what the compressed map's size in the checkpoint was computed
    # from -- a different tokenizer would not reproduce it
    assert config.dsv41_args.engram_layer_ids == (1, 14)
    _, size = build_compressed_token_map(model_mod.resolve_engram_tokenizer(config))
    assert size == config.dsv41_args.engram_compressed_vocab_size
