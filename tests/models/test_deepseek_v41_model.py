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


def _model(monkeypatch, device) -> model_mod.Transformer:
    from freetoken.distributed.info import get_tp_info, set_tp_info
    from freetoken.models.deepseek_v41.engram import ResidentEngramTable

    try:  # the tp info is process-global: the first test may already have set it
        get_tp_info()
    except RuntimeError:
        set_tp_info(rank=0, size=1)
    monkeypatch.setattr(model_mod, "MoE", lambda *a, **k: _StubFFN(DIM))
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
