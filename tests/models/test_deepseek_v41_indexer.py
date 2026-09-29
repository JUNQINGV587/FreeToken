"""CSA2 indexer vs the reference ``Indexer`` (M3, second piece).

``indexer_ref.pt`` is produced by ``research/v41-oracle/indexer_oracle.py``: the OFFICIAL
``inference/model.py:495`` class, three instances driven in the order the model drives them --

  * layer 2  (ratio 2) turns its compressor latent into the index keys its whole band reads,
  * layer 20 (ratio 1) does the same *and* publishes the candidate mask,
  * layer 24 (ratio 1, no K of its own) scores inside that mask and picks its own top-k.

The interleaving is the point: ``shared_attn.index_k``/``candidates`` are single slots written by the
source and read by later layers, so replaying layer 24 after layer 20 has already run its own decode
steps would read the wrong cache length. Weights and activations are bf16-exact, so the two
implementations execute the same arithmetic.

    FT_V41_INDEXER_DUMP  default /oracle/indexer_ref.pt
"""

from __future__ import annotations

import os

import pytest
import torch

from freetoken.models.deepseek_v41.args import DeepseekV41Args
from freetoken.models.deepseek_v41.indexer import Indexer, SharedAttentionRuntime
from freetoken.models.deepseek_v41.ops import get_freqs_cis

DUMP = os.environ.get("FT_V41_INDEXER_DUMP", "/oracle/indexer_ref.pt")

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and os.path.isfile(DUMP)),
    reason="needs CUDA + the indexer oracle dump",
)

_LAYERS = {"2": 2, "20": 20, "24": 24}


def _reference():
    return torch.load(DUMP, map_location="cpu", weights_only=False)


def _args(root: dict) -> DeepseekV41Args:
    cfg = root["config"]
    return DeepseekV41Args(
        dim=cfg["dim"],
        q_lora_rank=cfg["q_lora_rank"],
        head_dim=cfg["head_dim"],
        index_n_heads=cfg["index_n_heads"],
        index_head_dim=cfg["index_head_dim"],
        rope_head_dim=cfg["rope_head_dim"],
        index_topk=cfg["index_topk"],
        norm_eps=cfg["norm_eps"],
        max_batch_size=cfg["max_batch_size"],
        max_seq_len=cfg["max_seq_len"],
        window_size=cfg["window_size"],
        compress_ratios=tuple(root["ratios"]),
        kv_source_layers=tuple(cfg["kv_source_layers"]),
        candidate_source_layer=cfg["candidate_source_layer"],
        candidate_topk_blocks=cfg["candidate_topk_blocks"],
        candidate_block_size=cfg["candidate_block_size"],
        # the dump builds its rope table with YaRN off (original_seq_len=0, factor=1)
        original_seq_len=root["freqs"]["original_seq_len"],
        compress_rope_theta=root["freqs"]["base"],
        rope_factor=root["freqs"]["factor"],
        beta_fast=root["freqs"]["beta_fast"],
        beta_slow=root["freqs"]["beta_slow"],
    )


def _build(root: dict, args: DeepseekV41Args, layer_id: int, runtime) -> Indexer:
    from freetoken.distributed import set_tp_info, try_get_tp_info

    from freetoken.utils.torch_utils import torch_dtype

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    with torch_dtype(torch.bfloat16):
        idx = Indexer(
            layer_id, args, prefix=f"model.layers.{layer_id}.attn.indexer", runtime=runtime
        )
    target = idx.state_dict()
    state = {
        k: v.to(device="cuda", dtype=target[k].dtype)
        for k, v in root["weights"][str(layer_id)].items()
    }
    idx.load_state_dict(state)
    idx.bind(torch.device("cuda"))
    return idx


def _errors(actual, expected):
    a, b = actual.float(), expected.float()
    return float((a - b).abs().max()), float((a - b).abs().max() / b.abs().max().clamp_min(1e-6))


def test_the_rope_table_matches_the_reference():
    """The compressed rope (theta 160000) has to be the reference's table, not merely similar."""
    root = _reference()
    args = _args(root)
    freqs = get_freqs_cis(
        args.rope_head_dim, args.max_seq_len, args.original_seq_len, args.compress_rope_theta,
        args.rope_factor, args.beta_fast, args.beta_slow, torch.device("cuda"),
    )
    assert torch.equal(freqs.cpu(), root["freqs_cis"])


def test_indexers_match_the_reference_dump():
    root = _reference()
    args = _args(root)
    runtime = SharedAttentionRuntime()
    indexers = {tag: _build(root, args, layer_id, runtime) for tag, layer_id in _LAYERS.items()}
    steps = root["steps"]

    # the reference ran 2 -> 20 -> 24 within each step; the shared slots make the order load-bearing
    for i in range(len(steps["20"])):
        for tag in ("2", "20", "24"):
            step = steps[tag][i]
            idx = indexers[tag]
            x = step["x"].to("cuda")
            qr = step["qr"].to("cuda")
            latent = None if step["latent"] is None else step["latent"].to("cuda")
            got = idx.forward(x, qr, latent, step["start_pos"], step["offset"])
            want = step["idxs"].to("cuda")
            assert got.shape == want.shape, (tag, step["tag"], tuple(got.shape), tuple(want.shape))
            assert torch.equal(got, want), (
                f"layer {tag} {step['tag']}: idxs differ at "
                f"{(got != want).nonzero()[:8].tolist()}"
            )
            if idx.owns_k:
                rows = (step["start_pos"] + x.size(1)) // idx.ratio
                cache = idx._k_cache[: x.size(0), :rows]
                want_cache = step["k"].to("cuda")
                assert torch.equal(cache, want_cache), (
                    f"layer {tag} {step['tag']}: index-K cache differs -- "
                    f"abs={_errors(cache, want_cache)[0]:.3e}"
                )
            if step.get("candidates") is not None:
                assert torch.equal(runtime.candidates, step["candidates"].to("cuda")), (
                    f"layer {tag} {step['tag']}: candidate mask differs"
                )


def test_the_consumer_scores_inside_the_published_mask():
    """Layer 24 has no K of its own: its indices can only come from layer 20's cache + mask."""
    root = _reference()
    args = _args(root)
    runtime = SharedAttentionRuntime()
    idx24 = _build(root, args, 24, runtime)
    assert not idx24.owns_k
    with pytest.raises(TypeError):  # nothing published yet: index_k is None
        idx24.forward(
            torch.zeros(2, 8, args.dim, dtype=torch.bfloat16, device="cuda"),
            torch.zeros(2, 8, args.q_lora_rank, dtype=torch.bfloat16, device="cuda"),
            None, 0, 8,
        )

    idx20 = _build(root, args, 20, runtime)
    step = root["steps"]["20"][0]
    idx20.forward(
        step["x"].to("cuda"), step["qr"].to("cuda"), step["latent"].to("cuda"),
        step["start_pos"], step["offset"],
    )
    assert runtime.candidates is not None and runtime.index_k is not None
    con = root["steps"]["24"][0]
    got = idx24.forward(
        con["x"].to("cuda"), con["qr"].to("cuda"), None, con["start_pos"], con["offset"]
    )
    assert torch.equal(got, con["idxs"].to("cuda"))
