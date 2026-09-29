"""Engram layer semantics against the reference dump.

``engram_ref.pt`` comes from ``research/v41-oracle/engram_oracle.py``: the OFFICIAL
``inference/engram.py`` + ``model.py``'s ``Engram``, on a shrunken geometry but the checkpoint's
own tokenizer -- which is the whole point, because the compressed token map (99092) and every hash
multiplier derive from it. A synthetic tokenizer would agree with itself and prove nothing.

What this pins down: the prime bucket layout (handed out in order, never reused -- the count feeds
the multipliers), the compressed map including the U+FFFD partial-byte branch, the rolling XOR over
lookbacks with the pad/dead substitutes, the prime-range offsets, the fp8/e8m0 row dequantization,
and the signed-sqrt gate against the hyper-connection residual.

Needs the checkpoint (for the tokenizer) and the dump, so it skips unless both are present:
    FT_V41_CHECKPOINT   default /mnt/nvme/models/DeepSeek-V4.1-Flash-NVFP4
    FT_V41_ENGRAM_DUMP  default /oracle/engram_ref.pt
"""

from __future__ import annotations

import os

import pytest
import torch

from freetoken.models.deepseek_v41.args import DeepseekV41Args
from freetoken.models.deepseek_v41.engram import (
    Engram,
    EngramLayout,
    NgramHashState,
    ResidentEngramTable,
    tokenizer_of,
)

CKPT = os.environ.get("FT_V41_CHECKPOINT", "/mnt/nvme/models/DeepSeek-V4.1-Flash-NVFP4")
DUMP = os.environ.get("FT_V41_ENGRAM_DUMP", "/oracle/engram_ref.pt")

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and os.path.isdir(CKPT) and os.path.isfile(DUMP)),
    reason="needs CUDA + the V4.1 checkpoint + the engram dump",
)


def _args(root) -> DeepseekV41Args:
    return DeepseekV41Args(**{k: v for k, v in root["args"].items()})


def _layout_and_state(root, args, device):
    layout = EngramLayout.from_args(args)
    state = NgramHashState(args, layout, tokenizer_of(CKPT))
    state.bind(device)
    return layout, state


def _engram(layer_id: int, args, layout, root, device) -> Engram:
    from freetoken.utils.torch_utils import torch_dtype

    with torch_dtype(torch.bfloat16):
        eng = Engram(
            layer_id,
            args,
            layout,
            quant_config=None,
            prefix=f"model.layers.{layer_id}.engram",
            table=ResidentEngramTable(
                root["tables"][layer_id]["weight"], root["tables"][layer_id]["scale"]
            ).to(device),
        )
    w = root["weights"][layer_id]
    eng.load_state_dict(
        {
            "wkv.weight": w["wkv"].to(eng.wkv.weight.dtype).to(device),
            "q_weight": w["q_weight"].to(eng.q_weight.dtype).to(device),
            "k_weight": w["k_weight"].to(eng.k_weight.dtype).to(device),
        }
    )
    return eng


def test_the_prime_layout_is_handed_out_in_order_and_never_reused():
    root = torch.load(DUMP, weights_only=False)
    args = _args(root)
    layout = EngramLayout.from_args(args)
    assert layout.primes == root["layout"]["primes"]
    flat = [p for layer in layout.primes for ngram in layer for p in ngram]
    assert len(flat) == len(set(flat)), "a prime was reused across buckets"
    assert all(b > a for a, b in zip([args.engram_vocab_size - 1, *flat[:-1]], flat))
    # the bucket ranges are contiguous per (layer, n-gram size), which is what `+ offsets` assumes
    offsets = layout.bucket_offsets()
    assert offsets.shape == (len(layout.layer_ids), layout.n_hash_cols)


def test_hashes_and_the_gated_lookup_match_the_reference():
    root = torch.load(DUMP, weights_only=False)
    args = _args(root)
    device = torch.device("cuda")
    layout, state = _layout_and_state(root, args, device)
    engrams = {lid: _engram(lid, args, layout, root, device) for lid in layout.layer_ids}

    worst = 0.0
    for rec in root["records"]:
        ids = rec["input_ids"].to(device)
        mask = rec.get("token_mask")
        mask = mask.to(device) if mask is not None else None
        x = rec["x"].to(device)
        hashes = state.forward(ids, rec["start_pos"], mask)
        assert torch.equal(hashes, rec["hashes"].to(device)), f"hashes at {rec['start_pos']}"
        for lid in layout.layer_ids:
            h = hashes[:, :, layout.layer_ids.index(lid), :]
            out = engrams[lid].forward(x, h, mask)
            ref = rec[f"out_{lid}"].to(device)
            rel = (out.float() - ref.float()).abs().max().item() / ref.float().abs().max().item()
            worst = max(worst, rel)
            assert rel < 1e-4, (rec["start_pos"], lid, rel)
    print(f"engram worst relative error over all steps: {worst:.3e}")


def test_a_position_with_no_history_gets_the_pad_row_not_a_wrapped_lookback():
    """The decode steps carry the cache forward, so a wrong substitute shows up as a wrong hash."""
    root = torch.load(DUMP, weights_only=False)
    args = _args(root)
    device = torch.device("cuda")
    layout, state = _layout_and_state(root, args, device)
    prefill, decode = root["records"][0], root["records"][1]
    assert prefill["start_pos"] == 0 and decode["start_pos"] == prefill["input_ids"].size(1)
    # a fresh state that sees ONLY the decode token must not reproduce the carried hash: the
    # lookback into the prefill is real, and the pad row is what fills an absent one
    ids = decode["input_ids"].to(device)
    fresh = NgramHashState(args, layout, tokenizer_of(CKPT))
    fresh.bind(device)
    fresh.forward(ids, decode["start_pos"])
    state.forward(prefill["input_ids"].to(device), 0)
    carried = state.forward(ids, decode["start_pos"])
    assert torch.equal(carried, decode["hashes"].to(device))
    assert not torch.equal(carried, fresh.forward(ids, decode["start_pos"]))
