"""CSA2 band semantics: layers 20 and 24 against the reference dump (M3).

``band_ref.pt`` comes from ``research/v41-oracle/band_oracle.py`` -- the OFFICIAL ``Attention(20)``
and ``Attention(24)`` sharing the repo's module-global ``shared_attn``, in layer order, prefill 32
tokens then 2 decode steps. What only this test can catch:

  * the runtime is *shared*: layer 24 reads the compressed pool and the index-K cache that layer 20
    published, across separate forward calls,
  * layer 24 is an index source but not a kv source: it scores against someone else's K,
  * the candidate mask -- layer 20 marks the blocks, layer 24 may only score inside them,
  * ratio 1 (a plain projection, no ``wgate``) on the source side.

The dump overrides ``candidate_topk_blocks``/``candidate_block_size`` to 2/4 so the mask is
selective at this length; the args are rebuilt from the dump, so both sides agree.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest
import torch

from freetoken.layers.quantization import QuantConfig
from freetoken.models.deepseek_v41.args import load_args
from freetoken.models.deepseek_v41.attention import Attention
from freetoken.models.deepseek_v41.indexer import SharedAttentionRuntime
from freetoken.models.deepseek_v41.weight import iter_weights
from freetoken.models.register import get_model_spec

CKPT = os.environ.get("FT_V41_CHECKPOINT", "/mnt/nvme/models/DeepSeek-V4.1-Flash-NVFP4")
DUMP = os.environ.get("FT_V41_BAND_DUMP", "/oracle/band_ref.pt")

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and os.path.isdir(CKPT) and os.path.isfile(DUMP)),
    reason="needs CUDA + the V4.1 checkpoint + the band oracle dump",
)

# The two overrides the oracle made, so the mask is selective at 32 tokens (see the oracle).
_OVERRIDES = ("candidate_topk_blocks", "candidate_block_size")


def _quant_config():
    with open(os.path.join(CKPT, "config.json")) as f:
        quant = json.load(f)["quantization_config"]
    spec = get_model_spec("DeepseekV41ForCausalLM")
    return QuantConfig.from_hf(
        SimpleNamespace(quantization_config=quant), unquantized=spec.unquantized_modules
    )


def _build(dump, layers):
    from freetoken.distributed import set_tp_info, try_get_tp_info

    from freetoken.utils.torch_utils import torch_dtype

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    overrides = {k: dump["args"][k] for k in _OVERRIDES}
    args = load_args(CKPT, max_batch_size=1, max_seq_len=2048, **overrides)
    runtime = SharedAttentionRuntime()
    attentions = {}
    with torch_dtype(torch.bfloat16):
        for layer in layers:
            prefix = f"model.layers.{layer}.attn"
            attention = Attention(
                layer, args, quant_config=_quant_config(), prefix=prefix, runtime=runtime
            )
            wanted = attention.state_dict(prefix=prefix)
            loaded = {}
            for name, tensor in iter_weights(CKPT, "cpu", include_moe_experts=False):
                if name.startswith(f"model.layers.{layer + 1}."):
                    break
                if name in wanted:
                    loaded[name] = tensor.to(device="cuda", dtype=wanted[name].dtype)
            attention.load_state_dict(loaded, prefix=prefix)
            attention.bind(torch.device("cuda"))
            attentions[layer] = attention
    return args, runtime, attentions


def _errors(actual, expected):
    a, b = actual.float(), expected.float()
    return float((a - b).abs().max()), float((a - b).abs().max() / b.abs().max().clamp_min(1e-6))


def test_the_band_shares_the_compressed_pool_the_reference_way():
    dump = torch.load(DUMP, weights_only=False)
    layers = dump["layers"]
    args, runtime, attentions = _build(dump, layers)
    assert args.kv_source_for(layers[1]) == layers[0]

    for rec in dump["records"]:
        rec = {k: v.cuda() if torch.is_tensor(v) else v for k, v in rec.items()}
        layer, start_pos = rec["layer"], rec["start_pos"]
        trace: dict = {}
        out = attentions[layer].forward(rec["x"].to(torch.bfloat16), start_pos, trace=trace)
        tag = f"L{layer}pos{start_pos}"
        errs = {"q": _errors(trace["q"], rec["q"]),
                "window_kv": _errors(trace["window_kv"], rec["window_kv"]),
                "compress_kv": _errors(trace["compress_kv"], rec["compress_kv"]),
                "o": _errors(trace["o"], rec["o"]),
                "out": _errors(out, rec["out"])}
        assert torch.equal(trace["idxs"], rec["idxs"]), tag
        if start_pos == 0:
            assert torch.equal(trace["q"], rec["q"]), tag
            assert torch.equal(trace["window_kv"], rec["window_kv"]), tag
        # The candidate mask is published by layer 20 and read by layer 24; reproducing it is the
        # only proof the two levels agree about which blocks are in play.
        assert rec["candidates"] is None or torch.equal(runtime.candidates, rec["candidates"]), tag
        if rec["index_k"] is not None:
            assert torch.equal(runtime.index_k[: rec["index_k"].size(0), : rec["index_k"].size(1)],
                               rec["index_k"]), tag
        if "compress_cache" in rec:
            errs["compress_cache"] = _errors(
                runtime.compress_kv[:1, : rec["compress_cache"].size(1)], rec["compress_cache"]
            )
        print(tag, {k: f"{v[0]:.3e}/{v[1]:.3e}" for k, v in errs.items()})
        for name, (abs_err, rel_err) in errs.items():
            assert rel_err < 2e-2, (tag, name, abs_err, rel_err)
