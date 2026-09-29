"""Layer-2 CSA2 attention against the reference dump (M3).

``attn2_ref.pt`` comes from ``research/v41-oracle/attn2_oracle.py``: the OFFICIAL ``inference/model.py``
``Attention(2)`` with real V4.1 weights, prefill 15 tokens + 3 decode steps. Layer 2 is the first
layer whose behaviour is *all* of CSA2 -- ratio-2 gated pooling, an index source that publishes and
consumes the shared compressed pool, the fp4/E4M3 compressed KV, and the concatenated
``[window | compressed]`` sparse attention.

The three SM89 substitutions the dump needs (torch sparse attention, CUDA window indices, fp8 wo_a
dequantized at load) do not touch any of the arithmetic this test checks.

Needs the real checkpoint and the dump, so it skips unless both are present:
    FT_V41_CHECKPOINT   default /mnt/nvme/models/DeepSeek-V4.1-Flash-NVFP4
    FT_V41_ATTN2_DUMP   default /oracle/attn2_ref.pt
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
from freetoken.models.deepseek_v41.weight import iter_weights
from freetoken.models.register import get_model_spec

CKPT = os.environ.get("FT_V41_CHECKPOINT", "/mnt/nvme/models/DeepSeek-V4.1-Flash-NVFP4")
DUMP = os.environ.get("FT_V41_ATTN2_DUMP", "/oracle/attn2_ref.pt")

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and os.path.isdir(CKPT) and os.path.isfile(DUMP)),
    reason="needs CUDA + the V4.1 checkpoint + the layer-2 attention dump",
)


def _quant_config():
    with open(os.path.join(CKPT, "config.json")) as f:
        quant = json.load(f)["quantization_config"]
    spec = get_model_spec("DeepseekV41ForCausalLM")
    return QuantConfig.from_hf(
        SimpleNamespace(quantization_config=quant), unquantized=spec.unquantized_modules
    )


def _load_attention(layer: int):
    from freetoken.distributed import set_tp_info, try_get_tp_info

    from freetoken.utils.torch_utils import torch_dtype

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    # max_seq_len must match the oracle's: the rope table is built from it.
    args = load_args(CKPT, max_batch_size=1, max_seq_len=2048)
    prefix = f"model.layers.{layer}.attn"
    with torch_dtype(torch.bfloat16):
        attn = Attention(layer, args, quant_config=_quant_config(), prefix=prefix)
    wanted = attn.state_dict(prefix=prefix)
    loaded = {}
    for name, tensor in iter_weights(CKPT, "cpu", include_moe_experts=False):
        if name.startswith(f"model.layers.{layer + 1}."):
            break  # names stream in layer order
        if name in wanted:
            loaded[name] = tensor.to(device="cuda", dtype=wanted[name].dtype)
    attn.load_state_dict(loaded, prefix=prefix)
    attn.bind(torch.device("cuda"))
    return attn


def _errors(actual, expected):
    a, b = actual.float(), expected.float()
    return float((a - b).abs().max()), float((a - b).abs().max() / b.abs().max().clamp_min(1e-6))


def test_layer2_attention_matches_the_reference_dump():
    dump = torch.load(DUMP, weights_only=False)
    attn = _load_attention(dump["layer"])
    report = []
    for rec in dump["records"]:
        rec = {k: v.cuda() if torch.is_tensor(v) else v for k, v in rec.items()}
        x = rec["x"].to(torch.bfloat16)
        trace: dict = {}
        out = attn.forward(x, rec["start_pos"], trace=trace)
        tag = f"pos{rec['start_pos']}"
        # The index list is the layer's semantics: same compressed positions or the port is wrong.
        assert torch.equal(trace["idxs"], rec["idxs"]), tag
        if rec["start_pos"] == 0:
            # Prefill: the fp8 GEMMs, the window KV and the ring write all agree exactly. Decode
            # runs the M=1 split-K path instead, whose accumulation order differs from the
            # reference kernel in the last bf16 bit -- so decoding is checked with a tolerance.
            assert torch.equal(trace["q"], rec["q"]), tag
            assert torch.equal(trace["window_kv"], rec["window_kv"]), tag
            assert torch.equal(attn._window_cache[:1], rec["window_cache"]), tag
        errs = {
            "q": _errors(trace["q"], rec["q"]),
            "window_kv": _errors(trace["window_kv"], rec["window_kv"]),
            "compress_kv": _errors(trace["compress_kv"], rec["compress_kv"]),
            "window_cache": _errors(attn._window_cache[:1], rec["window_cache"]),
            "o": _errors(trace["o"], rec["o"]),
            "out": _errors(out, rec["out"]),
        }
        if rec["compress_kv"].numel():
            errs["compress_cache"] = _errors(
                attn._compress_cache[:1, : rec["compress_cache"].size(1)], rec["compress_cache"]
            )
        report.append((tag, errs))
        for name, (abs_err, rel_err) in errs.items():
            assert rel_err < 2e-2, (tag, name, abs_err, rel_err)
    for tag, errs in report:
        print(tag, {k: f"{v[0]:.3e}/{v[1]:.3e}" for k, v in errs.items()})
