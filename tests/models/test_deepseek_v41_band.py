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

The second dump is the same two layers at 300 tokens (``band_oracle.py 300 2``): past the
128-token window, so the prefill takes the prefill-ring's wrap branch and the compressed pool
grows to 300 positions. Its values are NOT a numeric oracle -- the official fp8 path does not
reproduce itself at that M (see ``test_a_prefill_past_the_window_...``); only its selection,
shapes and cache bookkeeping are compared.
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
# The same two layers at 300 tokens: past the 128-token window, so the prefill takes the ring's
# wrap branch and the compressed pool grows to 300 positions (band_oracle.py 300 2).
LONG_DUMP = os.environ.get("FT_V41_BAND_LONG_DUMP", "/oracle/band300_ref.pt")

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and os.path.isdir(CKPT)),
    reason="needs CUDA + the V4.1 checkpoint",
)


@pytest.fixture
def dump_path():
    if not os.path.isfile(DUMP):
        pytest.skip(f"no oracle dump at {DUMP}")
    return DUMP


@pytest.fixture
def long_dump():
    if not os.path.isfile(LONG_DUMP):
        pytest.skip(f"no oracle dump at {LONG_DUMP}")
    return LONG_DUMP


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


def test_the_band_shares_the_compressed_pool_the_reference_way(dump_path):
    dump = torch.load(dump_path, weights_only=False)
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


def test_a_prefill_past_the_window_wraps_the_ring_and_keeps_the_selection(long_dump):
    """The same band at 300 tokens: the prefill ring wraps before the first decode step.

    Only the *selection* is compared against the reference here, and only because the reference
    reproduces it: ``research/v41-oracle/dbg_ref_self.py`` runs the official ``Attention(20)``
    twice on the same 300-token input, and the two runs differ by up to 9.5 in ``q``, differ on
    232 candidate-mask entries, and hand back NaN in 10% of ``out`` -- the official fp8 path is
    not self-consistent at this M, so its *values* cannot be any implementation's standard. Its
    ``idxs``, on the other hand, come back bitwise identical across those two runs.

    Everything else is checked as the structure the reference documents: the ring's wrap order,
    the compressed pool length, the slot alignment of the window indices, the pinned newest
    candidate block, and that our own values are finite where the reference's are not.
    """
    dump = torch.load(long_dump, weights_only=False)
    seqlen = dump["pre"]
    args, runtime, attentions = _build(dump, dump["layers"])
    win = args.window_size
    assert seqlen > win, "this dump is only interesting past the window"

    for rec in dump["records"]:
        rec = {k: v.cuda() if torch.is_tensor(v) else v for k, v in rec.items()}
        layer, start_pos = rec["layer"], rec["start_pos"]
        trace: dict = {}
        out = attentions[layer].forward(rec["x"].to(torch.bfloat16), start_pos, trace=trace)
        torch.cuda.synchronize()
        tag = f"L{layer}pos{start_pos}"

        # The reference's own `out` is 10% NaN at this length; ours must not be, which is the one
        # thing a value comparison against a broken dump could never tell us.
        assert torch.isfinite(out).all(), tag
        assert torch.isfinite(trace["q"]).all() and torch.isfinite(trace["o"]).all(), tag

        # What the reference does reproduce.
        assert torch.equal(trace["idxs"], rec["idxs"]), tag
        assert trace["window_kv"].shape == rec["window_kv"].shape, tag
        assert trace["compress_kv"].shape == rec["compress_kv"].shape, tag

        # A decode step lists the ring's slots in rotation order -- oldest live slot first, then the
        # wrapped tail -- and hides a slot that has not happened yet. The reference's documented
        # rule, written out here rather than read back off the engine.
        if start_pos > 0:
            device = rec["idxs"].device
            oldest = start_pos % win + 1
            expected = torch.cat(
                [
                    torch.arange(oldest, win, device=device),
                    torch.arange(oldest, device=device),
                ]
            )
            expected = torch.where(expected > start_pos, -1, expected).to(trace["idxs"].dtype)
            slots = trace["idxs"][0, :, :win]
            assert torch.equal(slots, expected.expand_as(slots)), tag

        # The wrap itself: after the prefill the ring holds exactly the last `win` rows, rotated so
        # that slot `c` holds the row whose position is `c` mod `win`.
        if start_pos == 0:
            cutoff = seqlen % win
            cache = attentions[layer]._window_cache[0]
            rows = trace["window_kv"][0, -win:]
            assert torch.equal(cache[cutoff:], rows[: win - cutoff]), tag
            assert torch.equal(cache[:cutoff], rows[win - cutoff:]), tag
            assert int(runtime.compress_kv.size(1)) >= seqlen, tag

        # The candidate mask is block-aligned, never wider than topk_blocks blocks, and always keeps
        # the newest (partial) block the reference pins at +inf.
        if rec["candidates"] is not None:
            ref_c, eng_c = rec["candidates"][0], runtime.candidates[0]
            assert eng_c.shape == ref_c.shape, tag
            bs = args.candidate_block_size
            ratio = args.compress_ratios[layer]
            # The mask comes back truncated to the width the scores had; pad it back to whole blocks
            # the way the selection padded the scores (with False, since the tail is dropped).
            padded = torch.nn.functional.pad(eng_c, (0, (-eng_c.size(1)) % bs), value=False)
            per_row = padded.unflatten(-1, (-1, bs)).any(-1).sum(-1)
            assert int(per_row.max()) <= args.candidate_topk_blocks, tag
            assert int(per_row.min()) >= 1, tag
            if start_pos == 0:
                lens = torch.arange(1, rec["seqlen"] + 1, device=eng_c.device) // ratio
            else:
                lens = torch.full((1,), (start_pos + 1) // ratio, device=eng_c.device)
            starts = ((lens - 1) // bs) * bs
            for row in (0, eng_c.size(0) // 2, eng_c.size(0) - 1):
                s = int(starts[row])
                assert eng_c[row, s : min(s + bs, eng_c.size(1))].all(), (tag, "pinned", row)
