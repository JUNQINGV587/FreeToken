"""CSA2 compressor vs the reference ``Compressor`` (M3, first piece).

``compressor_ref.pt`` is produced by ``research/v41-oracle/compressor_oracle.py``: the OFFICIAL
``inference/model.py:429`` class with random bf16-exact weights and a synthetic config, driven
through one prefill and a run of decode steps. The weight *values* are bf16-exact so the engine
(bf16 parameters, fp32 activation stream) and the reference (fp32 parameters for ratio 2) execute
the same arithmetic rather than merely similar arithmetic.

What it pins down is the pooling: softmax-gated pair averaging in fp32, and the carry of an
incomplete trailing group in ``kv_state``/``score_state`` across forwards. Both cases matter --
``r2_odd`` prefills an odd token count, so its first decode step completes the carried group.

    FT_V41_COMPRESSOR_DUMP  default /oracle/compressor_ref.pt
"""

from __future__ import annotations

import os

import pytest
import torch

from freetoken.models.deepseek_v41.args import DeepseekV41Args
from freetoken.models.deepseek_v41.compress import Compressor

DUMP = os.environ.get("FT_V41_COMPRESSOR_DUMP", "/oracle/compressor_ref.pt")

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and os.path.isfile(DUMP)),
    reason="needs CUDA + the compressor oracle dump",
)

# the synthetic geometry compressor_oracle.py builds; compress_ratios only has to place the ratio
# of the layer under test (layer_id), everything else reads that one entry
_DIM = 256
_HEAD_DIM = 128
_ROPE_HEAD_DIM = 32


def _reference():
    return torch.load(DUMP, map_location="cpu", weights_only=False)


def _args(layer_id: int, ratio: int) -> DeepseekV41Args:
    ratios = tuple(0 if i != layer_id else ratio for i in range(layer_id + 9))
    return DeepseekV41Args(
        dim=_DIM,
        head_dim=_HEAD_DIM,
        rope_head_dim=_ROPE_HEAD_DIM,
        norm_eps=1e-20,
        max_batch_size=2,
        max_seq_len=512,
        compress_ratios=ratios,
    )


def _build(root: dict, layer_id: int, ratio: int) -> Compressor:
    from freetoken.distributed import set_tp_info, try_get_tp_info

    from freetoken.utils.torch_utils import torch_dtype

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    # the engine builds models under its compute dtype; the reference dump is bf16-exact, so the
    # module must be built in bf16 for the load to be exact too
    with torch_dtype(torch.bfloat16):
        comp = Compressor(layer_id, _args(layer_id, ratio), prefix=f"model.layers.{layer_id}.attn.compressor")
    # the dump keeps the reference's storage dtypes (fp32 for ratio 2); its VALUES are bf16-exact,
    # so casting to the engine's own dtypes here is lossless -- that is what the engine loader does
    target = comp.state_dict()
    state = {
        k: v.to(device="cuda", dtype=target[k].dtype) for k, v in root["weights"].items()
    }
    comp.load_state_dict(state)
    return comp


def _errors(actual, expected):
    a, b = actual.float(), expected.float()
    return float((a - b).abs().max()), float((a - b).abs().max() / b.abs().max().clamp_min(1e-6))


@pytest.mark.parametrize("case", ["r2_even", "r2_odd", "r1"])
def test_compressor_matches_the_reference_dump(case):
    root = _reference()[case]
    comp = _build(root, root["layer_id"], root["ratio"])
    produced = 0
    for step in root["steps"]:
        x = step["x"].to("cuda")
        got = comp.forward(x, step["start_pos"])
        want = step["latent"]
        if want is None:
            assert got is None, f"{case} start_pos={step['start_pos']}: expected no group"
            continue
        assert got is not None, f"{case} start_pos={step['start_pos']}: expected a group"
        want = want.to("cuda")
        assert got.shape == want.shape, (case, step["start_pos"], tuple(got.shape), tuple(want.shape))
        abs_err, rel_err = _errors(got, want)
        assert abs_err <= 1e-6, (
            f"{case} start_pos={step['start_pos']}: abs={abs_err:.3e} rel={rel_err:.3e}"
        )
        produced += 1
    assert produced == sum(1 for s in root["steps"] if s["latent"] is not None)


def test_ratio1_compressor_has_no_gate_and_no_state():
    """ratio 1 is a plain projection: the reference allocates neither wgate nor the carry state."""
    root = _reference()["r1"]
    comp = _build(root, root["layer_id"], root["ratio"])
    assert comp.wgate is None
    assert comp._kv_state is None


def test_ratio2_compressor_carries_an_incomplete_group():
    """The 7-token prefill leaves one token in the state and the next decode step consumes it."""
    root = _reference()["r2_odd"]
    comp = _build(root, root["layer_id"], root["ratio"])
    steps = root["steps"]
    comp.forward(steps[0]["x"].to("cuda"), 0)  # 7 tokens -> 3 groups, 1 token carried
    assert comp._kv_state is not None
    carried = comp._kv_state[0, 0]
    tail_kv = comp.wkv.forward(steps[0]["x"].to("cuda").float())[0, -1]
    assert torch.equal(carried, tail_kv)
    latent = comp.forward(steps[1]["x"].to("cuda"), 7)
    assert latent is not None and latent.shape[1] == 1
    comp.reset()
    assert comp._kv_state is None
