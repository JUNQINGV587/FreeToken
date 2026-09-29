"""TP 2 geometry for V4.1's head-parallel attention and lightning indexer.

V4.1's attention was written for a single rank: the module reshaped ``wq_b``'s output into
``(n_heads, head_dim)`` using the *global* head count, while the parallel linear classes (and
``weight._TP_SHARD_DIM``) had already cut that output to this rank's heads. At TP 2 the unflatten
therefore asked for twice the columns the tensor had -- ``RuntimeError: unflatten: Provided sizes
[64, 512] don't multiply up to the size of dim 2 (16384)`` -- which is what the first TP 2 request
hit. The fix is the usual tensor-parallel split: the module keeps *local* head/group counts for
every reshape and sink, while the parallel linears are still declared with GLOBAL sizes (they do
the cutting), exactly as the reader shards the raw checkpoint tensors.

The indexer takes the opposite decision: it is REPLICATED. The blocks it selects are indices into
the *shared* compressed pool that every rank's query heads read, so all ranks must select the same
ones; scoring every head on every rank -- rather than scoring a slice and summing the partials --
makes them agree by construction. The all-reduce alternative is not even available: the tp
communicator's NCCL path maps only fp16/bf16 (``python/freetoken/kernel/csrc/src/pynccl.cu``,
``kNCCLDtypeMap``) while index scores are fp32, and the first TP 2 request died in
``RuntimeError: unordered_map::at`` proving it.

The TP degree is process-global and can only be set once
(``freetoken.distributed.set_tp_info``), so each degree runs in a child process -- the same
subprocess re-entry convention as ``tests/models/test_deepseek_v41_owner_ep.py``.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

# one kv-source + index-source layer (id 1) beside a window-only layer (id 0)
ARGS = dict(
    dim=512,
    n_heads=4,
    head_dim=64,
    rope_head_dim=16,
    q_lora_rank=32,
    o_lora_rank=32,
    o_groups=2,
    index_n_heads=4,
    index_head_dim=16,
    index_topk=8,
    compress_ratios=(0, 2, 0, 0),
    kv_source_layers=(1,),
    index_source_layers=(1,),
)


def _args(**over):
    from freetoken.models.deepseek_v41.args import DeepseekV41Args

    kwargs = dict(ARGS)
    kwargs.update(over)
    return DeepseekV41Args(**kwargs)


def _attention(layer_id: int = 1, **over):
    from freetoken.models.deepseek_v41.attention import Attention
    from freetoken.models.deepseek_v41.indexer import SharedAttentionRuntime

    return Attention(layer_id, _args(**over), runtime=SharedAttentionRuntime())


def _indexer():
    return _attention(1).indexer


def _report() -> str:
    """Child-process entry: runs the whole interpreter at TP=2."""
    from freetoken.distributed import set_tp_info

    set_tp_info(0, 2)

    attn = _attention(1)
    assert attn.tp_size == 2, "the module has to see the real TP degree"
    assert attn.n_heads == 2, f"local heads, got {attn.n_heads}"
    assert attn.n_groups == 1, f"local output groups, got {attn.n_groups}"
    assert tuple(attn.attn_sink.shape) == (2,), "the sink follows the local heads, one row each"
    # the parallel linears are declared with GLOBAL sizes and cut to this rank themselves;
    # this is also what weight._TP_SHARD_DIM tells the reader to cut
    assert attn.wq_b.full_output_size == 4 * 64, "wq_b is declared global"
    assert attn.wq_b.local_output_size == 4 * 64 // 2, "wq_b keeps half the heads"
    assert tuple(attn.wo_a.shape) == (32, 128), "wo_a: local groups x global per-group width"
    assert attn.wo_b.full_input_size == 2 * 32, "wo_b is declared global"
    assert attn.wo_b.local_input_size == 2 * 32 // 2, "wo_b takes this rank's groups"
    # a window-only layer has no compressor/indexer and the same head split
    assert _attention(0).n_heads == 2

    indexer = _indexer()
    assert indexer.index_n_heads == 4, "the score scale keeps the global head count"
    assert indexer.n_heads == 4, f"the indexer scores every head on every rank, got {indexer.n_heads}"
    assert indexer.wq_b.full_output_size == 4 * 16, "indexer wq_b is replicated, not cut"
    assert indexer.wq_b.local_output_size == 4 * 16, "every rank projects all index heads"
    assert indexer.weights_proj.full_output_size == 4, "weights_proj is replicated too"

    import torch

    weights = indexer._head_weights(torch.zeros(1, 3, 512))
    assert tuple(weights.shape) == (1, 3, 4), f"every head is scored, got {weights.shape}"

    return (
        f"attn(heads={attn.n_heads},groups={attn.n_groups},sink={tuple(attn.attn_sink.shape)},"
        f"wq_b={attn.wq_b.local_output_size},wo_a={tuple(attn.wo_a.shape)},"
        f"wo_b={attn.wo_b.local_input_size}) "
        f"indexer(heads={indexer.n_heads},wq_b={indexer.wq_b.local_output_size},"
        f"weights={indexer.weights_proj.full_output_size},slice={tuple(weights.shape)})"
    )


def _reject_report() -> str:
    """A degree that cannot hand every rank whole output groups is refused, not silently mixed."""
    from freetoken.distributed import set_tp_info

    set_tp_info(0, 4)
    with pytest.raises(ValueError, match="whole output groups"):
        _attention(1)
    return "rejected"


def _child(mode: str) -> str:
    r = subprocess.run(
        [sys.executable, __file__],
        env={**os.environ, "DSV41_TP_MODE": mode},
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert r.returncode == 0, f"TP child ({mode}) failed:\n{r.stdout[-4000:]}\n{r.stderr[-4000:]}"
    return r.stdout.strip()


def test_tp2_splits_heads_and_output_groups():
    assert _child("2") == (
        "attn(heads=2,groups=1,sink=(2,),wq_b=128,wo_a=(32, 128),wo_b=32) "
        "indexer(heads=4,wq_b=64,weights=4,slice=(1, 3, 4))"
    )


def test_a_degree_that_cannot_own_whole_groups_is_refused():
    assert _child("reject") == "rejected"


if __name__ == "__main__":  # subprocess entry: the TP degree is one-shot per process
    print(_reject_report() if os.environ.get("DSV41_TP_MODE") == "reject" else _report())
