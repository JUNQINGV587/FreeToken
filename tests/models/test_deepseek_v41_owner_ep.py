"""Owner-local EP under TP 2: the routed experts must still look whole to the expert kernels.

``--moe-ep-size 2`` together with ``--tensor-parallel-size 2`` partitions the routed experts
between the ranks instead of sharding each one, which is the only way V4.1's NVFP4 expert banks
can be served at TP 2: every NVFP4 expert kernel rejects ``tp_size > 1``, because a sharded NVFP4
expert has no kernel that can run it. The layer therefore has to keep the REAL TP degree (its
output all-reduce is what sums the rank-local contributions) while handing ``expert_tp_size=1``
to the kernel config.

The TP degree is process-global and can only be set once
(``freetoken.distributed.set_tp_info``), so the assertions run in a child process -- the same
subprocess re-entry convention as ``tests/kernels/test_fp8_block32.py``.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

DIM = 64
INTER = 32
EXPERTS = 4
TOPK = 2
LIMIT = 10.0


def _args(moe_ep_size: int):
    from freetoken.models.deepseek_v41.args import DeepseekV41Args

    return DeepseekV41Args(
        dim=DIM,
        n_layers=2,
        vocab_size=256,
        n_heads=4,
        head_dim=64,
        rope_head_dim=16,
        q_lora_rank=32,
        o_lora_rank=32,
        o_groups=2,
        n_routed_experts=EXPERTS,
        n_activated_experts=TOPK,
        moe_inter_dim=INTER,
        n_shared_experts=1,
        hc_mult=2,
        compress_ratios=(0, 0),
        swiglu_limit=LIMIT,
        moe_ep_size=moe_ep_size,
    )


def _layer(moe_ep_size: int):
    from freetoken.layers.quantization import QuantBackend, QuantConfig, set_quant_backend
    from freetoken.models.deepseek_v41.moe import DSV41OffloadMoELayer

    set_quant_backend(QuantBackend.parse("moe.nvfp4=triton"))
    quant = QuantConfig.from_hf(
        {"quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4", "ignore": ["lm_head"]}}
    )
    return DSV41OffloadMoELayer(
        0, _args(moe_ep_size), strategy="offload", decode_target="gpu", quant_config=quant,
        prefix="model.layers.0.ffn",
    )


def _report() -> str:
    """Child-process entry: runs with TP=2 for this whole interpreter."""
    from freetoken.distributed import set_tp_info
    from freetoken.layers.quantization.moe.nvfp4 import TritonNvfp4MoEKernel

    set_tp_info(0, 2)
    kernel = TritonNvfp4MoEKernel()

    owner = _layer(moe_ep_size=2)
    assert owner.tp_size == 2, "the output all-reduce still runs at the real TP degree"
    assert owner.expert_tp_size == 1, "whole experts per rank -> the expert GEMM is unsharded"
    assert owner.quant_method.cfg.tp_size == 1, "the kernel has to see the owner-local view"
    assert kernel.unusable_reason(owner.quant_method.cfg) is None, "owner-local EP is servable at TP 2"

    # the negative control: TP 2 without owner-local EP is exactly the combination the engine
    # refuses instead of building it and failing later -- without the override the layer does not
    # even construct, because kernel selection rejects the sharded expert banks
    from freetoken.layers.quantization.method import KernelSelectionError
    from freetoken.layers.quantization.moe.base import MoEConfig

    with pytest.raises(KernelSelectionError, match="TP > 1"):
        _layer(moe_ep_size=1)

    sharded = MoEConfig(
        num_experts=EXPERTS, hidden=DIM, intermediate=INTER, top_k=TOPK, tp_size=2,
        strategy="offload", decode_target="gpu",
    )
    reason = kernel.unusable_reason(sharded)
    assert reason == "TP > 1 is not supported for this expert format", f"got {reason!r}"

    return (
        f"owner(tp={owner.tp_size},expert_tp={owner.expert_tp_size}) "
        f"sharded(tp=2,expert_tp=2) unusable={reason!r}"
    )


def test_owner_local_ep_keeps_the_experts_whole_under_tp2():
    r = subprocess.run(
        [sys.executable, __file__], env=dict(os.environ), capture_output=True, text=True, timeout=900
    )
    assert r.returncode == 0, f"TP=2 child failed:\n{r.stdout[-4000:]}\n{r.stderr[-4000:]}"
    assert r.stdout.strip() == (
        "owner(tp=2,expert_tp=1) sharded(tp=2,expert_tp=2) "
        "unusable='TP > 1 is not supported for this expert format'"
    ), r.stdout


if __name__ == "__main__":  # subprocess entry: TP=2 is one-shot per process
    print(_report())
