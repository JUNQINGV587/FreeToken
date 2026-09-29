"""DeepSeek-V4.1 routed experts: the swiglu limit has to survive kernel selection.

V4.1 trained with ``swiglu_limit=10`` (``silu(min(gate, L)) * clamp(up, +-L)``). The layer used to
hand the MoE a ``limit=None`` because no NVFP4 backend could express a clamp, which meant the
routed experts ran a different function than the shared expert. Now the epilogue dispatches a
finite limit onto ``swiglu_clamp``, so this pins the two things that make that real: the config
carries the limit, and the backend that can compute it is the one that gets selected.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

DIM = 64
INTER = 32
EXPERTS = 4
TOPK = 2
LIMIT = 10.0


def _layer():
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.layers.quantization import QuantBackend, QuantConfig, set_quant_backend
    from freetoken.models.deepseek_v41.args import DeepseekV41Args
    from freetoken.models.deepseek_v41.moe import DSV41OffloadMoELayer

    if try_get_tp_info() is None:
        set_tp_info(0, 1)
    set_quant_backend(QuantBackend.parse("moe.nvfp4=triton"))
    quant = QuantConfig.from_hf(
        {"quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4", "ignore": ["lm_head"]}}
    )
    args = DeepseekV41Args(
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
    )
    return DSV41OffloadMoELayer(
        0, args, strategy="offload", decode_target="gpu", quant_config=quant,
        prefix="model.layers.0.ffn",
    )


def test_the_routed_experts_keep_the_swiglu_limit_and_stay_on_a_kernel_that_clamps():
    layer = _layer()
    cfg = layer.quant_method.cfg
    assert cfg.activation == "silu"
    assert cfg.limit == LIMIT, "the layer dropped the trained swiglu limit"
    assert cfg.alpha == 1.0

    from freetoken.layers.quantization.method import select_kernel
    from freetoken.layers.quantization.moe.nvfp4 import (
        B12xNvfp4MoEKernel,
        MarlinNvfp4MoEKernel,
        TritonNvfp4MoEKernel,
    )

    assert TritonNvfp4MoEKernel().unusable_reason(cfg) is None
    assert MarlinNvfp4MoEKernel().unusable_reason(cfg) is not None
    assert layer.quant_method.kernel.name == "triton"
    # and the table picks it on its own: marlin/b12x cannot clamp, so a limit must not fall back
    # to a backend that would quietly run the unclamped activation
    auto = select_kernel(
        (TritonNvfp4MoEKernel, MarlinNvfp4MoEKernel, B12xNvfp4MoEKernel), "auto", cfg
    )
    assert auto.name == "triton"


def test_a_checkpoint_without_the_limit_keeps_the_plain_silu_fast_path():
    """limit<=0 must not fabricate one: no limit means the unclamped kernel is still the cheap one."""
    from freetoken.layers.quantization.moe.base import MoEConfig

    plain = MoEConfig(num_experts=EXPERTS, hidden=DIM, intermediate=INTER, top_k=TOPK, scheme=None)
    assert plain.plain_silu
    clamped = dataclasses.replace(plain, limit=LIMIT)
    assert not clamped.plain_silu


def test_the_shared_expert_is_clamped_by_the_dsv4_kernel():
    from freetoken.models.deepseek_v41.moe import Expert

    torch.manual_seed(0)
    with torch.device("cuda"):
        expert = Expert(DIM, INTER, LIMIT)
    # the linears come up as torch.empty: on recycled device memory they can hold NaN/inf bit
    # patterns, which makes a "the output is finite" assertion depend on what ran before it
    gen = torch.Generator(device="cuda").manual_seed(0)
    with torch.no_grad():
        for tensor in expert.state_dict().values():
            if tensor.is_floating_point() and tensor.numel() > 1:
                tensor.copy_(torch.randn(tensor.shape, generator=gen, device="cuda", dtype=torch.float32).mul_(0.05).to(tensor.dtype))
    x = torch.randn(4, DIM, dtype=torch.bfloat16, device="cuda") * 4
    out = expert.forward(x)
    assert out.shape == x.shape and torch.isfinite(out).all()

    gate = expert.w1.forward(x).float()
    up = expert.w3.forward(x).float()
    ref = (torch.nn.functional.silu(gate.clamp(max=LIMIT)) * up.clamp(-LIMIT, LIMIT)).to(torch.bfloat16)
    ref = expert.w2.forward(ref)
    assert (out.float() - ref.float()).abs().max().item() < 2e-2


class _StubSlotCache:
    """Just enough ``OffloadMoeCache`` to exercise the prefill shortcut's decisions.

    The shortcut's whole point is the *id space*: ``bank_views()`` returns the pool's rows, so
    the ids marching into the GEMM are slots, not experts. Only the methods the shortcut calls
    are implemented.
    """

    def __init__(self, slots: int) -> None:
        self.slots = slots
        self.collect_stats = False
        self.prefill_overlap = False
        self.calls: list[str] = []

    def is_unpinned_layer(self, layer_id: int) -> bool:
        return False

    def ensure_experts(self, layer_id: int, topk_ids) -> None:
        self.calls.append("ensure_experts")

    def copy_missing(self) -> None:
        self.calls.append("copy_missing")

    def record_decode_stats(self, layer_id: int) -> None:
        self.calls.append("record_decode_stats")

    def bank_views(self):
        return (torch.empty(self.slots, 4, 8), torch.empty(self.slots, 4))

    def alphas_for_slots(self, layer_id: int):
        return None


def test_a_small_prefill_chunk_routes_through_the_slot_cache(monkeypatch):
    """The shortcut is legal only because the align kernel can take the pool's id space.

    Both halves matter: a chunk that touches fewer rows than the layer has experts takes the
    decode-style slot path and tells the GEMM the id space is the *pool* (``n == pool``, since
    those rows are slots) -- and a chunk that touches as many rows as the layer owns must fall
    back to whole-layer streaming, where position == expert id.
    """
    from freetoken.layers.moe import OffloadMoELayer

    layer = _layer()
    slots = 1675
    layer.offload_cache = _StubSlotCache(slots)
    seen: dict = {}

    def fake_gemm(cache, hidden_states, topk_weights, topk_ids, **kwargs):
        seen.update(kwargs)
        return hidden_states

    streamed: list[int] = []

    def base_prefill(self, hidden_states, topk_weights, topk_ids):
        streamed.append(int(hidden_states.shape[0]))
        return hidden_states

    monkeypatch.setattr(layer, "_expert_gemm", fake_gemm)
    monkeypatch.setattr(OffloadMoELayer, "_prefill_routed", base_prefill)

    x = torch.zeros(1, DIM, dtype=torch.bfloat16, device="cuda")
    weights = torch.zeros(1, TOPK, dtype=torch.float32, device="cuda")
    ids = torch.zeros(1, TOPK, dtype=torch.int32, device="cuda")

    layer._prefill_routed(x, weights, ids)
    assert seen, "a 1-token chunk has to take the on-demand slot path"
    assert seen["is_prefill"] is True
    assert seen["n"] == slots, "n is the align kernel's num_experts: the pool these rows live in"
    assert seen["views"][0].shape[0] == slots
    assert layer.offload_cache.calls == ["ensure_experts", "copy_missing"]
    assert streamed == []

    # EXPERTS rows or more -> whole-layer streaming (2 tokens x top_k 2 == the toy layer's 4)
    seen.clear()
    x2 = torch.zeros(2, DIM, dtype=torch.bfloat16, device="cuda")
    weights2 = torch.zeros(2, TOPK, dtype=torch.float32, device="cuda")
    ids2 = torch.zeros(2, TOPK, dtype=torch.int32, device="cuda")
    layer._prefill_routed(x2, weights2, ids2)
    assert seen == {}, "a chunk that touches every expert row must not use the slot path"
    assert streamed == [2]
