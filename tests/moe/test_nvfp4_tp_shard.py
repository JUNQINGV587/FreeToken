"""TP slicing of the routed experts: the ranks' partial sums must add up to the full layer.

``nvfp4_banks._tp_shard`` slices checkpoint tensors along the intermediate axis; this checks
the arithmetic one stage later, on the bank tensors the kernel reads. Rank r's half-width bank
runs the same tokens with no collective (tp info stays 1, so ``_maybe_all_reduce`` is inert)
and the two outputs are summed, which is exactly what the MoE layer's single all-reduce does.

Why it holds: silu and the gate multiply are elementwise per intermediate index, and the down
projection contracts over that index, so a split of I is a split of the sum -- no cross-I term
couples the halves.
"""

from __future__ import annotations

import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

L, E, S = 1, 8, 8
H, I = 256, 128
TOPK = 2
TP = 2
HALF = I // TP

_E2M1 = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
)


def _flat_sources(seed: int = 0) -> dict[str, list[torch.Tensor]]:
    """Bank-ready native sources for ONE layer, in the modelopt layout the loader produces.

    Shapes carry the expert axis and the layer list is length 1, the way ``set_bank_sources``
    takes them. ``gate_up_*`` rows are ``[gate (I) | up (I)]``; ``down_*`` carry I on the
    column axis, packed two codes per byte and one scale per sixteen values.
    """
    g = torch.Generator().manual_seed(seed)

    def rand_u8(*shape):
        return torch.randint(0, 256, shape, dtype=torch.uint8, generator=g)

    def rand_scale(*shape):
        return (torch.rand(*shape, generator=g) * 1.5 + 0.25).to(torch.float8_e4m3fn)

    gate_up_global = torch.full((E, 2 * I), 1.0, dtype=torch.float16)
    gate_up_global[:, I:] = 0.5  # up global != gate global: exercises the alpha fold
    flat = {
        "gate_up_packed": rand_u8(E, 2 * I, H // 2),
        "gate_up_scale": rand_scale(E, 2 * I, H // 16),
        "gate_up_global": gate_up_global,
        "down_packed": rand_u8(E, H, I // 2),
        "down_scale": rand_scale(E, H, I // 16),
        "down_global": torch.full((E, H), 0.75, dtype=torch.float16),
    }
    return {name: [t.pin_memory()] for name, t in flat.items()}


def _rank_slice(full: dict[str, list[torch.Tensor]], rank: int) -> dict[str, list[torch.Tensor]]:
    """Rank ``rank``'s slice of one layer's banks: the I rows of gate and of up (kept in that
    order), the I columns of down (packed /2 codes, /16 block scales), globals untouched."""
    lo, hi = rank * HALF, (rank + 1) * HALF
    lo2, hi2 = lo // 2, hi // 2
    lo16, hi16 = lo // 16, hi // 16
    layer = {name: tensors[0] for name, tensors in full.items()}
    out = {
        "gate_up_packed": torch.cat(
            [layer["gate_up_packed"][:, lo:hi], layer["gate_up_packed"][:, I + lo : I + hi]], dim=1
        ),
        "gate_up_scale": torch.cat(
            [layer["gate_up_scale"][:, lo:hi], layer["gate_up_scale"][:, I + lo : I + hi]], dim=1
        ),
        "gate_up_global": torch.cat(
            [layer["gate_up_global"][:, lo:hi], layer["gate_up_global"][:, I + lo : I + hi]], dim=1
        ),
        "down_packed": layer["down_packed"][:, :, lo2:hi2],
        "down_scale": layer["down_scale"][:, :, lo16:hi16],
        "down_global": layer["down_global"],
    }
    # the cache requires contiguous per-layer sources; a column slice is a view
    return {name: [tensor.contiguous()] for name, tensor in out.items()}


def _run(sources: dict[str, list[torch.Tensor]], hidden, topk_weights, topk_ids, device):
    """One layer through the Triton inline-dequant grouped GEMM, straight from native banks."""
    from freetoken.moe.fused_nvfp4 import fused_experts_nvfp4
    from freetoken.moe.offload_cache import OffloadMoeCache

    cache = OffloadMoeCache(
        num_layers=L,
        num_experts=E,
        cache_size=2 * E,
        device=device,
        quant_format="nvfp4",
        prefill_overlap=True,
    )
    cache.set_bank_sources(sources)
    cache.reset()
    cache.begin_prefill()
    cache.prefetch_prefill_layer(0)
    cache.prefetch_prefill_layer(1)
    gu_p, gu_s, gu_g, dn_p, dn_s, dn_g = cache.wait_prefill_layer(0)
    out = fused_experts_nvfp4(
        hidden, gu_p, gu_s, gu_g, dn_p, dn_s, dn_g,
        topk_weights, topk_ids, E, "silu", False,
    )
    cache.release_prefill_layer(0)
    return out


@cuda
def test_rank_partial_sums_add_up_to_the_full_layer():
    device = torch.device("cuda")
    full = _flat_sources(seed=3)
    torch.manual_seed(4)
    M = 8
    hidden = torch.randn(M, H, dtype=torch.bfloat16, device=device) / 4
    topk_ids = torch.randint(0, E, (M, TOPK), dtype=torch.int32, device=device)
    topk_weights = torch.rand(M, TOPK, dtype=torch.float32, device=device)

    want = _run(full, hidden, topk_weights, topk_ids, device)
    parts = [
        _run(_rank_slice(full, r), hidden, topk_weights, topk_ids, device) for r in range(TP)
    ]
    got = parts[0].float() + parts[1].float()
    torch.testing.assert_close(got, want.float(), rtol=1e-3, atol=0.03 * float(want.abs().max()))
