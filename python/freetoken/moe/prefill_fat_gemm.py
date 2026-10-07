"""Prefill busy-expert fat GEMM (port-plan 2c): decision + assembly, env-gated, default off.

dsv41's D061 showed the FP16 busy-expert fat GEMM is the main prefill speedup
(93 -> 565-813 tok/s) AND the fidelity fix: its pure staged-DMA path failed the
fidelity guard (top-1 0.9856 < 0.988), only the fat-GEMM arm passed. The same
lesson applies here in reverse: our grouped Triton prefill kernel is the
production reference, and this fat path is the EXPERIMENTAL arm.

Gate: the flag FREETOKEN_PREFILL_FAT_GEMM stays experimental (default off)
until the fidelity gate (bf16 top-1 >= 0.98, rel-err <= 0.10, D061 metric;
``fidelity_gate`` / ``python -m freetoken.moe.prefill_fat_gemm``) passes on the
production geometry. DO NOT default-on before the gate.

Assembly: routes whose expert holds >= FREETOKEN_PREFILL_FAT_GEMM_MIN_ROWS
rows (default 32, dsv41's EXL3_DMA_GEMM cut was ">32 rows"; retune with the
wall-clock A/B after the gate) are dequantized NVFP4 -> fp32 -> bf16 and run as
one dense torch GEMM per busy expert; every other route stays on the
production grouped kernel with a host-built align that reuses the production
config, so light rows are bitwise identical to the status quo.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch

from freetoken.kernel.triton.nvfp4_fused_moe import _E2M1_VALUES

ENV_FLAG = "FREETOKEN_PREFILL_FAT_GEMM"
ENV_MIN_ROWS = "FREETOKEN_PREFILL_FAT_GEMM_MIN_ROWS"
# dsv41 cut busy experts at >32 routed rows (runtime/patch_dma_gemm.py); the
# crossover is hardware- and geometry-dependent, so this is a starting value to
# be re-tuned by the post-gate wall-clock A/B, not a validated optimum.
DEFAULT_MIN_ROWS = 32

# D061 fidelity gate (bf16). Module-level proxy for the end-to-end greedy
# top-1 gate; the real battery reruns it on the production checkpoint.
FIDELITY_TOP1_MIN = 0.98
FIDELITY_REL_ERR_MAX = 0.10

_E2M1 = torch.tensor(_E2M1_VALUES, dtype=torch.float32)

_STATS = {
    "calls": 0,           # dispatch calls with the flag ON
    "fallback_calls": 0,  # flag on but no busy expert -> plain grouped path
    "busy_experts": 0,
    "light_experts": 0,
    "busy_routes": 0,     # routed (token, expert) pairs on the fat path
    "light_routes": 0,
}


def fat_gemm_stats() -> dict:
    """Counters for the /v1/stats snapshot; routes are (token, expert) pairs."""
    out = dict(_STATS)
    out["enabled"] = os.environ.get(ENV_FLAG, "0") == "1"
    out["min_rows"] = _min_rows()
    return out


def reset_fat_gemm_stats() -> None:
    for key in _STATS:
        _STATS[key] = 0


def _flag_on() -> bool:
    return os.environ.get(ENV_FLAG, "0") == "1"


def _min_rows() -> int:
    value = int(os.environ.get(ENV_MIN_ROWS, "0") or 0)
    return value if value > 0 else DEFAULT_MIN_ROWS


# ---------------------------------------------------------------------------
# Decision (pure host logic, CPU-testable)
# ---------------------------------------------------------------------------


@dataclass
class FatSplit:
    """Host-side route partition: busy experts (fat GEMM) vs the light rest."""

    busy: list[int]                       # busy expert ids, ascending
    busy_positions: list[torch.Tensor]    # per busy expert: int64 flat route positions
    light_sorted_ids: torch.Tensor        # int32 [ntpp], padding sentinel == num_valid
    light_expert_ids: torch.Tensor        # int32 [ntpp // block_m]
    light_ntpp: int                       # num tokens post padding (blocks * block_m)


def plan_fat_split(
    flat_ids: torch.Tensor,
    *,
    top_k: int,
    num_experts: int,
    min_rows: int,
    block_m: int,
) -> FatSplit | None:
    """Partition flattened route ids by per-expert occupancy.

    Returns None when no expert reaches ``min_rows`` rows (caller falls back to
    the plain grouped path). The light align mirrors ``moe_align_block_size``
    semantics: routes grouped by ascending expert id (stable within an expert),
    each group padded to ``block_m`` with the sentinel ``num_valid``; row
    outputs of the grouped kernel are per-route dot products, so the light
    rows come out bitwise identical to the full production align.
    """
    flat = flat_ids.reshape(-1).to(torch.int64).cpu()
    num_valid = flat.numel()
    counts = torch.bincount(flat, minlength=num_experts)
    busy = torch.nonzero((counts >= min_rows) & (counts > 0)).flatten().tolist()
    if not busy:
        return None
    busy_set = set(busy)
    order = torch.argsort(flat, stable=True)
    ends = torch.cumsum(counts, 0)
    starts = ends - counts

    busy_positions: list[torch.Tensor] = []
    light_sorted: list[torch.Tensor] = []
    light_experts: list[int] = []
    for e in range(num_experts):
        c = int(counts[e])
        if c == 0:
            continue
        pos = order[int(starts[e]):int(ends[e])]
        if e in busy_set:
            busy_positions.append(pos)
            continue
        pad = (-c) % block_m
        light_sorted.append(pos.to(torch.int32))
        if pad:
            light_sorted.append(torch.full((pad,), num_valid, dtype=torch.int32))
        light_experts.extend([e] * ((c + pad) // block_m))

    if light_experts:
        sorted_ids = torch.cat(light_sorted)
        expert_ids = torch.tensor(light_experts, dtype=torch.int32)
        ntpp = int(sorted_ids.numel())
    else:
        sorted_ids = torch.empty(0, dtype=torch.int32)
        expert_ids = torch.empty(0, dtype=torch.int32)
        ntpp = 0
    return FatSplit(busy, busy_positions, sorted_ids, expert_ids, ntpp)


# ---------------------------------------------------------------------------
# Fat GEMM math (device-agnostic torch; CPU-testable)
# ---------------------------------------------------------------------------


def dequant_nvfp4_weight(
    packed: torch.Tensor,
    scale: torch.Tensor,
    glob: torch.Tensor,
) -> torch.Tensor:
    """NVFP4 projection(s) -> fp32 ``[..., N, K]``.

    ``packed`` ``[..., N, K//2]`` uint8 (byte j holds codes k=2j lo / k=2j+1 hi),
    ``scale`` ``[..., N, K//16]`` fp8-e4m3, ``glob`` ``[..., N]`` fp16; leading
    dims broadcast (pass a single expert slice or the whole bank). The kernel
    applies the per-row global AFTER its fp32 accumulation; folding it in here
    costs <=1 bf16 ulp once the result is cast for the GEMM (same fold the
    marlin pack does) and is far inside the fidelity budget.
    """
    *lead, n, k_half = packed.shape
    lut = _E2M1.to(packed.device)
    codes = torch.stack((packed & 0xF, packed >> 4), dim=-1).reshape(*lead, n, k_half * 2)
    w = lut[codes.long()]
    s = scale.to(torch.float32).repeat_interleave(16, dim=-1)
    return w * s * glob.to(torch.float32).unsqueeze(-1)


def _torch_gated_act(
    activation: str,
    x: torch.Tensor,
    *,
    alpha: float,
    limit: float,
) -> torch.Tensor:
    """CPU reference for ``layers.gated_act_and_mul`` (selftest only)."""
    d = x.shape[-1] // 2
    gate, up = x[..., :d], x[..., d:]
    if limit != float("inf"):
        # swiglu_clamp (V4.1 swiglu_limit): clamp BEFORE the sigmoid, no up bias.
        gate = gate.clamp(max=limit)
        up = up.clamp(-limit, limit)
        return gate * torch.sigmoid(alpha * gate) * up
    if activation == "silu":
        return torch.nn.functional.silu(gate) * up
    raise NotImplementedError(f"CPU reference act for {activation!r} (limit=inf)")


def _gated_act(activation: str, x: torch.Tensor, out: torch.Tensor, *, alpha: float, limit: float):
    if x.is_cuda:
        from freetoken.layers import gated_act_and_mul

        return gated_act_and_mul(activation, x, out, alpha=alpha, limit=limit)
    return out.copy_(_torch_gated_act(activation, x, alpha=alpha, limit=limit))


def _expert_rows_out(
    x_e: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w_e: torch.Tensor,
    activation: str,
    apply_router_weight_on_input: bool,
    act_alpha: float,
    act_limit: float,
) -> torch.Tensor:
    """Two dequantized GEMMs + gated epilogue for one expert's rows.

    Mirrors the production weight application: route weight on gemm1 when
    ``apply_router_weight_on_input`` else on gemm2. ``w1``/``w2`` are fp32
    dequantized ``[2I, H]`` / ``[H, I]`` and are cast to the activation dtype,
    matching the kernel's fp32-dequant -> cast -> dot order.
    """
    dt = x_e.dtype
    ic1 = x_e @ w1.to(dt).T
    if apply_router_weight_on_input:
        ic1 = ic1 * w_e.unsqueeze(1)
    inter = w2.shape[1]
    ic2 = torch.empty(ic1.shape[0], inter, device=x_e.device, dtype=dt)
    _gated_act(activation, ic1.contiguous(), ic2, alpha=act_alpha, limit=act_limit)
    ic3 = ic2 @ w2.to(dt).T
    if not apply_router_weight_on_input:
        ic3 = ic3 * w_e.unsqueeze(1)
    return ic3


# ---------------------------------------------------------------------------
# Fidelity gate (D061 metric)
# ---------------------------------------------------------------------------


def top1_agreement(candidate: torch.Tensor, reference: torch.Tensor) -> float:
    """Fraction of rows whose argmax matches (bf16 top-1 proxy)."""
    agree = (candidate.argmax(dim=-1) == reference.argmax(dim=-1)).to(torch.float64)
    return float(agree.mean().item())


def rel_err(candidate: torch.Tensor, reference: torch.Tensor) -> float:
    """Frobenius ||candidate - reference|| / ||reference||."""
    diff = (candidate.to(torch.float64) - reference.to(torch.float64)).norm()
    ref = reference.to(torch.float64).norm()
    return float((diff / ref.clamp_min(1e-30)).item())


def fidelity_gate(candidate: torch.Tensor, reference: torch.Tensor) -> dict:
    """D061 gate: bf16 top-1 >= 0.98 AND rel-err <= 0.10."""
    top1 = top1_agreement(candidate, reference)
    err = rel_err(candidate, reference)
    return {
        "top1": top1,
        "rel_err": err,
        "top1_min": FIDELITY_TOP1_MIN,
        "rel_err_max": FIDELITY_REL_ERR_MAX,
        "passed": top1 >= FIDELITY_TOP1_MIN and err <= FIDELITY_REL_ERR_MAX,
    }


# ---------------------------------------------------------------------------
# GPU orchestration
# ---------------------------------------------------------------------------


def fused_experts_nvfp4_fat(
    hidden_states: torch.Tensor,
    gate_up_packed: torch.Tensor,
    gate_up_scale: torch.Tensor,
    gate_up_global: torch.Tensor,
    down_packed: torch.Tensor,
    down_scale: torch.Tensor,
    down_global: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_experts: int,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    act_alpha: float = 1.0,
    act_limit: float = float("inf"),
) -> torch.Tensor:
    """Prefill MoE with the busy experts on dense dequantized GEMMs.

    Same calling convention as ``fused_nvfp4.fused_experts_nvfp4``; falls back
    to it bitwise when no expert reaches the busy threshold. Not CUDA-graph
    capturable (one host sync to bucket the routes) -- prefill never is.
    The per-layer D2H of the route ids is the known cost; a device-side
    counting variant is the follow-up if the gate passes.
    """
    from freetoken.kernel import moe_sum_reduce_triton
    from freetoken.moe import fused_nvfp4 as fnv4

    M, H = hidden_states.shape
    top_k = topk_ids.shape[1]

    _STATS["calls"] += 1
    split = plan_fat_split(
        topk_ids.reshape(-1),
        top_k=top_k,
        num_experts=num_experts,
        min_rows=_min_rows(),
        block_m=fnv4._prefill_config(M)["BLOCK_SIZE_M"],
    )
    if split is None:
        _STATS["fallback_calls"] += 1
        return fnv4.fused_experts_nvfp4(
            hidden_states, gate_up_packed, gate_up_scale, gate_up_global,
            down_packed, down_scale, down_global,
            topk_weights, topk_ids, num_experts, activation,
            apply_router_weight_on_input, act_alpha, act_limit,
        )

    two_i = gate_up_packed.shape[1]
    inter = two_i // 2
    dev, dt = hidden_states.device, hidden_states.dtype
    cfg = fnv4._prefill_config(M)

    _STATS["busy_experts"] += len(split.busy)
    _STATS["light_experts"] += len(split.light_expert_ids.unique()) if split.light_ntpp else 0
    busy_routes = sum(int(p.numel()) for p in split.busy_positions)
    _STATS["busy_routes"] += busy_routes
    _STATS["light_routes"] += topk_ids.numel() - busy_routes

    tw = topk_weights.reshape(-1).contiguous()
    num_valid = topk_ids.numel()
    out = torch.zeros_like(hidden_states)

    if split.light_ntpp:
        # Light routes: production grouped kernel over a host-built align, with
        # the production config and gemm binding (so the wide-load prototype
        # rebinding composes). Light rows are bitwise identical to the status
        # quo; only busy rows leave the grouped path.
        sorted_ids = split.light_sorted_ids.to(dev, non_blocking=True)
        expert_ids = split.light_expert_ids.to(dev, non_blocking=True)
        ntpp = torch.tensor([split.light_ntpp], dtype=torch.int32, device=dev)
        ic1 = torch.empty((M, top_k, two_i), device=dev, dtype=dt)
        fnv4._prefill_gemm(
            hidden_states, gate_up_packed, gate_up_scale, gate_up_global, ic1,
            tw, sorted_ids, expert_ids, ntpp, num_valid, top_k,
            apply_router_weight_on_input, cfg,
        )
        ic2 = torch.empty((M * top_k, inter), device=dev, dtype=dt)
        _gated_act(activation, ic1.view(-1, two_i), ic2, alpha=act_alpha, limit=act_limit)
        # zeros: sum-reduce also reads busy/padding rows, which gemm2 never writes
        ic3 = torch.zeros((M, top_k, H), device=dev, dtype=dt)
        fnv4._prefill_gemm(
            ic2, down_packed, down_scale, down_global, ic3,
            tw, sorted_ids, expert_ids, ntpp, num_valid, 1,
            not apply_router_weight_on_input, cfg,
        )
        moe_sum_reduce_triton(ic3, out)

    for e, pos_cpu in zip(split.busy, split.busy_positions):
        pos = pos_cpu.to(dev, non_blocking=True)
        tok = torch.div(pos, top_k, rounding_mode="floor")
        x_e = hidden_states.index_select(0, tok)
        w_e = tw.index_select(0, pos).to(dt)
        rows = _expert_rows_out(
            x_e,
            dequant_nvfp4_weight(gate_up_packed[e], gate_up_scale[e], gate_up_global[e]),
            dequant_nvfp4_weight(down_packed[e], down_scale[e], down_global[e]),
            w_e, activation, apply_router_weight_on_input, act_alpha, act_limit,
        )
        out.index_add_(0, tok, rows)
    return out


def dispatch_prefill(
    hidden_states: torch.Tensor,
    gate_up_packed: torch.Tensor,
    gate_up_scale: torch.Tensor,
    gate_up_global: torch.Tensor,
    down_packed: torch.Tensor,
    down_scale: torch.Tensor,
    down_global: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_experts: int,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    act_alpha: float = 1.0,
    act_limit: float = float("inf"),
) -> torch.Tensor:
    """Env gate: flag off -> the production grouped path, byte-identical."""
    args = (
        hidden_states, gate_up_packed, gate_up_scale, gate_up_global,
        down_packed, down_scale, down_global,
        topk_weights, topk_ids, num_experts, activation,
        apply_router_weight_on_input, act_alpha, act_limit,
    )
    if not _flag_on():
        from freetoken.moe.fused_nvfp4 import fused_experts_nvfp4

        return fused_experts_nvfp4(*args)
    return fused_experts_nvfp4_fat(*args)


# ---------------------------------------------------------------------------
# Selftest: ``python -m freetoken.moe.prefill_fat_gemm``
# ---------------------------------------------------------------------------


def _synthetic_case(device: str, dtype: torch.dtype, *, e: int = 8, h: int = 128,
                    i: int = 64, top_k: int = 4, m: int = 64, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    banks = dict(
        gate_up_packed=torch.randint(0, 256, (e, 2 * i, h // 2), generator=g, dtype=torch.uint8),
        gate_up_scale=(torch.rand(e, 2 * i, h // 16, generator=g) * 0.1).to(torch.float8_e4m3fn),
        gate_up_global=torch.rand(e, 2 * i, generator=g).to(torch.float16) + 0.5,
        down_packed=torch.randint(0, 256, (e, h, i // 2), generator=g, dtype=torch.uint8),
        down_scale=(torch.rand(e, h, i // 16, generator=g) * 0.1).to(torch.float8_e4m3fn),
        down_global=torch.rand(e, h, generator=g).to(torch.float16) + 0.5,
    )
    x = torch.randn(m, h, generator=g).to(dtype)
    # zipf-ish routing so a few experts clear the busy threshold
    ranks = torch.arange(1, e + 1, dtype=torch.float64)
    probs = (1.0 / ranks).numpy()
    probs /= probs.sum()
    ids = torch.multinomial(torch.tensor(probs, dtype=torch.float32), m * top_k,
                            replacement=True, generator=g).reshape(m, top_k)
    w = torch.rand(m, top_k, generator=g)
    w = w / w.sum(-1, keepdim=True)
    return x, ids, w, banks


def _naive_reference(x, ids, w, banks, *, min_rows: int, activation="silu",
                     apply_router_weight_on_input=False, alpha=1.0, limit=float("inf")):
    """Per-route fp32 loop: the independent truth the fat assembly is checked against."""
    m, top_k = ids.shape
    out = torch.zeros(m, x.shape[1], dtype=torch.float32)
    w1 = dequant_nvfp4_weight(banks["gate_up_packed"], banks["gate_up_scale"], banks["gate_up_global"])
    w2 = dequant_nvfp4_weight(banks["down_packed"], banks["down_scale"], banks["down_global"])
    counts = torch.bincount(ids.reshape(-1), minlength=w1.shape[0])
    for t in range(m):
        for k in range(top_k):
            e = int(ids[t, k])
            if int(counts[e]) < min_rows:
                continue  # light routes are the grouped path's business
            ic1 = x[t].float() @ w1[e].T
            wt = float(w[t, k])
            if apply_router_weight_on_input:
                ic1 = ic1 * wt
            d = ic1.numel() // 2
            gate, up = ic1[:d], ic1[d:]
            if limit != float("inf"):
                gate = gate.clamp(max=limit)
                up = up.clamp(-limit, limit)
                act = gate * torch.sigmoid(alpha * gate) * up
            else:
                act = torch.nn.functional.silu(gate) * up
            row = act @ w2[e].T
            out[t] += row if apply_router_weight_on_input else row * wt
    return out


def _fat_forward_all_busy(x, ids, w, banks, *, min_rows: int, activation="silu",
                          apply_router_weight_on_input=False, alpha=1.0, limit=float("inf")):
    """The fat path with the grouped half bypassed: every busy expert through
    ``_expert_rows_out`` (any device). Shares the assembly's per-expert math."""
    m, top_k = ids.shape
    out = torch.zeros(m, x.shape[1], dtype=x.dtype, device=x.device)
    flat = ids.reshape(-1)
    split = plan_fat_split(flat, top_k=top_k, num_experts=banks["gate_up_packed"].shape[0],
                           min_rows=min_rows, block_m=8)
    assert split is not None
    w1 = dequant_nvfp4_weight(banks["gate_up_packed"], banks["gate_up_scale"], banks["gate_up_global"])
    w2 = dequant_nvfp4_weight(banks["down_packed"], banks["down_scale"], banks["down_global"])
    tw = w.reshape(-1)
    for e, pos in zip(split.busy, split.busy_positions):
        tok = torch.div(pos, top_k, rounding_mode="floor")
        rows = _expert_rows_out(
            x.index_select(0, tok.to(x.device)), w1[e], w2[e],
            tw.reshape(-1)[pos].to(x.dtype).to(x.device),
            activation, apply_router_weight_on_input, alpha, limit,
        )
        out.index_add_(0, tok.to(x.device), rows)
    return out


def main() -> int:
    """Fidelity selftest. CPU portion always runs; the bf16 GPU gate runs when
    CUDA is present. THE FLAG MUST NOT LEAVE EXPERIMENTAL until the GPU gate
    passes on the production geometry (port-plan 2c, D061 lesson)."""
    ok = True
    x, ids, w, banks = _synthetic_case("cpu", torch.float32)
    ref = _naive_reference(x, ids, w, banks, min_rows=4)
    fat = _fat_forward_all_busy(x, ids, w, banks, min_rows=4)
    gate = fidelity_gate(fat, ref)
    print(f"[cpu/fp32] fat vs naive reference: {gate}")
    ok &= bool(gate["passed"])

    if torch.cuda.is_available():
        from freetoken.moe.fused_nvfp4 import fused_experts_nvfp4

        xg = x.to("cuda", torch.bfloat16)
        idsg, wg = ids.to("cuda"), w.to("cuda")
        banks_g = {k: v.to("cuda") for k, v in banks.items()}
        e = banks["gate_up_packed"].shape[0]
        ref_g = fused_experts_nvfp4(xg, banks_g["gate_up_packed"], banks_g["gate_up_scale"],
                                    banks_g["gate_up_global"], banks_g["down_packed"],
                                    banks_g["down_scale"], banks_g["down_global"],
                                    wg, idsg, e)
        fat_g = fused_experts_nvfp4_fat(xg, banks_g["gate_up_packed"], banks_g["gate_up_scale"],
                                        banks_g["gate_up_global"], banks_g["down_packed"],
                                        banks_g["down_scale"], banks_g["down_global"],
                                        wg, idsg, e)
        gate = fidelity_gate(fat_g, ref_g)
        print(f"[gpu/bf16] fat vs production grouped: {gate}")
        ok &= bool(gate["passed"])
    else:
        print("[gpu/bf16] CUDA unavailable; GPU half of the gate DEFERRED to the battery")
    print("FIDELITY SELFTEST:", "PASS" if ok else "FAIL -- do not enable beyond the experiment flag")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_MIN_ROWS",
    "ENV_FLAG",
    "ENV_MIN_ROWS",
    "FIDELITY_REL_ERR_MAX",
    "FIDELITY_TOP1_MIN",
    "FatSplit",
    "dequant_nvfp4_weight",
    "dispatch_prefill",
    "fat_gemm_stats",
    "fidelity_gate",
    "fused_experts_nvfp4_fat",
    "plan_fat_split",
    "rel_err",
    "reset_fat_gemm_stats",
    "top1_agreement",
]
