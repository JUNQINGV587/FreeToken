"""Pins for OUR divergences from the upstream-vendored KDA kernels.

The vendored kernels (freetoken/kernel/fla) are tested upstream and not re-tested
here. The eager reference replicates their exact math (safe gate ``gk =
lower_bound * sigmoid(exp(A_log) * (g_raw + dt_bias))``, ``beta =
sigmoid(beta_raw)``, in-loop q/k l2norm, per-channel-decayed delta rule on a
[V, K] state) so a divergence pin can assert numerics, not just reachability.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

H, D = 4, 128  # head count trimmed; head_dim matches GLM-5.3 (kernel specializes on D)
LOWER_BOUND = -5.0
SCALE = D**-0.5


def _l2norm(x: torch.Tensor) -> torch.Tensor:
    return x / torch.sqrt((x * x).sum(-1, keepdim=True) + 1e-6)


def _reference(
    q: torch.Tensor,  # [T, H, D] bf16
    k: torch.Tensor,
    v: torch.Tensor,
    g_raw: torch.Tensor,  # [T, H, D] bf16
    beta_raw: torch.Tensor,  # [T, H] bf16
    a_log: torch.Tensor,  # [H] fp32
    dt_bias: torch.Tensor,  # [H*D] fp32
    h0: torch.Tensor | None = None,  # [H, V, K] fp32
) -> tuple[torch.Tensor, torch.Tensor]:
    T = q.shape[0]
    h = (
        h0.clone().float()
        if h0 is not None
        else torch.zeros(H, D, D, dtype=torch.float32, device=q.device)
    )
    amp = a_log.float().exp().view(H, 1)
    bias = dt_bias.float().view(H, D)
    outs = []
    for t in range(T):
        gk = LOWER_BOUND * torch.sigmoid(amp * (g_raw[t].float() + bias))  # [H, K]
        h = h * gk.exp().unsqueeze(1)  # decay per k-channel: [H, V, K] * [H, 1, K]
        kt = _l2norm(k[t].float())
        v_err = v[t].float() - torch.einsum("hvk,hk->hv", h, kt)
        v_err = v_err * torch.sigmoid(beta_raw[t].float()).unsqueeze(-1)
        h = h + torch.einsum("hv,hk->hvk", v_err, kt)
        qt = _l2norm(q[t].float()) * SCALE
        outs.append(torch.einsum("hvk,hk->hv", h, qt))
    return torch.stack(outs), h


def _rand_inputs(T: int, seed: int = 0, device="cuda"):
    torch.manual_seed(seed)
    mk = lambda *s: torch.randn(*s, device=device, dtype=torch.bfloat16)
    q, k, v, g_raw = mk(T, H, D), mk(T, H, D), mk(T, H, D), mk(T, H, D)
    beta_raw = mk(T, H)
    a_log = torch.randn(H, device=device, dtype=torch.float32) * 0.5
    dt_bias = torch.randn(H * D, device=device, dtype=torch.float32) * 0.5
    return q, k, v, g_raw, beta_raw, a_log, dt_bias


def _assert_close(ours, ref, tag, atol=2e-2, rtol=2e-2):
    ours, ref = ours.float(), ref.float()
    err = (ours - ref).abs().max().item()
    rel = err / (ref.abs().max().item() + 1e-8)
    assert torch.allclose(ours, ref, atol=atol, rtol=rtol), (
        f"{tag}: max abs err {err:.5f}, rel {rel:.5f}"
    )


def test_fused_recurrent_serves_slot_zero():
    """--cache-type naive keys state by raw table_idx, so a real request can sit
    on slot 0. Upstream vLLM's kernel treats 0 as its NULL_BLOCK_ID sentinel and
    silently skips it (state frozen, garbage output); our vendored copy diverges
    to accept every non-negative slot (GDN-kernel parity). Same math as the
    parametrized reference test, just on slot 0."""
    from freetoken.kernel.fla import fused_recurrent_kda

    T = 7
    q, k, v, g_raw, beta_raw, a_log, dt_bias = _rand_inputs(T)
    ref_o, ref_h = _reference(q, k, v, g_raw, beta_raw, a_log, dt_bias)

    pool = torch.zeros(2, H, D, D, dtype=torch.float32, device="cuda")
    indices = torch.zeros((1, T), dtype=torch.int64, device="cuda")  # slot 0
    cu = torch.tensor([0, T], dtype=torch.int32, device="cuda")
    o, _ = fused_recurrent_kda(
        q=q.unsqueeze(0), k=k.unsqueeze(0), v=v.unsqueeze(0),
        g=g_raw.unsqueeze(0), beta=beta_raw.unsqueeze(0),
        initial_state=pool,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu,
        ssm_state_indices=indices,
        sigmoid_beta=True,
        a_log=a_log,
        g_bias=dt_bias,
        compute_gate=True,
        lower_bound=LOWER_BOUND,
    )
    _assert_close(o[0], ref_o, "slot-0 output")
    _assert_close(pool[0], ref_h, "slot-0 final state")
    assert pool[1].abs().max().item() == 0.0  # only slot 0 was touched


# --- P3-29 follow-up: KDA chunked prefill must also export fp32 chunk states ---

CHUNK = 64


def _chunk_inputs(total: int, seed: int = 7):
    torch.manual_seed(seed)
    mk = lambda *s: torch.randn(*s, device="cuda", dtype=torch.bfloat16)
    q, k, v = mk(1, total, H, D), mk(1, total, H, D), mk(1, total, H, D)
    raw_g = mk(1, total, H, D)
    beta = torch.sigmoid(mk(1, total, H)).float()
    a_log = torch.randn(H, device="cuda", dtype=torch.float32) * 0.5
    dt_bias = torch.randn(H * D, device="cuda", dtype=torch.float32) * 0.5
    cu = torch.tensor([0, total], dtype=torch.int32, device="cuda")
    return q, k, v, raw_g, beta, a_log, dt_bias, cu


def _run_chunk(q, k, v, raw_g, beta, a_log, dt_bias, cu, initial, return_h):
    from freetoken.kernel.fla import chunk_kda_with_fused_gate

    # chunk_kda_with_fused_gate writes its output IN-PLACE into v (the model
    # passes an ephemeral conv buffer by design); clone per call so repeated
    # runs in one test do not feed clobbered outputs back as inputs.
    return chunk_kda_with_fused_gate(
        q=q.clone(), k=k.clone(), v=v.clone(), raw_g=raw_g.clone(), beta=beta.clone(),
        A_log=a_log, g_bias=dt_bias,
        scale=SCALE, initial_state=initial, output_final_state=True,
        use_qk_l2norm_in_kernel=True, cu_seqlens=cu, safe_gate=True,
        # Mild decay (sigmoid(0)=0.5 -> |lower_bound|/2 per token); the default
        # -5.0 forgets the initial state within one chunk and would mask any
        # snapshot rounding (same lesson as the GDN probe).
        lower_bound=-0.01, return_h=return_h,
    )


def test_chunked_track_h_is_fp32():
    """Snapshot path copies h rows into the fp32 linear-state pool; h must be
    exported in fp32 so snapshots are not bf16-rounded (P3-29)."""
    total = CHUNK * 3
    q, k, v, raw_g, beta, a_log, dt_bias, cu = _chunk_inputs(total)
    initial = torch.zeros(1, H, D, D, dtype=torch.float32, device="cuda")
    _, _, h = _run_chunk(q, k, v, raw_g, beta, a_log, dt_bias, cu, initial, True)
    assert h.dtype == torch.float32, f"chunked KDA exports h as {h.dtype}"


def test_chunked_snapshot_resume_bit_exact():
    """Resume from a chunk-boundary snapshot must be bit-identical to cold."""
    total, boundary = CHUNK * 3, CHUNK * 2
    q, k, v, raw_g, beta, a_log, dt_bias, cu = _chunk_inputs(total)
    zero = torch.zeros(1, H, D, D, dtype=torch.float32, device="cuda")
    o_full, fs_full, h = _run_chunk(q, k, v, raw_g, beta, a_log, dt_bias, cu, zero, True)

    snap = h[0, 2].unsqueeze(0).contiguous()  # state after `boundary` tokens
    cu_sfx = torch.tensor([0, total - boundary], dtype=torch.int32, device="cuda")
    o_res, fs_res = _run_chunk(
        q[:, boundary:], k[:, boundary:], v[:, boundary:],
        raw_g[:, boundary:], beta[:, boundary:], a_log, dt_bias,
        cu_sfx, snap, False,
    )
    torch.testing.assert_close(o_res, o_full[:, boundary:], rtol=0, atol=0)
    torch.testing.assert_close(fs_res, fs_full, rtol=0, atol=0)


def test_chunked_resume_observes_state_rounding():
    """Sensitivity pin: a bf16-rounded snapshot MUST change the continuation --
    proves the probes above can actually detect snapshot-state rounding.
    Asserts on the fp32 final state: outputs pass through bf16 casts at every
    tl.dot, which can legitimately mask a bf16-ULP-scale state difference,
    while the register state recurrence (init * chunk decay, fp32) carries it
    straight into the fp32 final_state store."""
    total, boundary = CHUNK * 3, CHUNK * 2
    q, k, v, raw_g, beta, a_log, dt_bias, cu = _chunk_inputs(total)
    zero = torch.zeros(1, H, D, D, dtype=torch.float32, device="cuda")
    _, _, h = _run_chunk(q, k, v, raw_g, beta, a_log, dt_bias, cu, zero, True)

    cu_sfx = torch.tensor([0, total - boundary], dtype=torch.int32, device="cuda")
    sfx = (q[:, boundary:], k[:, boundary:], v[:, boundary:],
           raw_g[:, boundary:], beta[:, boundary:])
    exact = h[0, 2].unsqueeze(0).contiguous()
    rounded = h[0, 2].to(torch.bfloat16).to(torch.float32).unsqueeze(0).contiguous()
    _, fs_exact = _run_chunk(*sfx, a_log, dt_bias, cu_sfx, exact, False)
    _, fs_bad = _run_chunk(*sfx, a_log, dt_bias, cu_sfx, rounded, False)
    assert not torch.equal(fs_bad, fs_exact), (
        "rounded snapshot produced identical final state -- probe cannot detect P3-29"
    )
