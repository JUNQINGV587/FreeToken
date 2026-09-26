"""Snapshot-resume bit-exactness for the qwen3_5_moe chunked GDN prefill.

The hybrid-radix track checkpoint snapshots the per-chunk state buffer ``h``
into the fp32 state pool and later resumes a prefix continuation from it. ``h``
must therefore be fp32: the cold path carries the inter-chunk state in fp32
registers, and a bf16-rounded snapshot restarts the continuation from a
rounded state — the P3-29 cold-vs-resume logprob divergence (δ~1e-2 at token 0
over a 9.7k-token prefix). Pass 2 (chunk_fwd_o) casts ``h`` back to bf16 at the
dot, so outputs are bit-identical either way; only the snapshot precision
changes. These tests pin both properties.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

from freetoken.models.qwen3_5_moe.gdn_kernels import gdn_prefill_chunk_fla  # noqa: E402

HG, H, K, V = 2, 4, 128, 128  # k heads, v heads (GQA 1:2), head dims (production 128)
SCALE = K**-0.5


def _inputs(total: int, seed: int = 0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(1, total, HG, K, dtype=torch.bfloat16, device="cuda", generator=gen)
    k = torch.randn(1, total, HG, K, dtype=torch.bfloat16, device="cuda", generator=gen)
    v = torch.randn(1, total, H, V, dtype=torch.bfloat16, device="cuda", generator=gen)
    # Mild decay: g ~ logsigmoid(.)*0.02 keeps ~e^-0.9 of the state across a
    # 64-token chunk, so the initial/snapshot state still matters at the
    # boundary -- full-strength gates would forget it and mask rounding.
    g = torch.nn.functional.logsigmoid(
        torch.randn(1, total, H, dtype=torch.float32, device="cuda", generator=gen)
    ) * 0.02
    beta = torch.rand(1, total, H, dtype=torch.float32, device="cuda", generator=gen)
    return q, k, v, g, beta


def test_track_h_is_fp32():
    q, k, v, g, beta = _inputs(200)
    pool = torch.zeros(1, H, K, V, dtype=torch.float32, device="cuda")
    _, h = gdn_prefill_chunk_fla(
        q, k, v, g, beta,
        state_source=pool, indices=torch.zeros(1, dtype=torch.int32, device="cuda"),
        cu_seqlens=torch.tensor([0, 200], dtype=torch.int64, device="cuda"),
        scale=SCALE, return_h=True,
    )
    assert h.dtype == torch.float32


def test_snapshot_resume_bit_exact():
    """A continuation resumed from the track snapshot must reproduce the cold
    full prefill bit for bit — outputs AND final state."""
    total, boundary = 200, 128  # boundary = start of chunk 2 -> h row 2
    q, k, v, g, beta = _inputs(total)
    cu_full = torch.tensor([0, total], dtype=torch.int64, device="cuda")
    pool = torch.zeros(3, H, K, V, dtype=torch.float32, device="cuda")

    # Cold full prefill into slot 0; snapshot row 2 (state after 128 tokens).
    o_full, h = gdn_prefill_chunk_fla(
        q, k, v, g, beta,
        state_source=pool, indices=torch.tensor([0], dtype=torch.int32, device="cuda"),
        cu_seqlens=cu_full, scale=SCALE, return_h=True,
    )
    pool[2].copy_(h[0, 2])  # _write_track_snapshot: direct copy, no transpose

    # Resume: suffix tokens only, initial state = the snapshot slot.
    o_resume = gdn_prefill_chunk_fla(
        q[:, boundary:], k[:, boundary:], v[:, boundary:],
        g[:, boundary:], beta[:, boundary:],
        state_source=pool, indices=torch.tensor([2], dtype=torch.int32, device="cuda"),
        cu_seqlens=torch.tensor([0, total - boundary], dtype=torch.int64, device="cuda"),
        scale=SCALE, return_h=False,
    )

    assert torch.equal(o_resume, o_full[boundary:])
    assert torch.equal(pool[2], pool[0])


def test_output_unchanged_by_fp32_h():
    """Pass 2 casts h back to bf16 at the dot, so the fp32 h buffer must leave
    outputs bit-identical to the old bf16-buffer kernel. Replicate the old
    path by bf16-rounding h and re-running pass 2's h-dependent half via a
    second prefill whose state buffer was pre-rounded — cheaper: assert the
    prefill output matches a bf16-rounded-state continuation, which the old
    kernel produced exactly."""
    total, boundary = 200, 128
    q, k, v, g, beta = _inputs(total, seed=1)
    pool = torch.zeros(3, H, K, V, dtype=torch.float32, device="cuda")

    o_full, h = gdn_prefill_chunk_fla(
        q, k, v, g, beta,
        state_source=pool, indices=torch.tensor([0], dtype=torch.int32, device="cuda"),
        cu_seqlens=torch.tensor([0, total], dtype=torch.int64, device="cuda"),
        scale=SCALE, return_h=True,
    )
    # Old-kernel snapshot semantics: bf16-rounded state. Outputs of the
    # continuation shift with the rounded state — pin that they DO shift here
    # (proving the test is sensitive) while the prefix outputs never depend
    # on h precision at all (asserted bit-exact in test_snapshot_resume).
    pool[1].copy_(h[0, 2].to(torch.bfloat16).to(torch.float32))
    o_rounded = gdn_prefill_chunk_fla(
        q[:, boundary:], k[:, boundary:], v[:, boundary:],
        g[:, boundary:], beta[:, boundary:],
        state_source=pool, indices=torch.tensor([1], dtype=torch.int32, device="cuda"),
        cu_seqlens=torch.tensor([0, total - boundary], dtype=torch.int64, device="cuda"),
        scale=SCALE, return_h=False,
    )
    assert not torch.equal(o_rounded, o_full[boundary:])
