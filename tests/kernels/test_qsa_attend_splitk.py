"""Split-K sparse GQA with the packed count column (vllm #54873 bit-exact subset).

FreeToken keeps contiguous split ranges (upstream's strided reassignment regroups the
fp32 softmax accumulation — not bit-exact, deferred per the determinism red line) and
adopts only the trailing count column + dead-tile clip. Dead iterations are exact
no-ops (masked -1 indices give exp2(-1e20)=0), so clipping must be bitwise-invisible:
these tests pin clipped vs unclipped equality on the same kernel, a torch reference
for correctness, and zero-output inert rows (CUDA-graph padding, count=0).
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

from freetoken.kernel.triton.qsa import qsa_sparse_paged_attention  # noqa: E402

PAGE, KV_HEADS, Q_HEADS, DIM, TOPK = 64, 1, 16, 128, 256
NPAGES = 4096


def _build(seqs: list[int], live_blocks: list[int], seed: int = 11):
    """One query row per request; selection = live_blocks full pages + -1 padding."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    rows = len(seqs)
    k_cache = torch.randn(NPAGES, PAGE, KV_HEADS, DIM, dtype=torch.bfloat16, device="cuda", generator=gen)
    v_cache = torch.randn(NPAGES, PAGE, KV_HEADS, DIM, dtype=torch.bfloat16, device="cuda", generator=gen)
    width = -(-max(seqs) // PAGE)
    block_table = torch.stack(
        [torch.randperm(NPAGES, device="cuda", generator=gen)[:width] for _ in seqs]
    ).to(torch.int32)
    q = torch.randn(rows, Q_HEADS, DIM, dtype=torch.bfloat16, device="cuda", generator=gen)
    token_to_req = torch.arange(rows, dtype=torch.int32, device="cuda")

    indices = torch.full((rows, TOPK + 1), -1, dtype=torch.int32, device="cuda")
    for r, (seq, blocks) in enumerate(zip(seqs, live_blocks)):
        toks = torch.cat(
            [torch.arange(b * PAGE, (b + 1) * PAGE, dtype=torch.int32) for b in range(blocks)]
            + [torch.arange(blocks * PAGE, seq, dtype=torch.int32)]
        ).clamp(max=seq - 1)
        toks = toks[:TOPK]
        indices[r, : toks.numel()] = toks.to("cuda")
        indices[r, TOPK] = toks.numel()
    return q, k_cache, v_cache, indices, block_table, token_to_req


def _reference(q, k_cache, v_cache, indices, block_table, token_to_req):
    outs = []
    for r in range(q.shape[0]):
        req = int(token_to_req[r])
        sel = indices[r, : int(indices[r, TOPK])].long()
        sel = sel[sel >= 0]
        pages = block_table[req].long()[sel // PAGE]
        keys = k_cache[pages, sel % PAGE, 0, :].float()   # [n, dim]
        vals = v_cache[pages, sel % PAGE, 0, :].float()
        scores = (q[r].float() @ keys.T) * DIM**-0.5      # [heads, n]
        probs = torch.softmax(scores, dim=-1)
        outs.append((probs @ vals).to(q.dtype))
    return torch.stack(outs)


def test_attend_matches_torch_reference():
    seqs, blocks = [3000, 517, 8192], [32, 8, 120]
    q, k, v, idx, bt, t2r = _build(seqs, blocks)
    out = qsa_sparse_paged_attention(q, k, v, idx, bt, t2r)
    ref = _reference(q, k, v, idx, bt, t2r)
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("rows_blocks", [([3000], [32]), ([517, 8192], [8, 120])])
def test_count_column_clip_is_bitwise_invisible(rows_blocks):
    """count=true count (clip active) vs count=TOPK (clip disabled): identical bits."""
    q, k, v, idx, bt, t2r = _build(*rows_blocks)
    idx_full = idx.clone()
    idx_full[:, TOPK] = TOPK  # clip bound becomes NUM_TILES: the pre-#54873 loop
    out_clipped = qsa_sparse_paged_attention(q, k, v, idx, bt, t2r)
    out_unclipped = qsa_sparse_paged_attention(q, k, v, idx_full, bt, t2r)
    assert torch.equal(out_clipped, out_unclipped), (
        f"dead-tile clip changed bits: max |d| = "
        f"{(out_clipped.float() - out_unclipped.float()).abs().max().item():.3e}"
    )


def test_inert_rows_produce_zero_output():
    q, k, v, idx, bt, t2r = _build([3000], [32])
    # Append a CUDA-graph padding row: request -1, count 0.
    q = torch.cat([q, torch.zeros(1, Q_HEADS, DIM, dtype=torch.bfloat16, device="cuda")])
    idx = torch.cat([idx, torch.full((1, TOPK + 1), -1, dtype=torch.int32, device="cuda")])
    idx[-1, TOPK] = 0
    t2r = torch.cat([t2r, torch.tensor([-1], dtype=torch.int32, device="cuda")])
    out = qsa_sparse_paged_attention(q, k, v, idx, bt, t2r)
    assert torch.equal(out[-1], torch.zeros_like(out[-1])), "inert row wrote non-zero output"


def test_split_k_merge_with_clip_bitwise_stable():
    """Single decode row lands in the 64-split tier: clip + merge must still match
    the unclipped merge bit-for-bit (partials of dead splits are -inf LSE either way)."""
    q, k, v, idx, bt, t2r = _build([517], [8])
    idx_full = idx.clone()
    idx_full[:, TOPK] = TOPK
    a = qsa_sparse_paged_attention(q, k, v, idx, bt, t2r)
    b = qsa_sparse_paged_attention(q, k, v, idx_full, bt, t2r)
    assert torch.equal(a, b), "clip changed split-K merge results"
    torch.testing.assert_close(a, _reference(q, k, v, idx, bt, t2r), atol=2e-2, rtol=2e-2)
