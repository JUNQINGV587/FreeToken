"""Prefill/decode split of the QSA block-scoring kernel (vllm #54513 port).

``qsa_mqa_paged_prefill`` packs TILE_R=64 rows into one tensor-core dot where the
legacy per-row ``qsa_mqa_paged`` runs a [BLOCK_N, 16] mma per row — 6.3x faster on a
8k-token extend at 256k context on L20, 2.2x on the production 12k cold prefill.
The decode path keeps the legacy kernel bit-for-bit (CUDA-graph captured, dql is
always 1). These tests pin the prefill kernel against the legacy one as oracle:
visible-block counts bitwise, logits bitwise (empirically exact on sm89 — the fp32
dot K-reduction order survives the N-width change), top-k selection identical, and
logits-chunk boundaries cutting through a request handled by the cu_seqlens clamp.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

from freetoken.kernel.triton.qsa import qsa_mqa_paged, qsa_mqa_paged_prefill  # noqa: E402

RATIO, HEADS, DIM, PAGE = 16, 4, 128, 4  # cmp_page_size=4, production head_dim=128
NPAGES, WIDTH = 512, 128


def _build(seqs: list[int], extend_lens: list[int], seed: int = 7):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    bs = len(seqs)
    k_cache = torch.randn(NPAGES, PAGE, 1, DIM, dtype=torch.bfloat16, device="cuda", generator=gen)
    block_table = torch.stack(
        [torch.randperm(NPAGES, device="cuda", generator=gen)[:WIDTH] for _ in range(bs)]
    ).to(torch.int32)
    cu = torch.tensor([0] + list(torch.tensor(extend_lens).cumsum(0)), dtype=torch.int32, device="cuda")
    positions = torch.cat(
        [torch.arange(length - ext, length, dtype=torch.int32) for length, ext in zip(seqs, extend_lens)]
    ).to("cuda")
    seq_lens = torch.tensor(seqs, dtype=torch.int32, device="cuda")
    token_to_req = torch.repeat_interleave(
        torch.arange(bs, dtype=torch.int32), torch.tensor(extend_lens, dtype=torch.int32)
    ).to("cuda")
    rows = int(sum(extend_lens))
    q = torch.randn(rows, HEADS, DIM, dtype=torch.bfloat16, device="cuda", generator=gen)
    return q, k_cache, block_table, cu, positions, seq_lens, token_to_req, rows


@pytest.mark.parametrize(
    "seqs,ext,chunk_rows",
    [
        ([1000, 4096, 333], [37, 300, 128], 96),  # chunk cuts through requests
        ([8192], [1024], 4096),
        ([64, 128], [64, 128], 256),
        ([5000, 5000], [1, 1], 8),  # degenerate single-token extends
    ],
)
def test_prefill_score_matches_legacy_oracle(seqs, ext, chunk_rows):
    q, k_cache, block_table, cu, positions, seq_lens, token_to_req, rows = _build(seqs, ext)
    columns = WIDTH * PAGE

    logits_ref = torch.full((rows, columns), float("nan"), dtype=torch.float32, device="cuda")
    vis_ref = torch.full((rows,), -1, dtype=torch.int32, device="cuda")
    qsa_mqa_paged(q, k_cache, block_table, token_to_req, positions, seq_lens, RATIO, logits_ref, vis_ref)

    logits_new = torch.full((rows, columns), float("nan"), dtype=torch.float32, device="cuda")
    vis_new = torch.full((rows,), -1, dtype=torch.int32, device="cuda")
    for start in range(0, rows, chunk_rows):
        end = min(start + chunk_rows, rows)
        lg = torch.full((end - start, columns), float("nan"), dtype=torch.float32, device="cuda")
        vs = torch.full((end - start,), -1, dtype=torch.int32, device="cuda")
        qsa_mqa_paged_prefill(
            q, k_cache, block_table, cu, positions, seq_lens, RATIO, lg, vs,
            query_offset=start, num_rows=end - start, max_query_len=max(ext),
        )
        logits_new[start:end], vis_new[start:end] = lg, vs

    assert torch.equal(vis_ref, vis_new), "visible-block counts diverged"
    in_range = torch.arange(columns, device="cuda").unsqueeze(0) < vis_ref.unsqueeze(1)
    assert torch.equal(logits_ref[in_range], logits_new[in_range]), (
        "prefill logits not bitwise-identical to the legacy kernel "
        f"(max |d| = {(logits_ref[in_range] - logits_new[in_range]).abs().max().item():.3e})"
    )
    topk = 32
    chosen_ref = logits_ref.masked_fill(~in_range, -float("inf")).topk(topk, dim=-1).indices
    chosen_new = logits_new.masked_fill(~in_range, -float("inf")).topk(topk, dim=-1).indices
    assert torch.equal(chosen_ref, chosen_new), "top-k block selection diverged"


def test_prefill_score_empty_and_bounds():
    # zero-row chunk is a no-op; out-of-range chunk rejected
    q, k_cache, block_table, cu, positions, seq_lens, _, rows = _build([128], [16])
    columns = WIDTH * PAGE
    logits = torch.empty((0, columns), dtype=torch.float32, device="cuda")
    visible = torch.empty((0,), dtype=torch.int32, device="cuda")
    qsa_mqa_paged_prefill(
        q, k_cache, block_table, cu, positions, seq_lens, RATIO, logits, visible,
        query_offset=0, num_rows=0, max_query_len=16,
    )
    with pytest.raises(ValueError):
        qsa_mqa_paged_prefill(
            q, k_cache, block_table, cu, positions, seq_lens, RATIO,
            torch.empty((1, columns), dtype=torch.float32, device="cuda"),
            torch.empty((1,), dtype=torch.int32, device="cuda"),
            query_offset=rows, num_rows=1, max_query_len=16,
        )
