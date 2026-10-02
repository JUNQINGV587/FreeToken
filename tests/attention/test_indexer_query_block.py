"""Query-axis sub-blocking for the prefill indexer, and the causal selection it feeds.

Context: a long context makes ``n_blocks`` (context / ratio) large, so every prefill indexer
intermediate -- the fp32 score matrix, the causal-mask compare, the bool candidate mask -- grows
with the *chunk* times the *context*. A 24,576-token chunk against a 105k-token context wants tens
of GiB of scratch, which is what used to force long-context prefills down to ~8k chunks, and a
chunk is one full re-read of the disk-tier expert set.

``plan_query_block`` bounds those terms by scoring the query axis in sub-blocks; the MoE half of the
pass still covers the whole chunk (that is where the disk time is), and ``_publish`` already takes a
segment offset, so each sub-block publishes its own rows of the shared ``topk_idxs`` buffer.

These tests run on CPU and pin the two things that could silently corrupt a long prefill:

  * the sub-block plan itself (bounded, power of two, off by default only when told to be);
  * that scoring/selecting in sub-blocks is *identical* to doing it in one shot -- i.e. that the
    per-sub-block ``start_pos + s0`` / ``seqlen = n`` offsets and the causal mask line up, which is
    exactly what ``DeepseekV41Indexer.forward_paged`` does per sub-block.
"""

from __future__ import annotations

import pytest
import torch

from freetoken.attention.dsv41_indexer import DSV41IndexerBackendMixin
from freetoken.attention.indexer_memory import (
    BYTES_PER_ELEMENT,
    DEFAULT_BUDGET_BYTES,
    MAX_QUERY_BLOCK,
    MIN_QUERY_BLOCK,
    plan_query_block,
    subblock_budget_bytes,
)

ENV = "FREETOKEN_INDEXER_SUBBLOCK_BYTES"

# The production shape that motivated this: a 24,576-token chunk against a 105,241-token context
# (ratio 4 => 26,310 compressed blocks).
CHUNK, CONTEXT, RATIO = 24_576, 105_241, 4
BLOCKS = CONTEXT // RATIO


class _Indexer(DSV41IndexerBackendMixin):
    """The mixin needs no state: selection is a pure function of its arguments."""


def _reference_select(scores, *, start_pos, seqlen, ratio, topk, offset):
    """The pre-sub-blocking implementation, kept here as the behavioural reference (int64 grid)."""
    device = scores.device
    n_blocks = scores.shape[-1]
    live = ((start_pos + torch.arange(1, seqlen + 1, device=device)) // ratio).unsqueeze(1)
    blk = torch.arange(n_blocks, device=device).repeat(seqlen, 1)
    masked = scores + torch.where(blk >= live, float("-inf"), 0)
    picks = masked.topk(min(topk, n_blocks), dim=-1)[1]
    return torch.where(picks >= live, -1, picks + offset)


# --------------------------------------------------------------------------------------
# the plan
# --------------------------------------------------------------------------------------


def test_no_split_when_the_chunk_already_fits(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    assert DEFAULT_BUDGET_BYTES > 0
    # A 4k chunk against a short context is nowhere near the budget: one call, as before.
    assert plan_query_block(4_096, 1_000) == 0
    assert plan_query_block(24_576, 4) == 0


def test_long_context_splits_the_query_axis(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    block = plan_query_block(CHUNK, BLOCKS)
    assert block > 0
    assert block < CHUNK
    # one program per query row in the scoring kernel, so a power of two keeps the grid clean
    assert block & (block - 1) == 0
    assert MIN_QUERY_BLOCK <= block <= MAX_QUERY_BLOCK
    # the whole point: the transient is bounded by the block, not by the chunk
    assert block * BLOCKS * BYTES_PER_ELEMENT <= DEFAULT_BUDGET_BYTES
    assert CHUNK * BLOCKS * BYTES_PER_ELEMENT > DEFAULT_BUDGET_BYTES  # ...which is why it splits
    assert -(-CHUNK // block) >= 8  # several sub-blocks: this is a real split, not a rounding


def test_no_split_when_a_small_chunk_meets_the_floor():
    """The clamp can lift the block above the chunk, which means there is nothing to split."""
    assert plan_query_block(300, 10_000, budget_bytes=DEFAULT_BUDGET_BYTES) == 0  # fits outright
    assert plan_query_block(100, 10_000_000, budget_bytes=DEFAULT_BUDGET_BYTES) == 0  # floor >= n


def test_zero_budget_and_degenerate_shapes(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    assert plan_query_block(CHUNK, BLOCKS, budget_bytes=0) == 0  # disabled == no split
    assert plan_query_block(0, BLOCKS) == 0
    assert plan_query_block(CHUNK, 0) == 0  # no blocks to score


def test_tiny_budget_falls_back_to_the_floor():
    # A budget that would allow only a handful of rows clamps up to MIN_QUERY_BLOCK (a smaller
    # block means more kernel launches, not a smaller peak worth chasing).
    assert plan_query_block(CHUNK, BLOCKS, budget_bytes=10) == MIN_QUERY_BLOCK


def test_plan_is_monotone_in_the_budget():
    budgets = [128 << 20, 256 << 20, 512 << 20, 1 << 30, 4 << 30]
    blocks = [plan_query_block(CHUNK, BLOCKS, budget_bytes=b) for b in budgets]
    assert blocks == sorted(blocks)
    assert 0 < blocks[0] < blocks[-1] <= MAX_QUERY_BLOCK
    # A budget that covers the whole chunk needs no split at all.
    assert plan_query_block(CHUNK, BLOCKS, budget_bytes=16 << 30) == 0


def test_env_override_selects_the_budget(monkeypatch):
    monkeypatch.setenv(ENV, str(64 << 20))
    assert subblock_budget_bytes() == 64 << 20
    assert plan_query_block(CHUNK, BLOCKS) == plan_query_block(
        CHUNK, BLOCKS, budget_bytes=64 << 20
    )
    monkeypatch.setenv(ENV, "0")
    assert subblock_budget_bytes() == 0
    assert plan_query_block(CHUNK, BLOCKS) == 0
    monkeypatch.setenv(ENV, "")
    assert subblock_budget_bytes() == DEFAULT_BUDGET_BYTES
    monkeypatch.setenv(ENV, "junk")
    assert subblock_budget_bytes() == DEFAULT_BUDGET_BYTES  # a typo must not disable the guard
    monkeypatch.setenv(ENV, "-1")
    assert subblock_budget_bytes() == 0  # an explicit negative does disable it


# --------------------------------------------------------------------------------------
# selection equivalence
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("seqlen,n_blocks,topk,start_pos,ratio,offset", [
    (1, 1, 1, 0, 4, 0),
    (8, 5, 3, 0, 4, 0),
    (37, 23, 5, 13, 4, 0),
    (64, 40, 7, 511, 4, 12),
    (33, 129, 16, 1024, 4, 0),
    (16, 4, 8, 3, 4, 0),          # topk > live blocks: the -1 sentinel path
])
def test_mask_shortcut_matches_the_int64_reference(seqlen, n_blocks, topk, start_pos, ratio, offset):
    torch.manual_seed(seqlen * 131 + n_blocks)
    scores = torch.randn(1, seqlen, n_blocks)
    got = _Indexer().indexer_select_prefill(
        scores, start_pos=start_pos, seqlen=seqlen, ratio=ratio, topk=topk, offset=offset
    )
    want = _reference_select(
        scores, start_pos=start_pos, seqlen=seqlen, ratio=ratio, topk=topk, offset=offset
    )
    assert torch.equal(got, want)


@pytest.mark.parametrize("block", [1, 4, 16, 64])
def test_sub_blocks_compose_to_the_single_shot_result(block):
    """This mirrors ``forward_paged``: same scores, sliced; ``start_pos + s0``; ``seqlen = n``."""
    torch.manual_seed(7)
    seqlen, n_blocks, topk, start_pos, ratio = 37, 23, 5, 13, 4
    scores = torch.randn(1, seqlen, n_blocks)
    mix = _Indexer()

    whole = mix.indexer_select_prefill(
        scores, start_pos=start_pos, seqlen=seqlen, ratio=ratio, topk=topk, offset=0
    )
    parts = [
        mix.indexer_select_prefill(
            scores[:, s0 : s0 + min(block, seqlen - s0)],
            start_pos=start_pos + s0,
            seqlen=min(block, seqlen - s0),
            ratio=ratio,
            topk=topk,
            offset=0,
        )
        for s0 in range(0, seqlen, block)
    ]
    assert torch.equal(torch.cat(parts, dim=1), whole)
    # A sub-block view keeps the query axis stride, which is what the scoring kernel reads.
    view = scores[:, 4:8]
    assert view.stride(0) == scores.stride(0) and view.stride(2) == scores.stride(2)


def test_ragged_tail_of_a_sub_block_is_exact():
    """The model's last sub-block is short; a non-power-of-two row count must not shift picks."""
    torch.manual_seed(11)
    seqlen, n_blocks, ratio, topk, start_pos = 100, 31, 4, 6, 40
    scores = torch.randn(1, seqlen, n_blocks)
    mix = _Indexer()
    whole = mix.indexer_select_prefill(
        scores, start_pos=start_pos, seqlen=seqlen, ratio=ratio, topk=topk, offset=0
    )
    steps = [0, 32, 64, 96]  # 32 + 32 + 32 + 4
    parts = [
        mix.indexer_select_prefill(
            scores[:, s0:s1], start_pos=start_pos + s0, seqlen=s1 - s0, ratio=ratio, topk=topk,
            offset=0,
        )
        for s0, s1 in zip(steps, steps[1:] + [seqlen])
    ]
    assert torch.equal(torch.cat(parts, dim=1), whole)
