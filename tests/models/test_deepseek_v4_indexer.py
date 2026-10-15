"""Query sub-blocking for the DeepSeek-V4 indexer's prefill selection.

``Indexer._select_prefill`` splits the query axis when ``plan_query_block`` says the
chunk times context product would blow the indexer's transient budget. Splitting must not
change the picks: each sub-block scores its own rows with ``start_pos + s0`` and
``seqlen = n``, which is what the single-shot call did. These tests pin that on CPU -- the
scoring kernel is a deterministic stand-in, so only the split arithmetic is under test.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.attention.dsv4_indexer import IndexerBackendMixin
from freetoken.attention.indexer_memory import MAX_QUERY_BLOCK, MIN_QUERY_BLOCK
from freetoken.models.deepseek_v4 import compress
from freetoken.models.deepseek_v4.compress import Indexer

ENV = "FREETOKEN_INDEXER_SUBBLOCK_BYTES"
RATIO = 4


class _Attn(IndexerBackendMixin):
    """The real selection mixin with a deterministic stand-in for the scoring kernel."""

    def __init__(self):
        self.rows_per_call: list[int] = []

    def indexer_prefill_logits(self, q, keys, weights):
        self.rows_per_call.append(q.size(1))
        # Every live block scores the same, so the picks come from the causal mask alone;
        # the single-shot and sub-blocked paths must still agree row for row.
        # keys are [bsz, n_blocks, D] -- the block axis is -2, exactly as on the real path.
        return torch.ones(1, q.size(1), keys.size(1), dtype=torch.float32)


def _run(seqlen, n_blocks, topk, start_pos, offset, budget, monkeypatch):
    """Select once with ``budget`` bytes of indexer scratch, returning (picks, attn)."""
    monkeypatch.setenv(ENV, str(budget))
    attn = _Attn()
    # Indexer.attn is a property over the global context's backend, so the stub goes in there.
    monkeypatch.setattr(compress, "get_global_ctx", lambda: SimpleNamespace(attn_backend=attn))
    indexer = object.__new__(Indexer)
    indexer.index_topk = topk
    picks = indexer._select_prefill(
        torch.randn(1, seqlen, 8),
        torch.randn(1, n_blocks, 8),
        torch.randn(1, seqlen, 4),
        start_pos,
        seqlen,
        RATIO,
        offset,
    )
    return picks, attn


@pytest.mark.parametrize("seqlen,n_blocks,topk,start_pos,offset", [
    (1, 1, 1, 0, 0),
    (8, 5, 3, 0, 0),
    (300, 4, 8, 3, 0),        # topk > live blocks: the -1 sentinel path
    (600, 23, 5, 13, 12),
    (517, 129, 16, 1024, 0),
    (5000, 40, 7, 511, 0),    # more rows than MAX_QUERY_BLOCK: the split is uneven
])
def test_sub_blocks_are_identical_to_the_single_shot(
    seqlen, n_blocks, topk, start_pos, offset, monkeypatch
):
    whole, oracle = _run(seqlen, n_blocks, topk, start_pos, offset, 1 << 62, monkeypatch)
    assert oracle.rows_per_call == [seqlen]
    assert whole.dtype == torch.int64 and whole.shape == (1, seqlen, min(topk, n_blocks))

    split, parts = _run(seqlen, n_blocks, topk, start_pos, offset, 1, monkeypatch)
    assert sum(parts.rows_per_call) == seqlen
    assert max(parts.rows_per_call) <= MIN_QUERY_BLOCK
    assert len(parts.rows_per_call) == (1 if seqlen <= MIN_QUERY_BLOCK else -(-seqlen // MIN_QUERY_BLOCK))
    assert torch.equal(split, whole)


def test_a_chunk_that_fits_the_budget_keeps_the_single_call(monkeypatch):
    picks, attn = _run(4096, 40, 7, 0, 0, 32 << 20, monkeypatch)
    assert attn.rows_per_call == [4096]  # one kernel launch per indexer, as before
    assert picks.shape == (1, 4096, 7)


def test_max_query_block_caps_the_sub_block(monkeypatch):
    seqlen, n_blocks = 9000, 64
    whole, _ = _run(seqlen, n_blocks, 6, 0, 0, 1 << 62, monkeypatch)
    # 4 MiB of scratch: the chunk no longer fits (9000 x 64 x 12.5 B), a 4096-row block does.
    got, attn = _run(seqlen, n_blocks, 6, 0, 0, 4 << 20, monkeypatch)
    assert max(attn.rows_per_call) == MAX_QUERY_BLOCK
    assert sum(attn.rows_per_call) == seqlen
    assert torch.equal(got, whole)


def test_the_block_axis_is_read_from_the_real_three_dimensional_keys(monkeypatch):
    """Regression for the 2026-10-15 256K single-request OOM.

    On the real path ``indexer_keys`` returns ``[bsz, n_blocks, D]``, so a plan that reads
    ``keys.size(0)`` sees ONE block and never splits however long the context is -- the 256K
    request then materialised a single [4096, 37750] score grid and died. The budget here is
    chosen so that it only splits when the block axis (n_blocks) is the one being read:
    ``4096 x 1 x 12.5 B`` fits 8 MiB, ``4096 x 1024 x 12.5 B`` does not.
    """
    seqlen, n_blocks = 4096, 1024
    whole, oracle = _run(seqlen, n_blocks, 64, 0, 0, 1 << 62, monkeypatch)
    assert oracle.rows_per_call == [seqlen]

    split, parts = _run(seqlen, n_blocks, 64, 0, 0, 8 << 20, monkeypatch)
    assert len(parts.rows_per_call) == 8, parts.rows_per_call
    assert max(parts.rows_per_call) == 512
    assert sum(parts.rows_per_call) == seqlen
    assert torch.equal(split, whole)
