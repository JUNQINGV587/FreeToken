"""Query-axis blocking for the Lightning-Indexer's prefill pass.

Every intermediate the prefill indexer builds is O(seqlen x n_blocks), where ``n_blocks`` grows
with the CONTEXT (``context / ratio``) while ``seqlen`` grows with the CHUNK: the fp32 score
matrix, the causal-mask compare, the bool candidate mask, and -- the largest single term -- the
int64 ``arange(n_blocks).repeat(seqlen, 1)`` row grid the causal mask used to build.

Measured on the 2xL20 dsv41f box, the whole set costs ~12.5 B per (chunk x context) element, so a
25k-token chunk against a 105k-token context wants ~33 GiB of transient -- more than the box has
free after CUDA graphs. That is what forced the adaptive chunk policy to shrink long-context
chunks, and a shrunken chunk is expensive for a different reason: the MoE disk tier re-fetches
each layer's routed expert union once per prefill *pass* with O_DIRECT, so a 105k-token prompt in
~8k chunks reads the disk-resident expert set 13 times (~1,158 GiB, measured) where 24.5k chunks
need 5 passes.

Blocking the QUERY axis bounds each of those terms by ``block x n_blocks`` instead of
``seqlen x n_blocks``, while the MoE part of the prefill keeps running on the full chunk -- which
is the point, because the expert-fetch cost is per pass, not per token.

``plan_query_block`` returns 0 (no split) whenever the whole chunk already fits the budget: the
common path keeps its exact single-call arithmetic and one kernel launch per indexer, and only
the (chunk x context) cases that would otherwise be deadly are chunked.
"""

from __future__ import annotations

import os

# Transient bytes per (query row x scored block) in the prefill indexer, as measured end to end
# (the ~12.5 B per chunk x context element above, with n_blocks = context / 4). Broken down: fp32
# scores 4 B, causal-mask compare 1 B, bool candidate mask 1 B, and the masked/selected copies.
BYTES_PER_ELEMENT = 12.5

# The prefill indexer's transient budget. 512 MiB keeps every term comfortably inside the ~20 GiB
# the box has free after graphs while leaving the split coarse enough that the per-block kernel
# launches stay a rounding error next to the disk fetch (a 24,576-token chunk at a 105k context
# becomes ~24 blocks -- each one launch, versus one disk pass per chunk).
DEFAULT_BUDGET_BYTES = 512 << 20

MIN_QUERY_BLOCK = 256
MAX_QUERY_BLOCK = 4096

_ENV = "FREETOKEN_INDEXER_SUBBLOCK_BYTES"


def subblock_budget_bytes() -> int:
    """The configured query-block budget (0 disables blocking entirely)."""
    raw = os.environ.get(_ENV)
    if raw is None or raw == "":
        return DEFAULT_BUDGET_BYTES
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_BUDGET_BYTES
    return max(0, value)


def plan_query_block(
    seqlen: int, n_blocks: int, budget_bytes: int | None = None
) -> int:
    """Query rows to score per indexer call, or 0 when the chunk needs no split.

    The result is a power of two in ``[MIN_QUERY_BLOCK, MAX_QUERY_BLOCK]`` (the scoring kernel
    launches one program per query row, so any size is correct, but a power of two keeps the
    launch geometry clean), and it is only ever returned when it actually splits ``seqlen``.
    """
    budget = subblock_budget_bytes() if budget_bytes is None else max(0, int(budget_bytes))
    seqlen = max(int(seqlen), 0)
    n_blocks = max(int(n_blocks), 0)
    if budget <= 0 or seqlen == 0 or n_blocks == 0:
        return 0
    if seqlen * n_blocks * BYTES_PER_ELEMENT <= budget:
        return 0
    rows = int(budget // max(1.0, n_blocks * BYTES_PER_ELEMENT))
    rows = max(MIN_QUERY_BLOCK, min(MAX_QUERY_BLOCK, rows))
    rows = 1 << (rows.bit_length() - 1)
    if rows >= seqlen:
        return 0
    return max(MIN_QUERY_BLOCK, rows)


__all__ = [
    "BYTES_PER_ELEMENT",
    "DEFAULT_BUDGET_BYTES",
    "MAX_QUERY_BLOCK",
    "MIN_QUERY_BLOCK",
    "plan_query_block",
    "subblock_budget_bytes",
]
