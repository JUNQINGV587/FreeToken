"""Adaptive prefill chunk for the DSV4/DSV4.1 sliding-window models.

Cold prefill on these models is disk-bound: the MoE disk tier fetches each layer's routed expert
union with O_DIRECT (no page-cache help) once per prefill *pass*, so a prompt split into N passes
reads roughly N times the per-pass union. Measured on the 2xL20 box, the same 23,671-token prompt
(a window pool of 160 pages, ``--prefill-chunk-tokens 8192``) took 257.8 s and read ~354 GiB from
nvme0n1 in 3 passes, while a 24,576-token chunk (416 window pages) took 125.5 s and 142.7 GiB in a
single pass -- 2.05x faster, 2.4x less disk, and the engine logged exactly one ``Prefill batch``
line for the whole prompt. See notes/freetoken/20261001-dsv41-cold-prefill-chunk-merge.md.

That larger chunk cannot simply be pinned statically: the prefill indexer's transients are
O(chunk x context) -- a 105k-token context with an 8,192-token chunk already peaked at 45,475 of
49,140 MiB on GPU0 (3.6 GiB of headroom, measured; see
notes/freetoken/20260930-dsv41-prefill-chunk-sweep.md). Two things changed that, and this module
follows them:

  1. ``attention/indexer_memory.py`` scores the indexer's QUERY axis in sub-blocks, so its peak is
     bounded by the sub-block rather than by the whole chunk;
  2. ``dsv41_indexer.indexer_select_prefill`` masks by broadcast instead of materialising an int64
     row grid, which removed the largest single term for every caller.

With those in place the only remaining chunk x context term is the bool candidate mask (one column
per ``ratio`` positions), so a long context may take the big chunk as long as that mask stays
inside ``MASK_BUDGET_BYTES``: a 105,241-token prompt then runs 5 passes instead of 13, and a short
prompt still gets the whole ceiling in one pass. When the sub-blocking is switched off
(``FREETOKEN_INDEXER_SUBBLOCK_BYTES=0``) the conservative ``SAFE_PRODUCT`` this box has actually
run applies instead.
"""

from __future__ import annotations

import os

from freetoken.attention.indexer_memory import subblock_budget_bytes

# The envelope measured safe on this box: 8,192 tokens of chunk against a 105,000-token context,
# with 3.6 GiB of the fastest card still free. Used whenever the indexer's query axis is NOT
# sub-blocked (``FREETOKEN_INDEXER_SUBBLOCK_BYTES=0``), i.e. when the score matrix is still
# O(chunk x context) and a pass whose chunk x context stays under this product requests no more
# indexer transient than a configuration that has already run to completion on 105,241-token
# prompts.
SAFE_CHUNK_TOKENS = 8192
SAFE_CONTEXT_TOKENS = 105_000
SAFE_PRODUCT = SAFE_CHUNK_TOKENS * SAFE_CONTEXT_TOKENS

# The chunk a long context is allowed to take once the score matrix is sub-blocked: the same
# 24,576-token chunk that turned a 23,671-token cold prefill from 3 passes (257.8 s, ~354 GiB read)
# into 1 pass (121.6 s, 141.6 GiB) in production.
BIG_CHUNK_TOKENS = 24_576

# What still scales with the chunk at a long context is the bool candidate mask: one column per
# compressed block, i.e. ``chunk * context / ratio`` bytes. Budget it at what an already-proven
# configuration produces -- BIG_CHUNK_TOKENS against SAFE_CONTEXT_TOKENS, 615 MiB -- which makes
# the product BIG_CHUNK_TOKENS * SAFE_CONTEXT_TOKENS (2.58 Gi tokens^2, a 3x lift over
# SAFE_PRODUCT). A 105,241-token context then still takes 24,519 tokens per pass (5 passes for a
# 105k prompt instead of 13), and the MIN_CHUNK_TOKENS floor survives to a 1.2M-token context.
MASK_RATIO = 4
MASK_BUDGET_BYTES = BIG_CHUNK_TOKENS * SAFE_CONTEXT_TOKENS // MASK_RATIO
BIG_CHUNK_PRODUCT = MASK_BUDGET_BYTES * MASK_RATIO


def mask_budget_bytes() -> int:
    """Candidate-mask byte budget, overridable at call time via
    ``FREETOKEN_PREFILL_MASK_BUDGET_MB`` (integer MiB, ``> 0``; anything else falls back to the
    compiled-in ``MASK_BUDGET_BYTES``).

    The mask budget is one of the three caps on a prefill pass (alongside the
    ``--prefill-chunk-tokens`` ceiling and the window pool's ``prefill_chunk_budget``), and all
    three must move together: raising the SWA window pool (``--swa-num-pages-override``) to buy a
    bigger chunk is pointless unless the mask budget rises in step -- at a 105k context the mask
    envelope ``budget * MASK_RATIO // context`` is the binding cap, not the pool. The env is read
    on every call so a restarted engine (or a test) sees the current value. See
    notes/freetoken/20261003-dsv41-swa-pool-envelope.md for the account.
    """
    raw = os.environ.get("FREETOKEN_PREFILL_MASK_BUDGET_MB", "")
    if raw:
        try:
            mb = int(raw)
        except ValueError:
            mb = 0
        if mb > 0:
            return mb * 1_048_576
    return MASK_BUDGET_BYTES

# Below this the pass count (and therefore the disk-tier amplification) grows faster than the
# indexer transient falls, so the rule stops shrinking. The usable context of this deployment is
# bounded by the window pool well before SAFE_PRODUCT // MIN_CHUNK_TOKENS (~420k tokens), so the
# floor only ever prevents a pathological prompt from degenerating into thousands of disk passes.
MIN_CHUNK_TOKENS = 2048


def chunk_context_product(context_len: int) -> int:
    """``chunk x context`` budget for a prompt spanning ``context_len`` tokens.

    The big product applies only while the query-axis sub-blocking is enabled; with it disabled
    the score matrix is O(chunk x context) again and the conservative envelope must hold. The
    envelope is context-independent by construction -- ``context_len`` is taken so a call reads as
    the policy it implements, and to leave a seam for an envelope that does vary with context.
    """
    if subblock_budget_bytes() > 0:
        return mask_budget_bytes() * MASK_RATIO
    return SAFE_PRODUCT


def adaptive_prefill_budget(
    context_len: int,
    ceiling: int,
    *,
    safe_product: int | None = None,
    min_chunk: int = MIN_CHUNK_TOKENS,
) -> int:
    """Prefill pass budget for a prompt whose longest request spans ``context_len`` tokens.

    ``ceiling`` is the static cap (``min(--prefill-chunk-tokens, pool budget)``) and is also what
    sizes the buffers, warmup lengths and pynccl scratch, so the result never exceeds it. The
    result is never negative and never zero for a non-negative ceiling.

    The safety property the tests pin: whenever the ceiling allows it, ``budget * context_len <=
    safe_product`` -- i.e. short prompts may take the whole ceiling, long prompts are cut back
    exactly as far as the indexer transient (or the candidate mask) requires.
    """
    ceiling = max(int(ceiling), 0)
    if ceiling == 0:
        return 0
    ctx = max(int(context_len), 1)
    product = chunk_context_product(ctx) if safe_product is None else max(int(safe_product), 0)
    cap = product // ctx
    return min(ceiling, max(min_chunk, cap))
