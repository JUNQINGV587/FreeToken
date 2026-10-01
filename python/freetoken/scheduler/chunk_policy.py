"""Adaptive prefill chunk for the DSV4/DSV4.1 sliding-window models.

Cold prefill on these models is disk-bound: the MoE disk tier fetches each layer's routed expert
union with O_DIRECT (no page-cache help) once per prefill *pass*, so a prompt split into N passes
reads roughly N times the per-pass union. Measured on the 2xL20 box, the same 23,671-token prompt
(a window pool of 160 pages, ``--prefill-chunk-tokens 8192``) took 257.8 s and read ~354 GiB from
nvme0n1 in 3 passes, while a 24,576-token chunk (416 window pages) took 125.5 s and 142.7 GiB in a
single pass -- 2.05x faster, 2.4x less disk, and the engine logged exactly one ``Prefill batch``
line for the whole prompt. See notes/freetoken/20261001-dsv41-cold-prefill-chunk-merge.md.

That larger chunk cannot simply be pinned statically: ``dsv41_indexer.indexer_select_prefill``
allocates an O(chunk x context) causal-mask transient, and a 105k-token context with an 8,192-token
chunk already peaked at 45,475 of 49,140 MiB on GPU0 (3.6 GiB of headroom, measured; see
notes/freetoken/20260930-dsv41-prefill-chunk-sweep.md). The rule here keeps every pass inside that
already-proven envelope -- ``chunk x context <= SAFE_PRODUCT`` -- so a short prompt gets one big
pass while a long prompt falls back to (or below) the static chunk that is known to fit.
"""

from __future__ import annotations

# The envelope measured safe on this box: 8,192 tokens of chunk against a 105,000-token context,
# with 3.6 GiB of the fastest card still free. A pass whose chunk x context stays under this
# product requests no more indexer transient than a configuration that has already run to
# completion on 105,241-token prompts.
SAFE_CHUNK_TOKENS = 8192
SAFE_CONTEXT_TOKENS = 105_000
SAFE_PRODUCT = SAFE_CHUNK_TOKENS * SAFE_CONTEXT_TOKENS

# Below this the pass count (and therefore the disk-tier amplification) grows faster than the
# indexer transient falls, so the rule stops shrinking. The usable context of this deployment is
# bounded by the window pool well before SAFE_PRODUCT // MIN_CHUNK_TOKENS (~420k tokens), so the
# floor only ever prevents a pathological prompt from degenerating into thousands of disk passes.
MIN_CHUNK_TOKENS = 2048


def adaptive_prefill_budget(
    context_len: int,
    ceiling: int,
    *,
    safe_product: int = SAFE_PRODUCT,
    min_chunk: int = MIN_CHUNK_TOKENS,
) -> int:
    """Prefill pass budget for a prompt whose longest request spans ``context_len`` tokens.

    ``ceiling`` is the static cap (``min(--prefill-chunk-tokens, pool budget)``) and is also what
    sizes the buffers, warmup lengths and pynccl scratch, so the result never exceeds it. The
    result is never negative and never zero for a non-negative ceiling.

    The safety property the tests pin: whenever the ceiling allows it, ``budget * context_len <=
    safe_product`` -- i.e. short prompts may take the whole ceiling, long prompts are cut back
    exactly as far as the indexer transient requires.
    """
    ceiling = max(int(ceiling), 0)
    if ceiling == 0:
        return 0
    ctx = max(int(context_len), 1)
    cap = safe_product // ctx
    return min(ceiling, max(min_chunk, cap))
