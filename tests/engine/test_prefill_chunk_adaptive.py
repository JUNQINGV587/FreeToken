"""``--prefill-chunk-adaptive``: one pass for a short prompt, the old chunk for a long one.

Cold prefill on DSV4/DSV4.1 is disk-bound. The MoE disk tier fetches each layer's routed expert
union with O_DIRECT (no page-cache help), once per prefill *pass*, so a prompt split into N passes
reads roughly N times the per-pass union: the same 23,671-token prompt took 257.8 s / ~354 GiB read
in 3 passes (8,192-token chunk, 160 window pages) and 125.5 s / 142.7 GiB in one pass (24,576-token
chunk, 416 pages) -- measured on the 2xL20 box, see
notes/freetoken/20261001-dsv41-cold-prefill-chunk-merge.md.

The big chunk cannot be pinned statically: the prefill indexer's transients are O(chunk x context),
and a 105k context already peaked at 45,475 of 49,140 MiB on GPU0 with the 8,192 chunk (see
notes/freetoken/20260930-dsv41-prefill-chunk-sweep.md). So the flag makes
``--prefill-chunk-tokens`` a *ceiling* and the pass budget becomes
``adaptive_prefill_budget(longest pending context, ceiling)``: the whole ceiling while
``chunk x context`` stays inside the envelope for the current indexer, a context-proportional chunk
beyond it.

Two envelopes exist, and the tests pin both:

  * ``SAFE_PRODUCT`` (8,192 x 105,000) is what this box has actually run -- it applies when the
    indexer's query axis is not sub-blocked (``FREETOKEN_INDEXER_SUBBLOCK_BYTES=0``), because the
    score matrix is then O(chunk x context);
  * ``BIG_CHUNK_PRODUCT`` (24,576 x 105,000, 3x larger) applies by default, because
    ``attention/indexer_memory.py`` scores the query axis in sub-blocks and
    ``indexer_select_prefill`` no longer materialises an int64 row grid. What still scales with the
    chunk at a long context is the bool candidate mask, which is what the bigger product budgets.

These tests pin the arithmetic, the boundary behaviour, the config-layer resolution (including the
branch that must NOT raise ``max_extend_tokens`` to ``max_seq_len``), and the scheduler wiring.
"""

from __future__ import annotations

from types import SimpleNamespace

from freetoken.scheduler.chunk_policy import (
    BIG_CHUNK_PRODUCT,
    BIG_CHUNK_TOKENS,
    CONTENTION_CAP_ENV,
    MASK_BUDGET_BYTES,
    MIN_CHUNK_TOKENS,
    SAFE_CHUNK_TOKENS,
    SAFE_CONTEXT_TOKENS,
    SAFE_PRODUCT,
    adaptive_prefill_budget,
    chunk_context_product,
    contention_capped_budget,
    contention_chunk_cap_tokens,
    mask_budget_bytes,
)

PAGE = 128  # the DSV4/DSV4.1 window page, i.e. the pool's currency
SUB_BLOCK_ENV = "FREETOKEN_INDEXER_SUBBLOCK_BYTES"
MASK_BUDGET_ENV = "FREETOKEN_PREFILL_MASK_BUDGET_MB"


# --------------------------------------------------------------------------------------
# the policy itself
# --------------------------------------------------------------------------------------


def test_short_prompt_takes_the_whole_ceiling():
    """The measured win: a 23,671-token prompt fits one 24,576-token pass."""
    assert adaptive_prefill_budget(23_671, 24_576) == 24_576
    assert adaptive_prefill_budget(1, 24_576) == 24_576


def test_long_prompt_keeps_the_measured_chunk_when_the_indexer_is_not_sub_blocked():
    """With ``FREETOKEN_INDEXER_SUBBLOCK_BYTES=0`` a 105k context keeps the chunk measured to fit."""
    conservative = dict(safe_product=SAFE_PRODUCT)
    assert adaptive_prefill_budget(SAFE_CONTEXT_TOKENS, 24_576, **conservative) == SAFE_CHUNK_TOKENS
    # A prompt a hair past the calibration point is cut back by exactly the ratio -- a fraction of
    # a page, not a second pass' worth of disk traffic.
    assert adaptive_prefill_budget(105_241, 24_576, **conservative) == 8173
    assert adaptive_prefill_budget(105_241, 24_576, **conservative) <= SAFE_CHUNK_TOKENS


def test_long_prompt_takes_the_big_chunk_by_default():
    """The sub-blocked indexer lifts the envelope 3x, so 105k gets 5 passes instead of 13."""
    budget = adaptive_prefill_budget(105_241, 24_576)
    assert budget == BIG_CHUNK_PRODUCT // 105_241
    assert 24_000 < budget <= BIG_CHUNK_TOKENS
    assert -(-105_241 // budget) == 5  # ceil division: five passes for the 105k prompt
    # The conservative envelope is still what an unset/zero budget falls back to.
    assert chunk_context_product(105_241) == BIG_CHUNK_PRODUCT


def test_sub_blocking_switch_selects_the_envelope(monkeypatch):
    from freetoken.attention.indexer_memory import subblock_budget_bytes

    monkeypatch.delenv(SUB_BLOCK_ENV, raising=False)
    assert subblock_budget_bytes() > 0
    assert chunk_context_product(105_241) == BIG_CHUNK_PRODUCT

    monkeypatch.setenv(SUB_BLOCK_ENV, "0")
    assert subblock_budget_bytes() == 0
    assert chunk_context_product(105_241) == SAFE_PRODUCT
    assert adaptive_prefill_budget(105_241, 24_576) == 8173

    # A malformed value must not silently disable the sub-blocking (that would shrink chunks).
    monkeypatch.setenv(SUB_BLOCK_ENV, "not-a-number")
    assert chunk_context_product(105_241) == BIG_CHUNK_PRODUCT


def test_mask_budget_is_the_size_a_proven_configuration_produced():
    assert MASK_BUDGET_BYTES == BIG_CHUNK_TOKENS * SAFE_CONTEXT_TOKENS // 4
    assert BIG_CHUNK_PRODUCT == BIG_CHUNK_TOKENS * SAFE_CONTEXT_TOKENS
    assert BIG_CHUNK_PRODUCT == 3 * SAFE_PRODUCT


def test_mask_budget_env_override(monkeypatch):
    """``FREETOKEN_PREFILL_MASK_BUDGET_MB`` rescales the envelope at call time.

    The SWA window pool, the ``--prefill-chunk-tokens`` ceiling and this mask budget are three
    caps that must move together (notes/freetoken/20261003-dsv41-swa-pool-envelope.md): at a
    105,241-token context the compiled-in 615 MiB budget caps a pass at 24,519 tokens, while
    950 MiB lifts that cap to 37,861 -- above the 35,200 budget a 573-page pool provides.
    """
    assert mask_budget_bytes() == MASK_BUDGET_BYTES
    assert chunk_context_product(105_241) == BIG_CHUNK_PRODUCT

    monkeypatch.setenv(MASK_BUDGET_ENV, "950")
    assert mask_budget_bytes() == 950 * 1_048_576
    assert chunk_context_product(105_241) == 950 * 1_048_576 * 4
    # ... and it flows into the pass budget: the mask cap clears the 573-page pool's 35,200.
    assert chunk_context_product(105_241) // 105_241 == 37_861
    assert adaptive_prefill_budget(105_241, 40_960) == 37_861

    # Zero, negative and malformed values fall back to the compiled-in budget.
    for bad in ("0", "-5", "not-a-number"):
        monkeypatch.setenv(MASK_BUDGET_ENV, bad)
        assert mask_budget_bytes() == MASK_BUDGET_BYTES, bad
        assert chunk_context_product(105_241) == BIG_CHUNK_PRODUCT, bad


def test_budget_crosses_over_exactly_at_the_envelope():
    """At chunk x context == the envelope the ceiling is still allowed; one token more is not."""
    for product in (SAFE_PRODUCT, BIG_CHUNK_PRODUCT):
        for ceiling in (4_096, 8_192, 24_576, 65_536):  # > MIN_CHUNK_TOKENS: the floor cannot bite
            exact = product // ceiling
            assert adaptive_prefill_budget(exact, ceiling, safe_product=product) == ceiling
            assert adaptive_prefill_budget(exact + 1, ceiling, safe_product=product) < ceiling


def test_ceiling_is_never_exceeded():
    for ceiling in (0, 1, 127, 2_048, 8_192, 24_576, 250_000, 1_048_576):
        for context in (0, 1, 1_000, 23_671, SAFE_CONTEXT_TOKENS, 500_000, 2_000_000):
            budget = adaptive_prefill_budget(context, ceiling)
            assert 0 <= budget <= ceiling, (context, ceiling, budget)


def test_transient_stays_inside_the_envelope_unless_the_floor_bites():
    """Whenever the floor does not bite, the invariant ``budget x context <= product`` holds.

    Otherwise the budget is exactly the floor -- which only happens for contexts past ~420k tokens
    (conservative envelope) or ~1.26M tokens (sub-blocked envelope), where the window pool, not the
    indexer, is the binding limit.
    """
    for product in (SAFE_PRODUCT, BIG_CHUNK_PRODUCT):
        for ceiling in (2_048, 4_096, 8_192, 24_576, 1_048_576):
            for context in (1, 128, 1_000, 23_671, 60_000, SAFE_CONTEXT_TOKENS, 300_000):
                budget = adaptive_prefill_budget(context, ceiling, safe_product=product)
                assert budget * context <= product or budget == MIN_CHUNK_TOKENS, (
                    context,
                    ceiling,
                    budget,
                )


def test_the_default_path_satisfies_the_big_envelope():
    for ceiling in (2_048, 8_192, 24_576, 65_536):
        for context in (1, 128, 23_671, 105_241, 500_000):
            budget = adaptive_prefill_budget(context, ceiling)
            assert budget * context <= BIG_CHUNK_PRODUCT or budget == MIN_CHUNK_TOKENS, (
                context,
                ceiling,
                budget,
            )


def test_floor_stops_the_degenerate_shrink():
    """Above ``product // MIN_CHUNK_TOKENS`` the ratio would fall below the floor."""
    huge = SAFE_PRODUCT // MIN_CHUNK_TOKENS  # ~420k tokens
    assert adaptive_prefill_budget(huge, 24_576, safe_product=SAFE_PRODUCT) == MIN_CHUNK_TOKENS
    assert adaptive_prefill_budget(2_000_000, 24_576, safe_product=SAFE_PRODUCT) == MIN_CHUNK_TOKENS
    # The sub-blocked envelope holds out an order of magnitude further before the floor bites.
    assert adaptive_prefill_budget(huge, 24_576) > MIN_CHUNK_TOKENS
    assert adaptive_prefill_budget(2_000_000, 24_576) == MIN_CHUNK_TOKENS
    # and the floor never inflates past the ceiling either
    assert adaptive_prefill_budget(2_000_000, PAGE) == PAGE


def test_zero_and_degenerate_inputs():
    assert adaptive_prefill_budget(23_671, 0) == 0
    assert adaptive_prefill_budget(23_671, -5) == 0
    # a non-positive context is treated as one token rather than raising
    assert adaptive_prefill_budget(0, 24_576) == 24_576
    assert adaptive_prefill_budget(-1, 24_576) == 24_576


def test_result_never_exceeds_a_small_ceiling_even_under_a_tiny_context():
    assert adaptive_prefill_budget(1, MIN_CHUNK_TOKENS - 1) == MIN_CHUNK_TOKENS - 1


def test_monotone_in_context_and_ceiling():
    """More context can only shrink the budget; a larger ceiling can only grow it."""
    contexts = [1, 500, 1_000, 23_671, 50_000, SAFE_CONTEXT_TOKENS, 200_000, 900_000]
    budgets = [adaptive_prefill_budget(c, 24_576) for c in contexts]
    assert budgets == sorted(budgets, reverse=True)
    ceilings = [2_048, 4_096, 8_192, 24_576, 49_152]
    grown = [adaptive_prefill_budget(23_671, c) for c in ceilings]
    assert grown == sorted(grown)


# --------------------------------------------------------------------------------------
# config-layer resolution
# --------------------------------------------------------------------------------------


def _config(*, max_seq_len: int = 1048576, max_extend_tokens: int = 8192, **extra):
    """Mirrors tests/engine/test_prefill_chunk_tokens.py: a SimpleNamespace is enough."""
    model_config = SimpleNamespace(
        dsv4_args=SimpleNamespace(max_seq_len=0, max_batch_size=0, window_size=PAGE),
        dsv41_args=SimpleNamespace(
            max_seq_len=0, max_batch_size=0, moe_ep_size=1, window_size=PAGE
        ),
    )
    fields = dict(
        model_config=model_config,
        max_seq_len=max_seq_len,
        max_running_req=4,
        cache_type="naive",
        moe_ep_size=1,
        max_extend_tokens=max_extend_tokens,
        cuda_graph_max_bs=None,
        cuda_graph_bs=None,
        prefill_chunk_tokens=0,
        # mirrors the SchedulerConfig default; overridable through ``extra``
        prefill_chunk_adaptive=False,
    )
    fields.update(extra)
    return SimpleNamespace(**fields)


def _resolve(fn, config):
    calls: list[tuple[str, object]] = []

    def override(attr, value):
        calls.append((attr, value))
        setattr(config, attr, value)

    fn(config, override)
    return config, dict(calls)


def _dsv41():
    from freetoken.engine.engine import _adjust_dsv41_config

    return _adjust_dsv41_config


def _dsv4():
    from freetoken.engine.engine import _adjust_dsv4_config

    return _adjust_dsv4_config


def test_adaptive_cap_still_becomes_the_ceiling():
    """The flag must not change what sizes buffers/warmup/pynccl: the ceiling is the flag value."""
    config, calls = _resolve(
        _dsv41(), _config(prefill_chunk_tokens=24576, prefill_chunk_adaptive=True)
    )
    assert config.max_extend_tokens == 24576
    assert calls["max_extend_tokens"] == 24576


def test_adaptive_ceiling_is_page_rounded_and_clamped_like_the_static_cap():
    config, _ = _resolve(
        _dsv41(), _config(prefill_chunk_tokens=2500, prefill_chunk_adaptive=True)
    )
    assert config.max_extend_tokens == 2432  # 2500 // 128 * 128


def test_adaptive_without_a_ceiling_does_not_raise_max_extend_tokens():
    """The historical raise to ``max_seq_len`` exists so the pool chunks a long prompt; with the
    adaptive rule the pool is bounded by the ceiling instead, and a 1M-token
    ``max_extend_tokens`` would allocate pynccl scratch and warmup lengths for a pass no prompt can
    afford. So this branch keeps the configured value -- and the pool still bounds it."""
    config, calls = _resolve(_dsv41(), _config(prefill_chunk_adaptive=True))
    assert config.max_extend_tokens == 8192
    assert "max_extend_tokens" not in calls
    assert config.max_seq_len == 1048576


def test_without_the_flag_nothing_changes():
    """Regression guard: the adaptive field must be inert when the flag is off."""
    config, calls = _resolve(_dsv41(), _config())
    assert config.max_extend_tokens == config.max_seq_len == 1048576
    assert calls["max_extend_tokens"] == 1048576

    config, _ = _resolve(_dsv41(), _config(prefill_chunk_tokens=2048))
    assert config.max_extend_tokens == 2048


def test_flag_may_be_absent_from_an_older_config_object():
    """``getattr(..., False)`` keeps the resolution working for configs built before the field."""
    config = _config(prefill_chunk_tokens=8192)
    del config.prefill_chunk_adaptive
    config, _ = _resolve(_dsv41(), config)
    assert config.max_extend_tokens == 8192


def test_dsv4_resolves_the_adaptive_ceiling_too():
    config, calls = _resolve(
        _dsv4(), _config(prefill_chunk_tokens=24576, prefill_chunk_adaptive=True)
    )
    assert config.max_extend_tokens == 24576
    assert calls["page_size"] == PAGE


# --------------------------------------------------------------------------------------
# scheduler wiring
# --------------------------------------------------------------------------------------


def _manager(monkeypatch, adaptive: bool):
    from freetoken.scheduler import prefill as prefill_mod

    seen: dict = {}

    class RecordingAdder:
        def __init__(self, **kwargs):
            seen.update(kwargs)

        def try_add_one(self, pending_req):  # breaks the admission loop immediately
            return None

    monkeypatch.setattr(prefill_mod, "PrefillAdder", RecordingAdder)
    manager = prefill_mod.PrefillManager(
        cache_manager=None,
        table_manager=None,
        decode_manager=SimpleNamespace(inflight_tokens=7),
        adaptive_chunk=adaptive,
    )
    return manager, seen


def _pending(input_len: int):
    """The fields ``schedule_next_batch`` touches before (and after) admission."""
    return SimpleNamespace(input_len=input_len, chunked_req=None, mm_items=None)


def test_scheduler_hands_a_short_prompt_the_whole_ceiling(monkeypatch):
    manager, seen = _manager(monkeypatch, adaptive=True)
    manager.pending_list = [_pending(23_671)]
    assert manager.schedule_next_batch(24_576) is None  # nothing admitted -> no batch
    assert seen["token_budget"] == 24_576
    assert seen["reserved_size"] == 7


def test_scheduler_cuts_a_long_prompt_back_inside_the_envelope(monkeypatch):
    """Default (sub-blocked) envelope: a 105,241-token prompt gets ~24.5k tokens per pass."""
    monkeypatch.delenv(SUB_BLOCK_ENV, raising=False)
    manager, seen = _manager(monkeypatch, adaptive=True)
    manager.pending_list = [_pending(105_241)]
    assert manager.schedule_next_batch(24_576) is None
    assert seen["token_budget"] == BIG_CHUNK_PRODUCT // 105_241


def test_scheduler_falls_back_to_the_measured_chunk_without_sub_blocking(monkeypatch):
    monkeypatch.setenv(SUB_BLOCK_ENV, "0")
    manager, seen = _manager(monkeypatch, adaptive=True)
    manager.pending_list = [_pending(105_241)]
    assert manager.schedule_next_batch(24_576) is None
    assert seen["token_budget"] == 8173


def test_scheduler_uses_the_longest_pending_prompt(monkeypatch):
    monkeypatch.delenv(SUB_BLOCK_ENV, raising=False)
    manager, seen = _manager(monkeypatch, adaptive=True)
    manager.pending_list = [_pending(500), _pending(105_241)]
    manager.schedule_next_batch(24_576)
    assert seen["token_budget"] == BIG_CHUNK_PRODUCT // 105_241


def test_scheduler_is_byte_identical_when_the_flag_is_off(monkeypatch):
    monkeypatch.delenv(SUB_BLOCK_ENV, raising=False)
    manager, seen = _manager(monkeypatch, adaptive=False)
    manager.pending_list = [_pending(105_241)]
    manager.schedule_next_batch(24_576)
    assert seen["token_budget"] == 24_576


def test_scheduler_returns_none_for_an_empty_queue(monkeypatch):
    manager, seen = _manager(monkeypatch, adaptive=True)
    assert manager.schedule_next_batch(24_576) is None
    assert seen == {}


# --------------------------------------------------------------------------------------
# contention chunk cap (FREETOKEN_LONG_PREFILL_WHEN_WAITING, dsv41 port)
# --------------------------------------------------------------------------------------


def test_contention_cap_env_is_off_by_default_and_parses(monkeypatch):
    monkeypatch.delenv(CONTENTION_CAP_ENV, raising=False)
    assert contention_chunk_cap_tokens() == 0
    monkeypatch.setenv(CONTENTION_CAP_ENV, "7168")
    assert contention_chunk_cap_tokens() == 7168
    for raw in ("0", "-5", "junk"):
        monkeypatch.setenv(CONTENTION_CAP_ENV, raw)
        assert contention_chunk_cap_tokens() == 0, raw


def test_contention_capped_budget_requires_competition():
    assert contention_capped_budget(24_576, 2, 7168) == 7168
    assert contention_capped_budget(24_576, 1, 7168) == 24_576, (
        "a lone prompt keeps the big chunk -- the cap buys interactivity, not memory"
    )
    assert contention_capped_budget(4096, 5, 7168) == 4096, "the cap never raises a budget"
    assert contention_capped_budget(24_576, 5, 0) == 24_576, "cap 0 is off"


def test_scheduler_caps_the_chunk_only_while_requests_contend(monkeypatch):
    monkeypatch.setenv(CONTENTION_CAP_ENV, "7168")
    monkeypatch.delenv(SUB_BLOCK_ENV, raising=False)

    solo, seen = _manager(monkeypatch, adaptive=False)
    solo.pending_list = [_pending(105_241)]
    solo.schedule_next_batch(24_576)
    assert seen["token_budget"] == 24_576
    assert solo.contention_capped_passes == 0

    contended, seen = _manager(monkeypatch, adaptive=False)
    contended.pending_list = [_pending(105_241), _pending(2_000)]
    contended.schedule_next_batch(24_576)
    assert seen["token_budget"] == 7168
    assert contended.contention_capped_passes == 1


def test_scheduler_contention_cap_composes_with_the_adaptive_budget(monkeypatch):
    """The cap applies after the adaptive envelope, so the smaller of the two wins."""
    monkeypatch.setenv(CONTENTION_CAP_ENV, "7168")
    monkeypatch.setenv(SUB_BLOCK_ENV, "0")  # conservative envelope: 8173 at 105,241 tokens
    manager, seen = _manager(monkeypatch, adaptive=True)
    manager.pending_list = [_pending(105_241), _pending(2_000)]
    manager.schedule_next_batch(24_576)
    assert seen["token_budget"] == 7168, "cap 7168 < adaptive 8173"


def test_scheduler_contention_cap_off_by_default(monkeypatch):
    monkeypatch.delenv(CONTENTION_CAP_ENV, raising=False)
    monkeypatch.delenv(SUB_BLOCK_ENV, raising=False)
    manager, seen = _manager(monkeypatch, adaptive=False)
    manager.pending_list = [_pending(105_241), _pending(2_000)]
    manager.schedule_next_batch(24_576)
    assert seen["token_budget"] == 24_576
    assert manager.contention_capped_passes == 0
