"""``--prefill-chunk-tokens``: the DSV4/DSV4.1 prefill chunk must not scale with the KV pool.

The pool derives ``prefill_chunk_budget`` from the window pool (half of it, roughly), and the
window pool scales with the token reservation. ``--max-prefill-length`` cannot express a cap for
these models -- config resolution deliberately raises ``max_extend_tokens`` to ``max_seq_len`` so
the pool's budget is what chunks a prompt -- so a large reservation (a 1M-token pool resolves to a
~100k-token chunk) lets one chunk allocate the indexer's O(chunk x context) causal-mask transient
and OOMs the engine (dsv41_indexer.indexer_select_prefill).

These tests pin the resolution at the config layer, where it is decided: unset keeps the historical
pool-budget behaviour, set caps the chunk on a whole-window-page boundary.
"""

from __future__ import annotations

from types import SimpleNamespace

PAGE = 128  # the DSV4/DSV4.1 window page, i.e. the pool's currency


def _config(*, max_seq_len: int = 1048576, max_extend_tokens: int = 8192, **extra):
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
        # "naive" keeps the cache_type branch inert for both models.
        cache_type="naive",
        moe_ep_size=1,
        max_extend_tokens=max_extend_tokens,
        cuda_graph_max_bs=None,
        cuda_graph_bs=None,
        # mirrors the SchedulerConfig default; overridable through ``extra``
        prefill_chunk_tokens=0,
    )
    fields.update(extra)
    return SimpleNamespace(**fields)


def _resolve(fn, config):
    """Run one of the adjust functions with a real recording/applying override."""
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


def test_unset_keeps_the_pool_budget_behaviour():
    """No flag: max_extend_tokens is raised to max_seq_len and the pool budget chunks the prompt."""
    config, calls = _resolve(_dsv41(), _config())
    assert config.max_extend_tokens == config.max_seq_len == 1048576
    assert calls["max_extend_tokens"] == 1048576
    assert config.prefill_chunk_tokens == 0


def test_cap_is_applied_and_rounded_down_to_window_pages():
    config, calls = _resolve(_dsv41(), _config(prefill_chunk_tokens=2500))
    # 2500 // 128 * 128 == 2432: never round a chunk UP past what the operator asked for.
    assert config.max_extend_tokens == 2432
    assert calls["max_extend_tokens"] == 2432


def test_cap_below_one_page_floors_at_one_page():
    config, _ = _resolve(_dsv41(), _config(prefill_chunk_tokens=64))
    assert config.max_extend_tokens == PAGE


def test_cap_above_max_seq_len_is_clamped_to_the_context():
    config, _ = _resolve(_dsv41(), _config(prefill_chunk_tokens=8_000_000))
    assert config.max_extend_tokens == config.max_seq_len == 1048576


def test_cap_lowers_a_larger_default():
    """The flag must win over the 8192 default (and over the max_seq_len raise) -- that is the
    whole point: a 1M-token pool's own budget is ~100k tokens, which no prompt can afford."""
    config, _ = _resolve(_dsv41(), _config(prefill_chunk_tokens=2048))
    assert config.max_extend_tokens == 2048


def test_dsv4_takes_the_same_cap():
    config, calls = _resolve(_dsv4(), _config(prefill_chunk_tokens=2048))
    assert config.max_extend_tokens == 2048
    # page size for the DSV4 family is the window page, set before the cap is resolved
    assert calls["page_size"] == PAGE


def test_dsv4_unset_still_single_chunks():
    config, _ = _resolve(_dsv4(), _config())
    assert config.max_extend_tokens == config.max_seq_len


def test_cap_is_applied_with_prefix_reuse_enabled():
    """Reuse and the chunk cap are independent: resolving V4.1 onto the shared SWARadixCache must
    not disturb the cap, or a large pool hands the indexer a ~100k-token chunk again."""
    config, calls = _resolve(_dsv41(), _config(cache_type="radix", prefill_chunk_tokens=2048))
    assert config.cache_type == "swa_radix"
    assert config.max_extend_tokens == 2048
    assert calls["max_extend_tokens"] == 2048


def test_page_size_comes_from_the_model_not_the_cap():
    """A different window size must move the rounding unit with it."""
    config = _config(prefill_chunk_tokens=1000)
    config.model_config.dsv41_args.window_size = 64
    config, _ = _resolve(_dsv41(), config)
    assert config.max_extend_tokens == 960  # 1000 // 64 * 64
