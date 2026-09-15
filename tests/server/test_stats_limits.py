"""`/v1/stats` must report quantities the engine actually honors, not frontend-side defaults.

The frontend process never runs ``_adjust_config`` (that happens inside the scheduler process),
so ``config.page_size`` there can still be the CLI default while the engine pages at 64
(qsa_sparse). Reporting the default made ``total_pages`` read as a token count and hid the
enforced context ceiling from clients entirely.
"""
from __future__ import annotations

from types import SimpleNamespace

from freetoken.server.stats import build_stats


def _state(*, cache_pools=None, enforced=None, page_size=1, model_max=262144,
           served_modalities=(), mm=None, mm_stats=None):
    pools = dict(cache_pools or {})
    state = SimpleNamespace(
        config=SimpleNamespace(
            served_model_name="unit-model",
            model_path="/models/unit-model",
            page_size=page_size,
            max_seq_len=model_max,
            model_config=SimpleNamespace(
                has_linear_attention=True, has_swa_attention=False, is_moe=True,
                dsv4_args=None,
            ),
            served_modalities=frozenset(served_modalities),
            mm=mm,
        ),
        ready_at=None,
        instance_id="unit",
        gpus=[],
    )
    state.cache_pools = pools
    if enforced is not None:
        state.max_seq_len = enforced
    state.stats = SimpleNamespace(
        kv_used_pages=10, kv_total_pages=4096, mamba_used_slots=0, mamba_total_slots=0,
        swa_used_tokens=0, swa_total_tokens=0, vram_bytes=1 << 30, active=0, completed=0,
        prompt_tokens_total=0, completion_tokens_total=0, cached_tokens_total=0,
        moe_stats=None, mm_stats=mm_stats, decode_tps=lambda *_: 0.0, prefill_tps=lambda *_: 0.0,
    )
    return state


def test_stats_uses_the_engines_page_size_when_the_frontend_config_lags():
    """page_size comes from the engine's readiness meta, so pages x page_size is a token count.

    Regression: with the CLI default (1) the doc reported 4096 "tokens" for a pool the engine
    had actually sized at 4096 pages x 64 = 262144 tokens.
    """
    state = _state(cache_pools={"num_pages": 4096, "page_size": 64}, page_size=1)
    kv = build_stats(state, p95_ms=0, ttft_mean_ms=0)["kv"]

    assert kv["page_size"] == 64
    assert kv["total_pages"] * kv["page_size"] == 262144


def test_stats_falls_back_to_the_frontend_page_size_without_meta():
    """No readiness meta yet (older engine / mid-load): keep the old behaviour, never 0."""
    state = _state(cache_pools=None, page_size=1)
    assert build_stats(state, p95_ms=0, ttft_mean_ms=0)["kv"]["page_size"] == 1


def test_stats_publishes_the_enforced_context_ceiling_next_to_the_model_ceiling():
    """`limits.max_seq_len` is what the scheduler admits; the model's own ceiling is kept."""
    state = _state(enforced=245760, model_max=262144)
    limits = build_stats(state, p95_ms=0, ttft_mean_ms=0)["limits"]

    assert limits["max_seq_len"] == 245760
    assert limits["model_max_seq_len"] == 262144


def test_stats_limits_falls_back_to_the_model_ceiling_mid_load():
    """Before the meta lands the enforced value is unknown; report the ceiling, not 0."""
    state = _state(enforced=None, model_max=262144)
    limits = build_stats(state, p95_ms=0, ttft_mean_ms=0)["limits"]

    assert limits["max_seq_len"] == 262144


def test_stats_model_ctx_is_the_enforced_ceiling_not_the_checkpoint_ceiling():
    """``launch._stats_context_length`` reads ``model.ctx`` as its client-window fallback.

    Reporting the raw checkpoint ceiling there re-introduced exactly the bug ``limits``
    exists to prevent: a client sizes its window from the model card and then sends
    prompts the scheduler rejects. The raw ceiling stays in
    ``limits.model_max_seq_len``.
    """
    state = _state(enforced=245760, model_max=262144)
    doc = build_stats(state, p95_ms=0, ttft_mean_ms=0)

    assert doc["model"]["ctx"] == 245760
    assert doc["limits"]["max_seq_len"] == 245760
    assert doc["limits"]["model_max_seq_len"] == 262144


def test_stats_model_ctx_keeps_the_ceiling_while_the_meta_is_in_flight():
    """No readiness meta yet: ctx falls back to the ceiling rather than 0."""
    state = _state(enforced=None, model_max=262144)
    assert build_stats(state, p95_ms=0, ttft_mean_ms=0)["model"]["ctx"] == 262144


def test_stats_omits_the_multimodal_section_when_no_encoder_is_served():
    """A text-only process must not advertise an image budget it never honours."""
    assert build_stats(_state(), p95_ms=0, ttft_mean_ms=0)["mm"] is None


def test_stats_reports_the_image_budget_and_the_live_encoder_cache():
    """``/v1/stats`` is the only place the encoder embedding cache is observable.

    The image token budget matters because it is what bounds one client image (this
    deployment clamps it to 4096 tokens); None means "the checkpoint processor's own
    default", so it must be distinguishable from a configured 0.
    """
    mm_config = SimpleNamespace(image_min_tokens=None, image_max_tokens=4096, embed_cache_device="cpu")
    state = _state(served_modalities=("image",), mm=mm_config, mm_stats={"entries": 3, "bytes": 2048})
    mm = build_stats(state, p95_ms=0, ttft_mean_ms=0)["mm"]

    assert mm["served_modalities"] == ["image"]
    assert mm["image_tokens"] == {"min": None, "max": 4096}
    assert mm["embed_cache_device"] == "cpu"
    assert mm["encoder_cache"] == {"entries": 3, "bytes": 2048}


def test_stats_multimodal_section_survives_before_the_first_sample():
    """No sample yet: the section still carries the configured budget, cache is None."""
    mm_config = SimpleNamespace(image_min_tokens=None, image_max_tokens=4096, embed_cache_device="cpu")
    mm = build_stats(_state(served_modalities=("image",), mm=mm_config), p95_ms=0, ttft_mean_ms=0)["mm"]

    assert mm["encoder_cache"] is None
    assert mm["image_tokens"]["max"] == 4096
