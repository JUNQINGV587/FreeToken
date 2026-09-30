"""Config-surface flags that only need a parser entry, onto ServerArgs.

--linear-state-cache-ratio: parses onto ServerArgs and sizes the hybrid GDN pool.
--swa-num-pages-override: parses onto ServerArgs and pins the DSV4/DSV4.1 window tier.

ServerArgs inherits SchedulerConfig(EngineConfig), so both fields already exist on the engine
side (engine/config.py: linear_state_cache_ratio default 2.0, swa_num_pages_override default
None); the flags only need the parser entry, because parse_args splats the argparse namespace
straight into ServerArgs(**kwargs).
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from freetoken.engine.config import EngineConfig
from freetoken.kvcache.linear_state_pool import _linear_pool_num_slots
from freetoken.server.args import parse_args

ANON_PATH = "/models/anon"


class _Config:
    def to_dict(self):
        return {"architectures": ["Qwen3MoeForCausalLM"], "torch_dtype": "bfloat16"}


def _parse(extra):
    with patch("freetoken.utils.cached_load_hf_config", lambda _p: _Config()):
        args, _ = parse_args(["--model", ANON_PATH, *extra])
    return args


def test_linear_state_cache_ratio_default_is_engine_default():
    args = _parse([])
    assert args.linear_state_cache_ratio == 2.0
    assert EngineConfig.linear_state_cache_ratio == 2.0


def test_linear_state_cache_ratio_parses_float():
    assert _parse(["--linear-state-cache-ratio", "8"]).linear_state_cache_ratio == 8.0
    assert _parse(["--linear-state-cache-ratio", "0.5"]).linear_state_cache_ratio == 0.5


def test_linear_state_cache_ratio_sizes_the_pool():
    """mr=1: ratio 2.0 -> 9 slots (4 evictable), ratio 8 -> 13 slots (8 evictable)."""
    for ratio, slots in ((2.0, 9), (8.0, 13)):
        c = SimpleNamespace(max_running_req=1, cache_type="hybrid_radix",
                            linear_state_cache_ratio=ratio)
        assert _linear_pool_num_slots(c) == slots, (ratio, _linear_pool_num_slots(c))


def test_linear_state_cache_ratio_fractional_ceil():
    """2.5 * 3 -> extra = max(4, ceil(7.5)) = 8 (int() would truncate to 7):
    pool = 4*3 + 8 + 1 = 21."""
    c = SimpleNamespace(max_running_req=3, cache_type="hybrid_radix",
                        linear_state_cache_ratio=2.5)
    assert _linear_pool_num_slots(c) == 21


def test_linear_state_cache_ratio_rejects_non_positive():
    """<= 0 fails fast at engine-config adjustment (mirrors swa_full_tokens_ratio)
    instead of silently clamping to the 4-slot cache floor."""
    from freetoken.engine.engine import _adjust_config

    for bad in (0.0, -1.0):
        config = SimpleNamespace(
            model_config=SimpleNamespace(
                single_stream_only=False, dsv4_args=None, has_swa_attention=False,
                has_linear_attention=True, is_moe=True,
            ),
            max_running_req=4, cuda_graph_max_bs=None, linear_state_cache_ratio=bad,
        )
        with pytest.raises(ValueError, match="linear_state_cache_ratio must be > 0"):
            _adjust_config(config)


def test_swa_num_pages_override_default_is_engine_default():
    args = _parse([])
    assert args.swa_num_pages_override is None
    assert EngineConfig.swa_num_pages_override is None


def test_swa_num_pages_override_parses_int():
    """The pinned page count reaches the engine config as-is; _dsv41_pool_sizes floors it."""
    assert _parse(["--swa-num-pages-override", "160"]).swa_num_pages_override == 160


def test_swa_num_pages_override_rejects_non_positive():
    """A 0/negative pin fails fast at the parser instead of silently meaning 'the floor'."""
    with pytest.raises(SystemExit):
        _parse(["--swa-num-pages-override", "0"])
    with pytest.raises(SystemExit):
        _parse(["--swa-num-pages-override", "-4"])

