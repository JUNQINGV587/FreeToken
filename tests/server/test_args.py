"""--linear-state-cache-ratio: parses onto ServerArgs and sizes the hybrid GDN pool.

ServerArgs inherits SchedulerConfig(EngineConfig), so the field already exists on the engine
side (engine/config.py, default 2.0); the flag only needs the parser entry.
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
