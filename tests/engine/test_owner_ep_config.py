"""``_validate_owner_ep_config`` must reject what the owner runtime cannot honour.

Two settings used to be accepted and then silently ignored:

* ``--moe-cpu-layers`` -- ``_decode_owner`` is selected before the ``is_cpu_layer``
  branch, and ``OwnerOffloadMoeCache`` hard-codes a GPU inner cache, so the flag
  had no effect at all.
* FTW checkpoints -- ``load_ftw_banks`` rebuilds ``[num_experts, ...]`` GLOBAL
  expert rows with no ownership filter, so the banks cannot bind to the
  owner-local geometry.

Fail-fast is the contract for owner EP (``moe_ep_size > 1`` is opt-in and
validated before any allocation), so both must raise rather than serve a
different configuration than the one requested.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from freetoken.engine.engine import _validate_owner_ep_config


def _config(**over):
    base = dict(
        moe_ep_size=2,
        tp_info=SimpleNamespace(size=2, rank=0),
        moe_strategy="offload",
        moe_cache_rate=None,
        moe_cache_size=4096,
        moe_cache_auto=False,
        moe_cpu_layers=None,
        model_path="/models/unit-model",
    )
    base.update(over)
    return SimpleNamespace(**base)


@pytest.fixture(autouse=True)
def _not_ftw(monkeypatch):
    monkeypatch.setattr("freetoken.checkpoint.ftw.is_ftw_checkpoint", lambda path: False)


def test_a_valid_owner_topology_passes():
    _validate_owner_ep_config(_config())


def test_moe_cache_auto_satisfies_the_cache_requirement():
    _validate_owner_ep_config(_config(moe_cache_size=0, moe_cache_auto=True))


def test_ep_size_one_is_a_no_op(monkeypatch):
    monkeypatch.setattr(
        "freetoken.checkpoint.ftw.is_ftw_checkpoint",
        lambda path: pytest.fail("must not probe the checkpoint when EP is off"),
    )
    _validate_owner_ep_config(_config(moe_ep_size=1))


@pytest.mark.parametrize("layers", ["0", "0,1", "0-3"])
def test_cpu_layers_are_rejected_instead_of_silently_ignored(layers):
    with pytest.raises(ValueError, match="moe-cpu-layers"):
        _validate_owner_ep_config(_config(moe_cpu_layers=layers))


def test_ftw_checkpoints_are_rejected(monkeypatch):
    monkeypatch.setattr("freetoken.checkpoint.ftw.is_ftw_checkpoint", lambda path: True)
    with pytest.raises(ValueError, match="FTW"):
        _validate_owner_ep_config(_config())


def test_an_explicit_cache_size_is_still_required_without_auto():
    with pytest.raises(ValueError, match="moe-cache-size"):
        _validate_owner_ep_config(_config(moe_cache_size=0))


def test_moe_cache_rate_is_still_rejected():
    with pytest.raises(ValueError, match="moe-cache-rate"):
        _validate_owner_ep_config(_config(moe_cache_rate=0.5))


def test_hybrid_strategy_is_accepted():
    _validate_owner_ep_config(_config(moe_strategy="hybrid"))


@pytest.mark.parametrize("strategy", ["resident", "fused", "cpu"])
def test_other_strategies_are_still_rejected(strategy):
    with pytest.raises(ValueError, match="offload or hybrid"):
        _validate_owner_ep_config(_config(moe_strategy=strategy))


def test_a_topology_other_than_tp2_ep2_is_still_rejected():
    with pytest.raises(ValueError, match="TP2"):
        _validate_owner_ep_config(_config(tp_info=SimpleNamespace(size=4, rank=0)))


# --------------------------------------------------------------------------------------
# The model half of the gate. _validate_owner_ep_config checks the requested topology; the
# model has to declare that its MoE seam was adapted to it, and the banks have to be the
# ownership-filtering NVFP4 ones. The gate used to be a qwen4_exp model_type whitelist, which
# made a model that HAD been adapted (V4.1, whose MoE seam already asks
# owner_ep_expert_tp_size) unreachable without editing the engine.
# --------------------------------------------------------------------------------------


def _model_config(**over):
    base = dict(model_type="qwen4_exp", expert_quant="nvfp4", owner_ep=True)
    base.update(over)
    return SimpleNamespace(**base)


def _check(**over):
    from freetoken.engine.engine import _check_owner_ep_model

    _check_owner_ep_model(SimpleNamespace(model_config=_model_config(**over)))


def test_a_model_that_declares_owner_ep_passes():
    _check()


def test_v41_is_accepted_once_the_model_declares_it():
    _check(model_type="deepseek_v41")


def test_a_model_without_the_declaration_is_rejected_by_name():
    with pytest.raises(NotImplementedError, match="deepseek_v4'"):
        _check(model_type="deepseek_v4", owner_ep=False)


def test_the_rejection_says_what_the_model_has_to_declare():
    with pytest.raises(NotImplementedError, match="ModelConfig.owner_ep"):
        _check(model_type="glm5_next", owner_ep=False)


def test_a_format_the_ownership_filter_cannot_load_is_rejected():
    """Only the NVFP4 provider filters expert rows by ownership and renumbers them locally."""
    with pytest.raises(NotImplementedError, match="NVFP4"):
        _check(expert_quant="mxfp4")


def test_an_older_model_config_without_the_field_is_still_a_clean_refusal():
    """ModelConfig gained the field, but tests/tools hand-roll SimpleNamespaces."""
    config = SimpleNamespace(model_config=SimpleNamespace(model_type="qwen4_exp", expert_quant="nvfp4"))
    from freetoken.engine.engine import _check_owner_ep_model

    with pytest.raises(NotImplementedError, match="owner_ep"):
        _check_owner_ep_model(config)
