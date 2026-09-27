"""SmoothQuant-style attention alpha (k_alpha / v_alpha) consumer tests.

(a) CPU, pure functions: the header scan detects the tensors, ``_rename`` passes the
    keys through to the model's state-dict names, and the TP shard rule slices the 1-D
    [num_kv*head_dim] alpha exactly like the k/v_proj weight head rows;
(b) GPU, module level: buffers are registered iff the config flags say so, alpha ==
    ones is bitwise the no-alpha layer, and each apply point
    (pre_norm / post_norm / post_rope) lands the scaling where the capturing backend
    sees it. The capture doubles as proof that the KV-cache write path receives the
    already-scaled k/v.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

from .common import Fixture, parsed_config, requires_cuda

QSA_LAYER = 3


# ---------------------------------------------------------------- CPU: pure functions


def test_checkpoint_scan_detects_alpha_keys(tmp_path):
    from safetensors.torch import save_file

    from freetoken.models.qwen4_exp.weight import checkpoint_smooth_alpha

    save_file(
        {
            "model.language_model.layers.0.self_attn.k_alpha": torch.ones(8),
            "model.language_model.layers.0.self_attn.v_alpha": torch.ones(8),
        },
        str(tmp_path / "model.safetensors"),
    )
    assert checkpoint_smooth_alpha(str(tmp_path)) == (True, True)

    plain = tmp_path / "plain"
    plain.mkdir()
    save_file(
        {"model.language_model.layers.0.self_attn.q_proj.weight": torch.ones(4, 4)},
        str(plain / "model.safetensors"),
    )
    assert checkpoint_smooth_alpha(str(plain)) == (False, False)

    k_only = tmp_path / "k_only"
    k_only.mkdir()
    save_file(
        {"model.language_model.layers.0.self_attn.k_alpha": torch.ones(8)},
        str(k_only / "model.safetensors"),
    )
    assert checkpoint_smooth_alpha(str(k_only)) == (True, False)


def test_rename_passes_alpha_through():
    from freetoken.models.qwen4_exp.weight import _rename

    assert (
        _rename("model.language_model.layers.5.self_attn.k_alpha")
        == "model.layers.5.self_attn.k_alpha"
    )
    assert (
        _rename("model.language_model.layers.5.self_attn.v_alpha")
        == "model.layers.5.self_attn.v_alpha"
    )


def test_shard_alpha_matches_kv_weight_rows():
    from types import SimpleNamespace

    from freetoken.models.qwen4_exp.weight import shard_qwen4_exp_dense_tensor

    config = SimpleNamespace(
        num_qo_heads=4,
        num_kv_heads=4,
        head_dim=8,
        linear_attention_group=lambda: None,
    )
    alpha = torch.arange(4 * 8, dtype=torch.float32)
    key = "model.layers.0.self_attn.k_alpha"
    pieces = [
        shard_qwen4_exp_dense_tensor(key, alpha, config=config, rank=r, world_size=2)
        for r in (0, 1)
    ]
    assert torch.equal(torch.cat(pieces), alpha)
    assert [p.shape for p in pieces] == [torch.Size([16]), torch.Size([16])]
    # TP1 identity
    assert torch.equal(
        shard_qwen4_exp_dense_tensor(key, alpha, config=config, rank=0, world_size=1),
        alpha,
    )


# ------------------------------------------------------------- GPU: module behaviour


def _alpha_config(apply: str):
    config = parsed_config()
    return dataclasses.replace(
        config, smooth_alpha_k=True, smooth_alpha_v=True, smooth_alpha_apply=apply
    )


class _CaptureBackend:
    """Records the q/k/v handed to qsa_forward (i.e. what the KV-cache write sees)."""

    def __init__(self, out_features: int):
        self.out_features = out_features
        self.captured: dict[str, torch.Tensor] = {}

    def qsa_forward(self, q, k, v, index, layer_id, batch):
        self.captured = {
            "q": q.detach().clone(),
            "k": k.detach().clone(),
            "v": v.detach().clone(),
        }
        return torch.zeros(q.shape[0], self.out_features, device=q.device, dtype=q.dtype)


def _run_with_capture(fixture: Fixture, layer, x: torch.Tensor) -> dict[str, torch.Tensor]:
    import freetoken.core as core
    from freetoken.core import set_global_ctx

    # a later Fixture's fresh_ctx replaced the global one; reset like fresh_ctx does,
    # then restore this fixture's
    core._GLOBAL_CTX = None  # test-only
    set_global_ctx(fixture.ctx)
    capture = _CaptureBackend(layer.qo_attn_dim)
    fixture.ctx.attn_backend = capture
    reqs = [fixture.req(0, 0, x.shape[0])]
    with torch.no_grad():
        layer.forward(x, fixture.batch(reqs, "prefill"))
    return capture.captured


@requires_cuda
def test_buffers_registered_iff_config_flags():
    fixture = Fixture(parsed_config(), num_pages=8)
    layer = fixture.layer(QSA_LAYER)
    assert "k_alpha" not in layer.state_dict()
    assert layer.k_alpha is None and layer.v_alpha is None

    fixture2 = Fixture(_alpha_config("pre_norm"), num_pages=8)
    layer2 = fixture2.layer(QSA_LAYER)
    state = layer2.state_dict()
    kv_dim = layer2.num_kv * layer2.head_dim
    assert state["k_alpha"].shape == (kv_dim,)
    assert state["v_alpha"].shape == (kv_dim,)
    assert state["k_alpha"].dtype == fixture2.dtype


@requires_cuda
def test_alpha_ones_is_identity():
    """alpha == ones must be bitwise the no-alpha layer (capture-level, no pool noise)."""
    config = _alpha_config("post_norm")
    fixture = Fixture(config, num_pages=8)
    layer = fixture.layer(QSA_LAYER, seed=5)
    layer.k_alpha.fill_(1.0)
    layer.v_alpha.fill_(1.0)

    plain_fixture = Fixture(parsed_config(), num_pages=8)
    plain = plain_fixture.layer(QSA_LAYER, seed=5)
    # fill_weights randomizes every floating state-dict entry, and the alpha buffers
    # shift the random sequence; align the shared weights explicitly instead.
    shared = {k: v for k, v in layer.state_dict().items() if not k.endswith("_alpha")}
    plain.load_state_dict(shared)  # shared holds exactly the plain layer's keys

    gen = torch.Generator(device=fixture.device).manual_seed(3)
    x = torch.randn(
        40, config.hidden_size, device=fixture.device, dtype=fixture.dtype, generator=gen
    )
    got = _run_with_capture(fixture, layer, x)
    want = _run_with_capture(plain_fixture, plain, x)
    for name in ("q", "k", "v"):
        torch.testing.assert_close(got[name], want[name], rtol=0, atol=0)


@requires_cuda
@pytest.mark.parametrize("apply", ["pre_norm", "post_norm", "post_rope"])
def test_k_alpha_lands_at_the_configured_point(apply):
    config = _alpha_config(apply)
    fixture = Fixture(config, num_pages=8)
    layer = fixture.layer(QSA_LAYER, seed=7)
    kv_dim = layer.num_kv * layer.head_dim
    gen = torch.Generator(device=fixture.device).manual_seed(9)
    k_alpha = (
        torch.rand(kv_dim, device=fixture.device, dtype=fixture.dtype, generator=gen) + 0.5
    )
    v_alpha = (
        torch.rand(kv_dim, device=fixture.device, dtype=fixture.dtype, generator=gen) + 0.5
    )
    layer.k_alpha.copy_(k_alpha)
    layer.v_alpha.copy_(v_alpha)

    capture = _CaptureBackend(layer.qo_attn_dim)
    fixture.ctx.attn_backend = capture
    reqs = [fixture.req(0, 0, 24)]
    gen_x = torch.Generator(device=fixture.device).manual_seed(4)
    x = torch.randn(
        24, config.hidden_size, device=fixture.device, dtype=fixture.dtype, generator=gen_x
    )
    batch = fixture.batch(reqs, "prefill")
    with torch.no_grad():
        layer.forward(x, batch)

    # Reference: replay the layer's own pipeline by hand from the fused qkv output.
    with torch.no_grad():
        qkv = layer.qkv_proj.forward(x)
        _, k_raw, v_raw = qkv.split(layer._qkv_split, dim=-1)
        want_v = v_raw.contiguous() * v_alpha
        k3 = k_raw.contiguous().view(-1, layer.num_kv, layer.head_dim)
        if apply == "pre_norm":
            k3 = layer.k_norm.forward(k3 * k_alpha.view(layer.num_kv, layer.head_dim))
        elif apply == "post_norm":
            k3 = layer.k_norm.forward(k3) * k_alpha.view(layer.num_kv, layer.head_dim)
        else:
            k3 = layer.k_norm.forward(k3)
        # q is only a placeholder for the rotary signature; the k rotation is what matters
        q_placeholder = torch.zeros(
            x.shape[0], layer.qo_attn_dim, device=fixture.device, dtype=fixture.dtype
        )
        _, want_k = layer.rotary.forward(
            batch.get_attn_positions(), q_placeholder, k3.view(-1, kv_dim)
        )
        if apply == "post_rope":
            want_k = want_k * k_alpha

    torch.testing.assert_close(capture.captured["v"], want_v, rtol=0, atol=0)
    torch.testing.assert_close(capture.captured["k"], want_k, rtol=2e-2, atol=2e-2)
