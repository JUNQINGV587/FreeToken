"""Tests for ft doctor (read-only deployment health check)."""
import json
import struct

import pytest
from safetensors.torch import save_file
import torch

from freetoken.doctor import check_model_dir, main


def _make_model(root, n_layers=2, n_exp=4, dup=False, truncate=False, index=True):
    cfg = {"num_hidden_layers": n_layers, "hidden_size": 8, "num_experts": n_exp,
           "tie_word_embeddings": False}
    (root / "config.json").write_text(json.dumps(cfg))
    tensors = {
        "model.embed_tokens.weight": torch.zeros(16, 8),
        "lm_head.weight": torch.zeros(16, 8),
        "model.norm.weight": torch.zeros(8),
    }
    for l in range(n_layers):
        tensors[f"model.layers.{l}.mlp.gate.weight"] = torch.zeros(n_exp, 8)
        for e in range(n_exp):
            tensors[f"model.layers.{l}.mlp.experts.{e}.w"] = torch.zeros(8)
    save_file(tensors, str(root / "s1.safetensors"))
    if dup:
        save_file({"lm_head.weight": torch.zeros(16, 8)}, str(root / "s2.safetensors"))
        index_map = {**{k: "s1.safetensors" for k in tensors}, "lm_head.weight": "s2.safetensors"}
    else:
        index_map = {k: "s1.safetensors" for k in tensors}
    if index:
        (root / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": index_map}))
    if truncate:
        p = root / "s1.safetensors"
        with open(p, "r+b") as f:
            f.truncate(16)


def _report(root):
    return {c["check"]: c for c in check_model_dir(root)}


def test_healthy_model_passes(tmp_path):
    _make_model(tmp_path)
    rep = _report(tmp_path)
    assert rep["shards.integrity"]["ok"] and rep["weights.experts_complete"]["ok"]
    assert rep["weights.lm_head"]["ok"] and rep["weights.final_norm"]["ok"]
    assert main([str(tmp_path)]) == 0


def test_truncated_shard_fails(tmp_path):
    _make_model(tmp_path, truncate=True)
    assert not _report(tmp_path)["shards.integrity"]["ok"]
    assert main([str(tmp_path)]) == 1


def test_duplicate_tensor_across_shards_fails(tmp_path):
    _make_model(tmp_path, dup=True)
    assert not _report(tmp_path)["shards.no_duplicates"]["ok"]


def test_incomplete_experts_fail(tmp_path):
    _make_model(tmp_path)
    # Rewrite shard with one expert missing on layer 1.
    cfg = json.loads((tmp_path / "config.json").read_text())
    tensors = {"model.embed_tokens.weight": torch.zeros(16, 8),
               "lm_head.weight": torch.zeros(16, 8), "model.norm.weight": torch.zeros(8)}
    for l in range(2):
        tensors[f"model.layers.{l}.mlp.gate.weight"] = torch.zeros(4, 8)
        for e in range(4 if l == 0 else 3):  # layer 1 missing expert 3
            tensors[f"model.layers.{l}.mlp.experts.{e}.w"] = torch.zeros(8)
    save_file(tensors, str(tmp_path / "s1.safetensors"))
    assert not _report(tmp_path)["weights.experts_complete"]["ok"]


def test_missing_config_fails_fast(tmp_path):
    rep = _report(tmp_path)
    assert not rep["config.present"]["ok"]


def test_json_output(tmp_path, capsys):
    _make_model(tmp_path)
    assert main([str(tmp_path), "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] and any(c["check"] == "shards.integrity" for c in out["checks"])


def test_not_a_directory_is_usage_error(tmp_path, capsys):
    assert main([str(tmp_path / "nope")]) == 2


def test_text_config_nesting(tmp_path):
    """Multimodal-style configs nest the LM under text_config (Qwen3.8-Flash-Next)."""
    _make_model(tmp_path)
    flat = json.loads((tmp_path / "config.json").read_text())
    (tmp_path / "config.json").write_text(json.dumps(
        {"model_type": "vl", "text_config": flat, "vision_config": {}}))
    rep = _report(tmp_path)
    assert rep["config.sane"]["ok"]
    assert "layers=2" in rep["config.sane"]["detail"]
