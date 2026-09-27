"""Tests for the machine-verifiable experiment manifest."""
import hashlib
import json

from freetoken.moe.experiment_manifest import make_manifest, median, validate, main


def _good(tmp_path):
    art = tmp_path / "run.log"
    art.write_text("log")
    return make_manifest(
        "window4-vs-0", {"window": 4, "model": "qwen38"}, [10.0, 11.0, 10.5],
        artifacts={str(art): hashlib.sha256(b"log").hexdigest()},
        quality_gate="greedy_consistency", quality_passed=True)


def test_honest_manifest_validates(tmp_path):
    assert validate(_good(tmp_path)) == []


def test_median_recomputation_catches_lies(tmp_path):
    m = _good(tmp_path)
    m["median"] = 999.0
    assert any("median" in e for e in validate(m))


def test_config_sha_pin(tmp_path):
    m = _good(tmp_path)
    m["config"]["window"] = 8  # tamper after pinning
    assert any("config_sha256" in e for e in validate(m))


def test_min_samples_enforced(tmp_path):
    m = _good(tmp_path)
    m["samples"] = [10.0, 11.0]
    m["median"] = 10.5
    assert any("samples" in e for e in validate(m))


def test_quality_gate_required(tmp_path):
    m = _good(tmp_path)
    m["quality"] = {"gate": "greedy_consistency", "passed": False}
    assert any("quality.passed" in e for e in validate(m))
    m["quality"] = {}
    assert any("quality.gate" in e for e in validate(m))


def test_artifact_tampering_detected(tmp_path):
    m = _good(tmp_path)
    (tmp_path / "run.log").write_text("edited")
    assert any("sha256 mismatch" in e for e in validate(m))


def test_cli(tmp_path, capsys):
    m = _good(tmp_path)
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(m))
    assert main(["verify", str(p)]) == 0
    m["median"] = 0.0
    p.write_text(json.dumps(m))
    assert main(["verify", str(p)]) == 1
