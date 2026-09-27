"""Machine-verifiable experiment manifests (colibri §7.5 experiment_manifest port).

An experiment claim ("window=4 is faster on this box") is only trustworthy if
its evidence is checkable by a script, not by reading a notebook. A manifest
pins:

- ``config``: the exact configuration dict, plus its sha256;
- ``samples``: >=3 raw metric samples; ``median`` must equal the recomputed
  median of ``samples`` (no cherry-picked summaries);
- ``artifacts``: name -> sha256 of every artifact the claim depends on
  (checkpoint file, log, etc.);
- ``quality``: ``{"gate": <name>, "passed": true}`` -- a claim without a
  passing quality gate is not a result.

``validate()`` recomputes everything and returns the list of violations; an
empty list means the manifest is self-consistent. CLI:
``python -m freetoken.moe.experiment_manifest verify manifest.json``
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

MIN_SAMPLES = 3


def config_sha256(config: dict) -> str:
    return hashlib.sha256(
        json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()


def median(samples: list[float]) -> float:
    s = sorted(samples)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def make_manifest(experiment_id: str, config: dict, samples: list[float],
                  artifacts: dict[str, str] | None = None,
                  quality_gate: str | None = None,
                  quality_passed: bool = False) -> dict:
    """Build a manifest with correctly computed fields (the honest path)."""
    return {
        "schema": 1,
        "experiment_id": experiment_id,
        "config": config,
        "config_sha256": config_sha256(config),
        "samples": list(samples),
        "median": median(samples) if samples else math.nan,
        "artifacts": artifacts or {},
        "quality": {"gate": quality_gate, "passed": bool(quality_passed)},
    }


def validate(manifest: dict) -> list[str]:
    """Recompute and check every pinned field; return violations (empty = ok)."""
    errors: list[str] = []
    if manifest.get("schema") != 1:
        errors.append(f"unknown schema {manifest.get('schema')!r}")
    if not manifest.get("experiment_id"):
        errors.append("missing experiment_id")

    config = manifest.get("config")
    if not isinstance(config, dict):
        errors.append("config must be a dict")
    elif manifest.get("config_sha256") != config_sha256(config):
        errors.append("config_sha256 does not match config")

    samples = manifest.get("samples")
    if not isinstance(samples, list) or len(samples) < MIN_SAMPLES:
        errors.append(f"samples must list >= {MIN_SAMPLES} measurements")
    elif not all(isinstance(x, (int, float)) for x in samples):
        errors.append("samples must be numeric")
    else:
        want = median(samples)
        got = manifest.get("median")
        if not isinstance(got, (int, float)) or abs(got - want) > 1e-9 * max(1.0, abs(want)):
            errors.append(f"median {got!r} != recomputed {want!r}")

    artifacts = manifest.get("artifacts", {})
    if not isinstance(artifacts, dict):
        errors.append("artifacts must be a dict of name -> sha256")
    else:
        for name, sha in artifacts.items():
            if not isinstance(sha, str) or len(sha) != 64:
                errors.append(f"artifact {name!r}: not a sha256 hex digest")
                continue
            p = Path(name)
            if p.is_file():
                actual = hashlib.sha256(p.read_bytes()).hexdigest()
                if actual != sha:
                    errors.append(f"artifact {name!r}: sha256 mismatch (file changed)")

    quality = manifest.get("quality", {})
    if not isinstance(quality, dict) or not quality.get("gate"):
        errors.append("quality.gate missing: which gate vouches for this result?")
    elif quality.get("passed") is not True:
        errors.append("quality.passed is not true: an ungated measurement is not a result")
    return errors


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="experiment_manifest",
                                 description="Verify an experiment manifest")
    ap.add_argument("command", choices=["verify"])
    ap.add_argument("manifest", type=Path)
    args = ap.parse_args(argv)
    manifest = json.loads(args.manifest.read_text())
    errors = validate(manifest)
    if errors:
        for e in errors:
            print(f"FAIL  {e}")
        return 1
    print(f"OK    {manifest.get('experiment_id')}: "
          f"median={manifest.get('median')} n={len(manifest.get('samples', []))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
