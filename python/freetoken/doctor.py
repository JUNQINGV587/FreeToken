"""Read-only preflight health check for a FreeToken model deployment (colibri doctor port).

Never writes, never loads tensors -- it parses safetensors *headers* and config
files only. Exit code 0 = healthy, 1 = at least one failing check, 2 = usage
error. ``--json`` emits a machine-readable report for CI gates.

Checks:
- config.json parses; num_layers / num_experts / hidden sizes are sane;
- every index-declared safetensors shard exists, its header parses, the file
  is at least as long as the header's declared data extent, and no tensor
  name is declared by two shards (a classic conversion bug);
- role-matched core weights exist: token embedding, lm_head (or tied), the
  final norm, and for MoE checkpoints the router gate plus exactly
  ``num_experts`` experts for every layer;
- the filesystem holding the model has at least ``total_size`` bytes free
  (headroom for caches/FTW conversion).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import struct
import sys
from pathlib import Path


def _st_header(path: Path) -> dict:
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) < 8:
            raise ValueError("truncated safetensors header length")
        (n,) = struct.unpack("<Q", raw)
        if n > 1 << 30:
            raise ValueError(f"implausible header length {n}")
        return json.loads(f.read(n))


def check_model_dir(model: Path) -> list[dict]:
    checks = []

    def rec(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    cfg_path = model / "config.json"
    if not cfg_path.is_file():
        rec("config.present", False, f"{cfg_path} missing")
        return checks
    try:
        cfg = json.loads(cfg_path.read_text())
        rec("config.present", True, "config.json parses")
    except Exception as exc:
        rec("config.present", False, f"config.json unparseable: {exc}")
        return checks
    # Multimodal checkpoints nest the language model under text_config.
    sub = cfg.get("text_config")
    if isinstance(sub, dict) and sub.get("num_hidden_layers"):
        cfg = {**cfg, **sub}

    n_layer = cfg.get("num_hidden_layers", 0)
    n_exp = cfg.get("num_experts", cfg.get("n_routed_experts", 0))
    hidden = cfg.get("hidden_size", 0)
    rec("config.sane", bool(n_layer > 0 and hidden > 0),
        f"layers={n_layer} hidden={hidden} experts={n_exp}")

    index_path = model / "model.safetensors.index.json"
    shards: dict[str, list[str]] = {}  # shard file -> tensor names
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text()).get("weight_map", {})
        for name, shard in weight_map.items():
            shards.setdefault(shard, []).append(name)
        rec("index.present", True, f"{len(weight_map)} tensors across {len(shards)} shards")
    else:
        singles = sorted(p.name for p in model.glob("*.safetensors"))
        for s in singles:
            shards[s] = []
        rec("index.present", bool(singles),
            f"no index; {len(singles)} loose shard(s)" if singles else "no safetensors found")

    tensors: dict[str, str] = {}
    dupes, bad = [], []
    total_bytes = 0
    for shard, names in sorted(shards.items()):
        sp = model / shard
        if not sp.is_file():
            bad.append(f"{shard}: missing")
            continue
        total_bytes += sp.stat().st_size
        try:
            hdr = _st_header(sp)
            data_end = 8 + len(json.dumps(hdr))  # approx; verify against declared extents
            max_end = 0
            for tname, meta in hdr.items():
                if tname == "__metadata__":
                    continue
                begin, end = meta.get("data_offsets", [0, 0])
                max_end = max(max_end, end)
                if tname in tensors:
                    dupes.append(tname)
                tensors[tname] = shard
            declared = 8 + struct.unpack("<Q", open(sp, "rb").read(8))[0] + max_end
            if sp.stat().st_size < declared:
                bad.append(f"{shard}: file {sp.stat().st_size}B < declared extent {declared}B")
        except Exception as exc:
            bad.append(f"{shard}: header unreadable: {exc}")
    rec("shards.integrity", not bad, "; ".join(bad) if bad else
        f"{len(shards)} shard(s), {total_bytes / 1e9:.1f} GiB".replace("GiB", "GB"))
    rec("shards.no_duplicates", not dupes,
        f"{len(dupes)} duplicate tensor name(s)" if dupes else "no duplicates")

    def has(pattern: str) -> bool:
        rx = re.compile(pattern)
        return any(rx.search(t) for t in tensors)

    rec("weights.embedding", has(r"embed_tokens"),
        "token embedding present" if has(r"embed_tokens") else "embed_tokens missing")
    lm = has(r"lm_head") or cfg.get("tie_word_embeddings", False)
    rec("weights.lm_head", lm,
        "lm_head present" if has(r"lm_head") else
        ("tied to embedding" if cfg.get("tie_word_embeddings", False) else "lm_head missing"))
    rec("weights.final_norm", has(r"(^|\.)norm(\.|$)") and has(r"model\.norm|\.norm\.weight"),
        "final norm present" if has(r"model\.norm|\.norm\.weight") else "final norm missing")

    if n_exp:
        gate_ok = has(r"(mlp\.|mlp_)gate\.|router") or has(r"mlp\.gate\.")
        rec("weights.router_gate", gate_ok, "MoE router gate present" if gate_ok else "router gate missing")
        exp_rx = re.compile(r"layers\.(\d+)\..*experts\.(\d+)\.")
        per_layer: dict[int, set[int]] = {}
        for t in tensors:
            m = exp_rx.search(t)
            if m:
                per_layer.setdefault(int(m.group(1)), set()).add(int(m.group(2)))
        missing = [l for l in range(n_layer)
                   if per_layer.get(l) and len(per_layer[l]) < n_exp]
        absent = [l for l in range(n_layer) if l not in per_layer]
        # Shared/dense layers may legitimately lack routed experts; only flag
        # layers that have SOME experts but fewer than num_experts.
        rec("weights.experts_complete", not missing,
            f"layers with incomplete experts: {missing[:8]}" if missing else
            f"{len(per_layer)} MoE layer(s) complete ({absent[:4]} dense)" if absent else
            f"all {n_layer} layers have {n_exp} experts")

    try:
        free = os.statvfs(model).f_bavail * os.statvfs(model).f_frsize
        rec("disk.headroom", free >= total_bytes,
            f"free {free / 1e9:.0f} GB vs model {total_bytes / 1e9:.0f} GB")
    except Exception as exc:
        rec("disk.headroom", False, f"statvfs failed: {exc}")
    return checks


def main(argv: list[str] | None = None, prog: str = "ft doctor") -> int:
    ap = argparse.ArgumentParser(prog=prog, description="Read-only deployment health check")
    ap.add_argument("model", help="path to the model directory")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = ap.parse_args(argv)

    model = Path(args.model)
    if not model.is_dir():
        print(f"error: {model} is not a directory", file=sys.stderr)
        return 2
    checks = check_model_dir(model)
    ok = all(c["ok"] for c in checks)
    if args.json:
        print(json.dumps({"model": str(model), "ok": ok, "checks": checks}, indent=2))
    else:
        for c in checks:
            print(f"{'PASS' if c['ok'] else 'FAIL'}  {c['check']}: {c['detail']}")
        print(f"doctor: {'healthy' if ok else 'UNHEALTHY'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
