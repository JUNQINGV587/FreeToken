#!/usr/bin/env python3
"""Replay an ordered MoE route trace offline through admission policies.

This is the tool ``--moe-trace-route`` points at (freetoken/server/args.py,
freetoken/engine/engine.py): the capture side lives in ``freetoken/moe/route_trace.py``
and records every ``ensure_experts`` call's raw global expert ids in order, and this reads
that file back and answers the questions the histogram cannot -- how many expert copies a
given pool size would need under LRU, under LFU, and under a pinned hot set, on the actual
activation order.

Stdlib only (no torch/CUDA), so it runs anywhere the trace file does.

Typical use
-----------
    # capture (on an experiment instance; graphs must be off -- the recorder is host-side)
    ft serve ... --cuda-graph-max-bs 0 --moe-trace-route /data/tmp/route.bin

    # what does a bigger pool buy, and does pinning beat the LRU we already have?
    python3 tools/trace/replay_route_trace.py /data/tmp/route.bin \
        --cache-sizes 600,800,1000,1200 --policies lru,lfu,pin --pin-fracs 0.5 \
        --md /data/tmp/route_replay.md

Under TP>1 the recorder writes one file per rank (``<path>.rank0``, ``<path>.rank1``);
pass the base path with ``--ranks`` and the report adds the per-step slow side, which is
what sets the decode step time (the ranks are lockstep).

Every number here is a *miss count over the recorded ids*: it says what the pool would
have had to copy, not how long a copy took. ``--row-mib`` prices a copy (default 18.98 MiB,
the measured DeepSeek-V4.1-Flash-NVFP4 row) and the GiB column is an upper bound, both
because the engine may fetch only part of a row and because a miss that the pool would have
served is not necessarily a miss the disk tier pays for.
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import sys
from pathlib import Path


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # @dataclass resolves cls.__module__ via sys.modules
    spec.loader.exec_module(mod)
    return mod


_ROOT = Path(__file__).resolve().parents[2]
_ROUTE_TRACE = _load_module("route_trace", _ROOT / "python" / "freetoken" / "moe" / "route_trace.py")
_ROUTE_POLICY = _load_module("route_policy", _ROOT / "python" / "freetoken" / "moe" / "route_policy.py")
read_trace = _ROUTE_TRACE.read_trace
curve = _ROUTE_POLICY.curve


def _rank_traces(base: str) -> list[str]:
    """``base.rankN`` bodies only -- the ``.meta.json`` sidecars are not traces."""
    return [p for p in sorted(glob.glob(base + ".rank*")) if not p.endswith(".meta.json")]


def _resolve_traces(base: str, ranks: bool) -> list[str]:
    if os.path.exists(base):
        return [base]
    if ranks:
        found = _rank_traces(base)
        if found:
            return found
    ranked = _rank_traces(base)
    if ranked:
        raise SystemExit(
            f"{base} does not exist, but per-rank traces do ({', '.join(ranked)}); "
            f"pass --ranks to replay them (one pool per rank)")
    raise SystemExit(f"no such trace: {base}")


def _phase_of(name: str) -> int:
    return {"decode": 0, "prefill": 1, "all": -1}[name]


def _fmt_table(rows: list[dict], phase: str) -> str:
    head = ("| trace | policy | pool | pinned | calls | active | misses | miss% | "
            "GiB copied | miss/call (mean/p95/max) |")
    sep = "|---|---|---|---|---|---|---|---|---|---|"
    out = [head, sep]
    for r in rows:
        gib = "-" if r["copied_gib"] is None else f"{r['copied_gib']:.1f}"
        out.append(
            f"| {r['trace']} | {r['policy']} | {r['cache_size']} | {r['pin_rows']} | "
            f"{r['calls']} | {r['active']} | {r['misses']} | {100 * r['miss_rate']:.2f}% | "
            f"{gib} | {r['mean_call_miss']:.2f}/{r['p95_call_miss']}/{r['max_call_miss']} |"
        )
    return "\n".join(out) + f"\n\n_phase: {phase}_\n"


def _emit_pin(out_path: str, records, num_experts: int, num_layers: int, cache_size: int,
              fracs, mode: str, warmup_frac: float, phase_arg: str) -> None:
    """Write the learned hot set(s) the baked-pin loader would read.

    ``rows`` are global fids (``layer * num_experts + local_id``), which is the id space the
    engine's pin mask uses; ``by_layer`` keeps the local ids for readability and for a
    per-rank mask that does not need the global stride.
    """
    payload = {"num_experts": num_experts, "num_layers": num_layers,
               "cache_size": cache_size, "mode": mode, "warmup_frac": warmup_frac,
               "phase": phase_arg, "budgets": {}}
    for frac in fracs:
        budget = int(round(cache_size * frac))
        picked = _ROUTE_POLICY.learn_pin_set(records, budget, num_experts, mode=mode,
                                            warmup_frac=warmup_frac)
        rows = sorted(picked)
        by_layer: dict[str, list[int]] = {}
        for fid in rows:
            by_layer.setdefault(str(fid // num_experts), []).append(fid % num_experts)
        payload["budgets"][f"{frac:g}"] = {"frac": frac, "budget": budget,
                                           "pinned": len(rows), "rows": rows,
                                           "by_layer": by_layer}
    Path(out_path).write_text(json.dumps(payload, indent=1))
    print(f"wrote {out_path}", file=sys.stderr)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("trace", help="trace body path (or its base path with --ranks)")
    ap.add_argument("--ranks", action="store_true",
                    help="the given path is a base path; replay every <path>.rankN")
    ap.add_argument("--cache-sizes", default="600,800,1000,1100",
                    help="comma-separated pool sizes to test (default 600,800,1000,1100)")
    ap.add_argument("--policies", default="lru,lfu,pin",
                    help="comma-separated subset of lru,lfu,pin (default all three)")
    ap.add_argument("--pin-fracs", default="0.5",
                    help="comma-separated share of the pool reserved when pinning")
    ap.add_argument("--pin-mode", default="global", choices=("global", "per_layer"),
                    help="how the pinned hot set is chosen from the trace")
    ap.add_argument("--warmup-frac", type=float, default=1.0,
                    help="learn the pin set from this prefix of the trace; 1.0 = hindsight "
                         "(an upper bound, not a deployable policy), 0.2 = causal")
    ap.add_argument("--phase", default="decode", choices=("decode", "prefill", "all"))
    ap.add_argument("--ep-split", type=int, default=0, metavar="N",
                    help="model owner-local EP: keep only rank R's ids (needs --ep-rank)")
    ap.add_argument("--ep-rank", type=int, default=0)
    ap.add_argument("--row-mib", type=float, default=18.98,
                    help="size of one expert copy in MiB, to report GiB copied. Default "
                         "18.98 = the measured DeepSeek-V4.1-Flash-NVFP4 row "
                         "(layers.N.ffn.experts.E.*, 19,906,584 B); other models differ, and "
                         "the engine may fetch only part of a row, so treat GiB as an upper bound")
    ap.add_argument("--limit", type=int, default=0,
                    help="only replay the first N records of the wanted phase (quick runs)")
    ap.add_argument("--emit-pin", default=None, metavar="PATH",
                    help="also write the learned pin set as JSON, for the baked-pin "
                         "implementation (one file per trace; --ranks appends .rankN)")
    ap.add_argument("--emit-cache-size", type=int, default=0,
                    help="pool size the emitted pin set is sized for "
                         "(default: the last --cache-sizes entry)")
    ap.add_argument("--json", default=None, help="write the full result table as JSON")
    ap.add_argument("--md", default=None, help="write the result table as Markdown")
    args = ap.parse_args(argv)

    sizes = [int(x) for x in args.cache_sizes.split(",") if x.strip()]
    policies = tuple(p.strip() for p in args.policies.split(",") if p.strip())
    fracs = tuple(float(x) for x in args.pin_fracs.split(",") if x.strip())
    bad = [p for p in policies if p not in _ROUTE_POLICY.POLICIES]
    if bad:
        raise SystemExit(f"unknown policy/policies: {bad}; known: {_ROUTE_POLICY.POLICIES}")

    row_bytes = int(args.row_mib * 1024 * 1024) if args.row_mib else None
    phase = _phase_of(args.phase)
    traces = _resolve_traces(args.trace, args.ranks)

    rows: list[dict] = []
    for path in traces:
        meta, records = read_trace(path)
        if phase >= 0:
            records = [r for r in records if r[0] == phase]
        else:
            # "all" mixes phases into one timeline; the policy state spans both, which is
            # what the engine's single pool actually does, but prefill and decode ids then
            # compete for slots. Reported separately per phase below.
            pass
        if args.limit:
            records = records[: args.limit]
        if not records:
            print(f"{path}: no {args.phase} records", file=sys.stderr)
            continue
        num_experts = meta.num_experts
        if args.ep_split > 1:
            lo = args.ep_rank * num_experts // args.ep_split
            hi = (args.ep_rank + 1) * num_experts // args.ep_split
            kept = []
            for ph, layer, ids in records:
                local = tuple(e - lo for e in ids if lo <= e < hi)
                kept.append((ph, layer, local))
            records = kept
            num_experts = hi - lo
        for res in curve(records, sizes, num_experts, policies=policies,
                         pin_fracs=fracs, warmup_frac=args.warmup_frac,
                         pin_mode=args.pin_mode, row_bytes=row_bytes,
                         phase=(phase if phase >= 0 else 0)):
            row = res.as_dict()
            row["trace"] = os.path.basename(path)
            row["meta"] = dict(meta.__dict__)
            rows.append(row)
        if args.emit_pin:
            emit = args.emit_pin
            if len(traces) > 1:
                emit = f"{args.emit_pin}.{os.path.basename(path).rsplit('.', 1)[-1]}"
            _emit_pin(emit, records, num_experts, meta.num_layers,
                      args.emit_cache_size or sizes[-1], fracs, args.pin_mode,
                      args.warmup_frac, args.phase)

    table = _fmt_table(rows, args.phase)
    print(table)
    if args.md:
        Path(args.md).write_text(
            f"# MoE route replay: {os.path.basename(args.trace)}\n\n"
            f"capture meta: `{json.dumps(dict(rows[0]['meta'])) if rows else '{}'}`\n\n"
            f"pin budget from `--pin-fracs {args.pin_fracs}` "
            f"(`{args.pin_mode}`, warmup `{args.warmup_frac}`), "
            f"copies priced at `{args.row_mib}` MiB/row.\n\n" + table)
        print(f"wrote {args.md}", file=sys.stderr)
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1))
        print(f"wrote {args.json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
