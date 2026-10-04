#!/usr/bin/env python3
"""Generate a learned per-layer RAM pin set for the disk tier from route traces.

The disk tier pins the first ``ram_experts`` local rows of every layer in host RAM
(``disk_tier.py``: ``self._ram``) and serves slot-cache misses on those rows from pinned
memory instead of NVMe.  A *contiguous prefix* is the default only because it needs no
input: measured on DeepSeek-V4.1-Flash route traces it covers exactly the uniform-random
share (budget / local_num) of the rows that actually miss.  This tool learns which rows
are hot -- per layer, per EP rank -- from recorded routing, and writes a pin file that
``ft serve --moe-ram-pin-file`` loads at startup to pin those rows instead.

Stdlib only (no torch/CUDA), same convention as ``replay_route_trace.py``: pass trace
base paths (the recorder writes ``<base>.rank<N>`` per rank under TP/EP).

Typical use
-----------

    python3 tools/trace/make_ram_pin_set.py \
        --trace /data/research/runs/20261002-hotset/capture2/route.bin \
        --trace /data/research/runs/20261002-hotset/capture3/route.bin \
        --trace /data/research/runs/20261002-hotset/capture4/route.bin \
        --ep 2 --budgets 64,56 \
        --out /data/research/runs/20261002-hotset/pin_set_v1.json

Only decode-phase records count by default (prefill routing is the saturated coupon
collector and would flatten every histogram to uniform); pass ``--phase all`` to
include prefill anyway.  The report prints self-fit coverage (optimistic) and, with
more than one trace, leave-one-trace-out coverage -- the honest estimate of how much a
pin set learned on other windows still covers a window it has never seen.
"""
from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # @dataclass resolves cls.__module__ via sys.modules
    spec.loader.exec_module(mod)
    return mod


_ROUTE_TRACE = _load_module("route_trace", _ROOT / "python" / "freetoken" / "moe" / "route_trace.py")
read_trace = _ROUTE_TRACE.read_trace

FORMAT = "freetoken.ram_pin_set.v1"
PHASES = {"decode": (0,), "all": (0, 1)}


def local_window(rank: int, ep: int, num_experts: int) -> tuple[int, int]:
    """``[lo, hi)`` global ids owned by ``rank`` -- mirrors DiskTier's EP split."""
    lo = rank * num_experts // ep
    hi = (rank + 1) * num_experts // ep
    return lo, hi


def count_freq(records, lo: int, hi: int, num_layers: int, phases=(0,)):
    """Per-layer Counter of LOCAL expert ids over the given phases."""
    freq = [collections.Counter() for _ in range(num_layers)]
    for ph, layer, ids in records:
        if ph not in phases or layer >= num_layers:
            continue
        for e in set(ids):  # one activation per expert per record, like route_hist
            if lo <= e < hi:
                freq[layer][e - lo] += 1
    return freq


def topk_per_layer(freq, budget: int, local_num: int) -> list[list[int]]:
    """Hottest ``budget`` local ids per layer, ascending.  Deterministic ties (lowest
    id); layers with fewer distinct ids seen are padded with the lowest unseen ids so
    every layer always carries exactly ``budget`` rows."""
    out = []
    for layer_freq in freq:
        chosen = sorted(e for e, _ in sorted(layer_freq.items(), key=lambda kv: (-kv[1], kv[0]))[:budget])
        if len(chosen) < budget:  # trace too short to saturate: pad deterministically
            chosen_set = set(chosen)
            chosen += [e for e in range(local_num) if e not in chosen_set][: budget - len(chosen)]
        out.append(sorted(chosen))
    return out


def coverage(records, pin_rows, lo: int, hi: int, phases=(0,)) -> tuple[float, int]:
    """Share of decode activations whose row is pinned (unconditional, i.e. before the
    slot cache -- the pin set also eats LRU hits, which costs nothing)."""
    hit = tot = 0
    for ph, layer, ids in records:
        if ph not in phases:
            continue
        pinned = pin_rows[layer]
        for e in set(ids):
            if lo <= e < hi:
                tot += 1
                hit += (e - lo) in pinned
    return (hit / tot if tot else 0.0), tot


def build_doc(traces, ep: int, budgets, num_layers: int, num_experts: int, phases=(0,)):
    """Learn per-rank pin rows from ``[(name, records), ...]``; return the JSON doc."""
    ranks = {}
    for rank in range(ep):
        lo, hi = local_window(rank, ep, num_experts)
        freq = [collections.Counter() for _ in range(num_layers)]
        for _, records in traces:
            layer_freq = count_freq(records, lo, hi, num_layers, phases)
            for layer in range(num_layers):
                freq[layer].update(layer_freq[layer])
        ranks[str(rank)] = topk_per_layer(freq, budgets[rank], hi - lo)
    return {
        "format": FORMAT,
        "num_layers": num_layers,
        "num_experts": num_experts,
        "ep": ep,
        "budgets": list(budgets),
        "sources": [name for name, _ in traces],
        "ranks": ranks,
    }


def validate_doc(doc) -> None:
    """Shape checks the engine applies at load; raises ``ValueError`` on any mismatch."""
    if doc.get("format") != FORMAT:
        raise ValueError(f"bad format {doc.get('format')!r} (want {FORMAT!r})")
    L, E, ep = doc["num_layers"], doc["num_experts"], doc["ep"]
    budgets = doc["budgets"]
    ranks = doc["ranks"]
    if len(budgets) != ep or sorted(ranks) != [str(r) for r in range(ep)]:
        raise ValueError("budgets/ranks do not match ep")
    for rank in range(ep):
        lo, hi = local_window(rank, ep, E)
        rows = ranks[str(rank)]
        if len(rows) != L:
            raise ValueError(f"rank {rank}: {len(rows)} layers, want {L}")
        for layer, ids in enumerate(rows):
            if len(ids) != budgets[rank] or len(set(ids)) != len(ids):
                raise ValueError(f"rank {rank} layer {layer}: want {budgets[rank]} unique ids")
            if ids != sorted(ids) or ids[0] < 0 or ids[-1] >= hi - lo:
                raise ValueError(f"rank {rank} layer {layer}: ids must be ascending in [0, {hi - lo})")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trace", action="append", required=True,
                    help="trace base path (writes/read <base>.rank<N>); repeatable")
    ap.add_argument("--ep", type=int, required=True, help="EP size (number of rank files per trace)")
    ap.add_argument("--budgets", required=True,
                    help="per-rank pin budgets, comma-separated (e.g. 64,56); one value = uniform")
    ap.add_argument("--phase", choices=sorted(PHASES), default="decode")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    budgets = [int(x) for x in args.budgets.split(",")]
    if len(budgets) == 1:
        budgets *= args.ep
    if len(budgets) != args.ep or any(b <= 0 for b in budgets):
        ap.error(f"--budgets must give {args.ep} positive integers (or one uniform value)")
    phases = PHASES[args.phase]

    traces = []
    meta0 = None
    for base in args.trace:
        for rank in range(args.ep):
            path = f"{base}.rank{rank}"
            if not Path(path).exists():
                raise SystemExit(f"missing trace file: {path}")
        meta, _ = read_trace(f"{base}.rank0")
        if meta0 is None:
            meta0 = meta
        elif (meta.num_layers, meta.num_experts) != (meta0.num_layers, meta0.num_experts):
            raise SystemExit(f"{base}: geometry mismatch ({meta.num_layers}x{meta.num_experts} vs "
                             f"{meta0.num_layers}x{meta0.num_experts})")
        traces.append((base, meta))

    per_trace = {name: [read_trace(f"{name}.rank{r}")[1] for r in range(args.ep)] for name, _ in traces}
    L, E = meta0.num_layers, meta0.num_experts
    flat = [(name, recs) for name, rank_recs in per_trace.items() for recs in [None]]  # placeholder
    combined = [(name, [ph_l_ids for recs in rank_recs for ph_l_ids in recs])
                for name, rank_recs in per_trace.items()]
    doc = build_doc(combined, args.ep, budgets, L, E, phases)
    validate_doc(doc)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1) + "\n")

    print(f"wrote {out}  (layers={L} experts={E} ep={args.ep} budgets={budgets} phase={args.phase})")
    for rank in range(args.ep):
        lo, hi = local_window(rank, args.ep, E)
        pin = [set(rows) for rows in doc["ranks"][str(rank)]]
        for name, rank_recs in per_trace.items():
            c, tot = coverage(rank_recs[rank], pin, lo, hi, phases)
            print(f"  rank{rank} self-fit on {Path(name).parent.name}: {c:.4f} over {tot} acts")
        if len(per_trace) > 1:
            for name, rank_recs in per_trace.items():
                others = [(n, rr) for n, rr in per_trace.items() if n != name]
                sub = build_doc([(n, [x for recs in rrs for x in recs]) for n, rrs in others],
                                args.ep, budgets, L, E, phases)
                sub_pin = [set(rows) for rows in sub["ranks"][str(rank)]]
                c, tot = coverage(rank_recs[rank], sub_pin, lo, hi, phases)
                print(f"  rank{rank} leave-out {Path(name).parent.name}: {c:.4f} over {tot} acts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
