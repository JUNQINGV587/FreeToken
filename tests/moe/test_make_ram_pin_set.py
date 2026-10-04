"""Tests for tools/trace/make_ram_pin_set.py: frequency counting, deterministic top-k,
the pin-file schema/validation, coverage, and the CLI end to end on a synthetic trace.

Pure CPU, no torch.  The tool is loaded by path (it is stdlib-only); the synthetic trace
uses the recorder's on-disk format directly (``<bii`` header + int32 ids + meta.json).
"""
from __future__ import annotations

import importlib.util
import json
import struct
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, _ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_gen = _load("make_ram_pin_set", "tools/trace/make_ram_pin_set.py")
_HDR = struct.Struct("<bii")

E, L = 8, 3  # tiny geometry: 8 experts, 3 layers


def _records(rank0_pairs, rank1_pairs=None):
    """``[(phase, layer, ids)]``; rank1 defaults to the same ids shifted into [4,8)."""
    if rank1_pairs is None:
        rank1_pairs = [(ph, layer, tuple(i + E // 2 for i in ids)) for ph, layer, ids in rank0_pairs]
    return rank0_pairs, rank1_pairs


def test_local_window_partitions():
    assert _gen.local_window(0, 2, 384) == (0, 192)
    assert _gen.local_window(1, 2, 384) == (192, 384)


def test_count_freq_decode_only_and_rebase():
    recs = [(0, 0, (1, 2, 2, 7)), (1, 0, (1, 1)), (0, 1, (3,)), (0, 1, (6,))]  # ph=1 ignored
    freq = _gen.count_freq(recs, 0, 4, L, phases=(0,))
    assert freq[0] == {1: 1, 2: 1}  # dedup within a record; 7 out of window
    assert freq[1] == {3: 1}  # 6 out of window [0,4)
    assert freq[2] == {}


def test_count_freq_includes_prefill_when_asked():
    recs = [(1, 0, (0, 1))]
    assert _gen.count_freq(recs, 0, 4, L, phases=(0,))[0] == {}
    assert _gen.count_freq(recs, 0, 4, L, phases=(0, 1))[0] == {0: 1, 1: 1}


def test_topk_deterministic_tiebreak_and_padding():
    freq = [{0: 5, 1: 5, 2: 3}, {3: 1}, {}]  # ties on layer 0; deficits on 1 and 2
    rows = _gen.topk_per_layer(freq, budget=2, local_num=4)
    assert rows[0] == [0, 1]  # tie 5==5 -> lowest ids
    assert rows[1] == [0, 3]  # only id 3 seen -> pad with lowest unseen (0)
    assert rows[2] == [0, 1]  # nothing seen -> first ids
    assert all(len(r) == 2 and r == sorted(r) for r in rows)


def _doc():
    recs0, recs1 = _records([(0, 0, (0, 1, 5)), (0, 1, (1, 2, 6)), (0, 2, (0, 3, 7))])
    traces = [("t", recs0 + recs1)]
    return _gen.build_doc(traces, ep=2, budgets=(2, 1), num_layers=L, num_experts=E)


def test_build_doc_and_validate_round_trip():
    doc = _doc()
    _gen.validate_doc(doc)
    assert doc["ranks"]["0"][0] == [0, 1]  # ids 0,1 (5 is rank1's)
    assert doc["ranks"]["1"][0] == [1]  # global 5 -> local 1
    assert doc["budgets"] == [2, 1]


def test_validate_rejects_bad_docs():
    doc = _doc()
    for mutate in (
        lambda d: d.update(format="nope"),
        lambda d: d["ranks"]["0"].__setitem__(0, [0, 0]),  # dup
        lambda d: d["ranks"]["0"].__setitem__(0, [1, 0]),  # unsorted
        lambda d: d["ranks"]["0"].__setitem__(0, [0, 9]),  # out of window
        lambda d: d["ranks"]["0"].pop(),  # wrong layer count
        lambda d: d.update(budgets=[2]),  # budget/ep mismatch
    ):
        bad = json.loads(json.dumps(doc))
        mutate(bad)
        try:
            _gen.validate_doc(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"validate accepted bad doc: {bad}")


def test_coverage_counts_window_and_dedups():
    recs = [(0, 0, (0, 0, 1, 5))]  # 5 is rank1's; 0 deduped
    pin = [{0, 1}, set(), set()]
    c, tot = _gen.coverage(recs, pin, 0, 4)
    assert (c, tot) == (1.0, 2)


def _write_trace(base: Path, rank: int, records):
    body = base.parent / f"{base.name}.rank{rank}"
    with open(body, "wb") as f:
        for ph, layer, ids in records:
            f.write(_HDR.pack(ph, layer, len(ids)))
            f.write(struct.pack(f"<{len(ids)}i", *ids))
    meta = {"num_experts": E, "num_layers": L, "cache_size": 100, "top_k": 2,
            "model": "synthetic", "records": len(records)}
    body.with_suffix(body.suffix + ".meta.json").write_text(json.dumps(meta))


def test_cli_end_to_end(tmp_path):
    base = tmp_path / "route.bin"
    recs0, recs1 = _records([(0, 0, (0, 1, 4)), (0, 0, (0, 2, 5)), (1, 1, (3, 3))])
    _write_trace(base, 0, recs0)
    _write_trace(base, 1, recs1)
    out = tmp_path / "pin.json"
    rc = subprocess.run(
        [sys.executable, str(_ROOT / "tools/trace/make_ram_pin_set.py"),
         "--trace", str(base), "--ep", "2", "--budgets", "2", "--out", str(out)],
        capture_output=True, text=True)
    assert rc.returncode == 0, rc.stderr
    doc = json.loads(out.read_text())
    _gen.validate_doc(doc)
    assert doc["ranks"]["0"][0] == [0, 1]  # counts: 0->2, 1->1, 2->1 (tie -> lowest)
    assert doc["ranks"]["1"][0] == [0, 1]  # globals 4,5 -> locals 0,1
    assert "self-fit" in rc.stdout
