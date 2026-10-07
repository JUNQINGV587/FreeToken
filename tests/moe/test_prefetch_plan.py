"""CPU tests for moe/prefetch_plan.py -- the pure planning half of the 2b
block-level bulk prefetch (prediction-doc loading, rolling plan generation,
block formation). No torch, no CUDA, no checkpoint needed."""

import json

import pytest

from freetoken.moe.prefetch_plan import (
    RANKED_FORMAT, build_prefetch_plan, form_blocks, load_prediction_doc)


def _write(tmp_path, doc):
    p = tmp_path / "pred.json"
    p.write_text(json.dumps(doc))
    return str(p)


def test_load_histogram_doc_drops_zero_scores(tmp_path):
    path = _write(tmp_path, {"layers": 2, "experts": 3,
                             "hist": [[0, 1.5, 0], [2, 0, 3]]})
    L, E, scores = load_prediction_doc(path)
    assert (L, E) == (2, 3)
    assert scores == [{1: 1.5}, {0: 2.0, 2: 3.0}]


def test_load_ranked_doc(tmp_path):
    path = _write(tmp_path, {"format": RANKED_FORMAT, "num_layers": 2,
                             "num_experts": 4,
                             "layers": [[[3, 0.9], [1, 0.5]], []]})
    L, E, scores = load_prediction_doc(path)
    assert (L, E) == (2, 4)
    assert scores == [{3: 0.9, 1: 0.5}, {}]


@pytest.mark.parametrize("doc", [
    [1, 2, 3],                                            # not an object
    {"layers": 2, "experts": 3},                          # unknown shape
    {"layers": 2, "experts": 3, "hist": [[0, 0, 0]]},     # wrong row count
    {"layers": 1, "experts": 3, "hist": [[0, 0]]},        # wrong row width
    {"layers": 0, "experts": 3, "hist": []},              # non-positive dims
    {"format": RANKED_FORMAT, "num_layers": 1, "num_experts": 2,
     "layers": [[[2, 1.0]]]},                             # expert id out of range
    {"format": RANKED_FORMAT, "num_layers": 1, "num_experts": 2,
     "layers": [[[True, 1.0]]]},                          # bool is not an id
    {"format": RANKED_FORMAT, "num_layers": 1, "num_experts": 2,
     "layers": [[[0, "fast"]]]},                          # bad score
])
def test_load_rejects_malformed_docs(tmp_path, doc):
    path = _write(tmp_path, doc)
    with pytest.raises(ValueError):
        load_prediction_doc(path)


def _scores(rows):
    return [dict(r) for r in rows]


def test_build_plan_window_topk_floor_and_exclusions():
    # 4 layers x 6 experts; experts 0/1 are RAM-pinned (row_map identity, ram=2).
    scores = _scores([
        {},
        {0: 9.0, 2: 1.0, 3: 0.8, 4: 0.4, 5: 0.1},   # pinned 0 must be excluded
        {2: 1.0, 3: 0.49, 4: 0.5},                  # floor 0.5 x best = 0.5
        {5: 1.0},
    ])
    row_map = [list(range(6)) for _ in range(4)]
    resident = [set(), set(), {4}, set()]
    plan = build_prefetch_plan(scores, row_map, 2, start_layer=0, depth=2,
                               top_k=2, min_frac=0.5, resident=resident)
    # Layer 1: candidates {2:1.0, 3:0.8} after pin exclusion + floor 4.5? No --
    # the floor is min_frac x the layer's best SCORE among all scores (9.0 is
    # the pinned expert's, so the floor is 4.5 and only... nothing qualifies).
    assert plan.get(1) is None
    # Layer 2: floor = 0.5 x 1.0 = 0.5 -> {2:1.0, 4:0.5}, but 4 is resident.
    assert plan[2] == [2]


def test_build_plan_orders_by_score_then_id():
    scores = _scores([{}, {2: 0.5, 3: 0.9, 4: 0.9, 5: 0.1}])
    row_map = [list(range(6)) for _ in range(2)]
    plan = build_prefetch_plan(scores, row_map, 2, start_layer=0, depth=1,
                               top_k=2)
    assert plan[1] == [3, 4]  # tie on 0.9 broken by id


def test_build_plan_bounds_and_empty():
    scores = _scores([{2: 1.0}])
    row_map = [list(range(4))]
    assert build_prefetch_plan(scores, row_map, 2, start_layer=0, depth=0,
                               top_k=4) == {}
    assert build_prefetch_plan(scores, row_map, 2, start_layer=0, depth=1,
                               top_k=0) == {}
    # Window past the last layer is empty, never an IndexError.
    assert build_prefetch_plan(scores, row_map, 2, start_layer=0, depth=8,
                               top_k=4) == {}
    # Everything pinned -> no candidates.
    assert build_prefetch_plan(scores, row_map, 4, start_layer=-1, depth=1,
                               top_k=4) == {}


def test_form_blocks_sorts_and_chunks():
    assert form_blocks([5, 2, 3, 0, 4], 2) == [[0, 2], [3, 4], [5]]
    assert form_blocks([3, 1], 4) == [[1, 3]]
    assert form_blocks([2, 0], 1) == [[0], [2]]
    assert form_blocks([], 4) == []
