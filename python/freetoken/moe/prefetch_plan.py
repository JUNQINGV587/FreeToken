"""Block-level bulk prefetch planning for the disk tier (port plan 2b).

Cold prefill leaves the GPU idle ~93.5% of the layer window waiting on NVMe
demand reads (runs/20261007-port-battery TIMELINE). The demand reads already
saturate the disk inside their window, so the only harvest is the disk-idle
compute windows (GEMM + attention, ~6.4% of the cold chunk) -- see
notes/engines/20261007-bulk-prefetch-design.md section 2 for the honest
ceiling. This module is the pure host-side planning half of the mechanism:

* :func:`load_prediction_doc` -- load the LEARNED per-layer expert score
  distribution the speculation is driven by (never an identity predictor:
  the PILOT decode prefetch measured 1.01% precision with one and was a net
  loss on a saturated disk);
* :func:`build_prefetch_plan` -- roll the distribution into a per-layer
  candidate list (top-k, confidence threshold, pinned/slot-resident
  exclusion, lookahead depth);
* :func:`form_blocks` -- chunk a layer's candidates into read blocks so
  adjacent experts' file extents merge into fewer preadv calls (the R1x
  ``_row_groups`` idea, extended across experts).

Everything here is plain Python over lists/dicts -- no torch, no CUDA -- so
the planner is unit-testable on CPU in isolation from the tier.
"""

from __future__ import annotations

import json

HIST_DOC_KEYS = ("layers", "experts", "hist")
RANKED_FORMAT = "freetoken.prefetch_plan.v1"


def load_prediction_doc(path: str) -> tuple[int, int, list[dict[int, float]]]:
    """Load and validate a bulk-prefetch prediction document.

    Two accepted shapes:

    * histogram: ``{"layers": L, "experts": E, "hist": [[float, ...], ...]}``
      (the ``DiskTier.save_histogram`` payload) -- one score per (layer,
      GLOBAL expert id);
    * ranked: ``{"format": "freetoken.prefetch_plan.v1", "num_layers": L,
      "num_experts": E, "layers": [[[expert, score], ...], ...]}`` -- per
      layer a list of (expert, score) pairs, the natural output of an
      offline route-trace learner.

    Returns ``(num_layers, num_experts, scores)`` with scores as one
    ``{expert_id: float}`` dict per layer (sparse; unrouted experts simply
    absent). Raises ValueError on any malformed content -- a bad prediction
    file must fail the boot, never speculate from garbage.
    """
    with open(path) as f:
        doc = json.load(f)
    if not isinstance(doc, dict):
        raise ValueError(f"{path}: prediction doc must be a JSON object")
    if doc.get("format") == RANKED_FORMAT:
        return _parse_ranked(path, doc)
    if all(k in doc for k in HIST_DOC_KEYS):
        return _parse_histogram(path, doc)
    raise ValueError(
        f"{path}: unknown prediction doc shape (want {RANKED_FORMAT} or a "
        "save_histogram payload with layers/experts/hist)")


def _parse_ranked(path: str, doc: dict) -> tuple[int, int, list[dict[int, float]]]:
    try:
        L = int(doc["num_layers"])
        E = int(doc["num_experts"])
        layers = doc["layers"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{path}: malformed {RANKED_FORMAT} document: {exc}") from exc
    if not (L > 0 and E > 0):
        raise ValueError(f"{path}: num_layers/num_experts must be positive")
    if not isinstance(layers, list) or len(layers) != L:
        raise ValueError(f"{path}: layers must have {L} entries")
    scores: list[dict[int, float]] = []
    for layer, pairs in enumerate(layers):
        if not isinstance(pairs, list):
            raise ValueError(f"{path}: layers[{layer}] must be a list of [expert, score]")
        per: dict[int, float] = {}
        for item in pairs:
            if (not isinstance(item, (list, tuple)) or len(item) != 2
                    or not isinstance(item[0], int) or isinstance(item[0], bool)
                    or not 0 <= item[0] < E):
                raise ValueError(
                    f"{path}: layers[{layer}] entries must be [expert in [0,{E}), score]")
            try:
                per[item[0]] = float(item[1])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}: layers[{layer}] bad score: {exc}") from exc
        scores.append(per)
    return L, E, scores


def _parse_histogram(path: str, doc: dict) -> tuple[int, int, list[dict[int, float]]]:
    try:
        L = int(doc["layers"])
        E = int(doc["experts"])
        hist = doc["hist"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{path}: malformed histogram document: {exc}") from exc
    if not (L > 0 and E > 0):
        raise ValueError(f"{path}: layers/experts must be positive")
    if not isinstance(hist, list) or len(hist) != L:
        raise ValueError(f"{path}: hist must have {L} rows")
    scores: list[dict[int, float]] = []
    for layer, row in enumerate(hist):
        if not isinstance(row, list) or len(row) != E:
            raise ValueError(f"{path}: hist[{layer}] must have {E} entries")
        per: dict[int, float] = {}
        for e, v in enumerate(row):
            try:
                v = float(v)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}: hist[{layer}][{e}] not a number: {exc}") from exc
            if v > 0:
                per[e] = v
        scores.append(per)
    return L, E, scores


def build_prefetch_plan(scores: list[dict[int, float]],
                        row_map, ram: int, *,
                        start_layer: int, depth: int, top_k: int,
                        min_frac: float = 0.0,
                        resident=None) -> dict[int, list[int]]:
    """Roll the learned distribution into per-layer prefetch candidates.

    ``scores`` / ``row_map`` are per-layer structures in the LOCAL expert
    namespace (``row_map[layer][e] < ram`` means e is RAM-pinned on that
    layer -- those rows are served from the host banks and must not be
    prefetched). ``resident`` (optional) is a per-layer SEQUENCE of iterables
    of slot-resident local ids (warm chunks shrink the plan for free).
    Returns ``{target_layer: [expert, ...]}`` for layers in
    ``(start_layer, start_layer + depth]``, each list holding at most
    ``top_k`` ids sorted by descending score (ties by id), keeping only
    scores >= ``min_frac`` x the layer's best. Layers with no candidate are
    omitted. Pure planning: the caller owns slab budgets and dedup.
    """
    if depth < 1 or top_k < 1:
        return {}
    plan: dict[int, list[int]] = {}
    for target in range(start_layer + 1, start_layer + 1 + depth):
        if target >= len(scores):
            break
        per = scores[target]
        if not per:
            continue
        rm = row_map[target]
        res = (set(resident[target])
               if resident is not None and target < len(resident) else ())
        floor = min_frac * max(per.values()) if min_frac > 0 else 0.0
        cands = [
            (e, s) for e, s in per.items()
            if s >= floor and int(rm[e]) >= ram and e not in res
        ]
        if not cands:
            continue
        cands.sort(key=lambda t: (-t[1], t[0]))
        plan[target] = [e for e, _s in cands[:top_k]]
    return plan


def form_blocks(experts: list[int], block_size: int) -> list[list[int]]:
    """Chunk one layer's candidates into read blocks of up to ``block_size``
    experts, sorted by id first so file-adjacent experts land in the same
    block (their row extents merge into single preadv calls downstream;
    non-adjacent layouts simply don't merge, same rule as ``_row_groups``).
    ``block_size`` <= 1 means one block per expert."""
    ids = sorted(int(e) for e in experts)
    if block_size <= 1:
        return [[e] for e in ids]
    return [ids[i:i + block_size] for i in range(0, len(ids), block_size)]
