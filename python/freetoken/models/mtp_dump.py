"""Debug dump of DSpark-MTP replay inputs, gated by FREETOKEN_MTP_DUMP_DIR (default OFF).

Purely additive diagnostics for the MTP acceptance experiment
(/data/research/runs/20261004-mtp-accept). When the env var is unset the module is
inert: `enabled()` is False and every entry point returns immediately, so production
behaviour is bit-identical to before the patch.

What it captures, per request (TP primary rank only, never during CUDA graph capture):

  * PREFILL: for each segment, the hc-mean hidden states at target layers 37/38/39
    (the attention INPUT, matching `Transformer.forward` in the authors'
    inference/model.py:1265-1268) for the last <=128 prompt positions, the segment's
    token ids, and the prompt's top-k next-token logits row.
  * DECODE (every step): the same hidden-state triple at the current position, the
    fed token id, the absolute position, and the main model's top-k logits row
    (greedy reference for the next position).

Layout: $FREETOKEN_MTP_DUMP_DIR/req_<nnnn>/{meta.json, prefill_*.npz, steps/step_*.npz}
The offline replay (replay/mtp_replay.py) consumes this layout verbatim.

Hidden states are stored raw (bf16 as uint16, or fp32) -- exactly what the engine
computed, no rerounding.
"""
from __future__ import annotations

import json
import os
import threading

import numpy as np
import torch

TARGET_LAYERS = (37, 38, 39)
_WINDOW = 128
_TOPK_DEFAULT = 8

_lock = threading.Lock()
_pending_h: dict = {}        # (table_row, layer_idx) -> hidden [dim]
_pending_pos: dict = {}      # table_row -> absolute position of the pending step
_prefill_parts: dict = {}    # (bsz, seg_idx) -> running tail [<=128, 3*dim]
_prefill_meta: dict = {}     # (bsz, seg_idx) -> {req_id, token_ids, start_pos, topk_ids, topk_vals}
_reqs: dict = {}             # req_id -> {"table_rows": set, "steps": int, "seg_keys": set}
_counter = [0]


def enabled() -> bool:
    return bool(os.environ.get("FREETOKEN_MTP_DUMP_DIR"))


def _topk() -> int:
    return int(os.environ.get("FREETOKEN_MTP_DUMP_TOPK", str(_TOPK_DEFAULT)))


def _out_dir() -> str:
    return os.environ["FREETOKEN_MTP_DUMP_DIR"]


def _is_primary() -> bool:
    try:
        from freetoken.distributed import try_get_tp_info

        info = try_get_tp_info()
        return info is None or info.is_primary()
    except Exception:
        return True


def _capturing() -> bool:
    try:
        return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
    except Exception:
        return False


def capture(h: torch.Tensor, layer_idx: int, start_pos, keep_rows) -> None:
    """Layer-loop hook: stash h.mean(dim=2) at target layers (author's attention input).

    h: [B, S, hc_mult, dim] (paged path) -- identical math for the eager path.
    """
    if _capturing() or not _is_primary():
        return
    from freetoken.core import get_global_ctx

    md = get_global_ctx().batch.attn_metadata
    segments = getattr(md, "segments", None) if md is not None else None
    with _lock:
        if segments:
            # PREFILL: keep the last <=_WINDOW positions of every segment, per target
            # layer (keyed by layer so chunks extend the token axis and layers extend
            # the feature axis; dump_prefill concatenates the layers at the end).
            # A segment starting at sp==0 means a NEW prompt is prefilling on this
            # slot: drop any leftover parts from the slot's previous request first
            # (otherwise the old request's parts would cat into / shadow the new one;
            # the slot-reuse bug this fixes made W17 lose every prefill npz after the
            # first request). Capture fires once per target layer, so clear only at
            # the FIRST target layer -- a clear on every call would wipe the sibling
            # layers captured moments ago in the same prefill.
            if layer_idx == TARGET_LAYERS[0] and any(int(sp) == 0 for (_o, _n, _t, sp) in segments):
                for k in [k for k in _prefill_parts if k[0] == 0]:
                    _prefill_parts.pop(k, None)
            for seg_i, (off, n, _ti, _sp) in enumerate(segments):
                off, n = int(off), int(n)
                tail = min(n, _WINDOW)
                part = h[0, off + n - tail : off + n].mean(dim=1).detach()  # [tail, dim]
                key = (0, seg_i, layer_idx)
                prev = _prefill_parts.get(key)
                cat = part if prev is None else torch.cat([prev, part], dim=0)
                _prefill_parts[key] = cat[-_WINDOW:]
        else:
            # DECODE: one row per request, keyed by the request's stable table row.
            rows = getattr(md, "table_rows", None) if md is not None else None
            if rows is None:
                rows = torch.arange(h.shape[0], device=h.device)
            hid = h.mean(dim=2)[:, 0, :]  # [B, dim]
            for j, r in enumerate(rows.tolist()):
                _pending_h[(int(r), layer_idx)] = hid[j]


def _stack_and_clear(row: int) -> torch.Tensor | None:
    try:
        triple = [_pending_h.pop((row, li)) for li in TARGET_LAYERS]
    except KeyError:
        return None
    return torch.cat(triple, dim=-1)  # [3*dim]


def _store_hidden(t: torch.Tensor, z: dict, name: str) -> None:
    t = t.detach().cpu()
    if t.dtype == torch.bfloat16:
        z[name] = t.view(torch.uint16).numpy()
        z[name + "_dtype"] = "bfloat16"
    elif t.dtype == torch.float16:
        z[name] = t.view(torch.uint16).numpy()
        z[name + "_dtype"] = "float16"
    else:
        z[name] = t.float().numpy()
        z[name + "_dtype"] = "float32"


def _logits_topk(logits: torch.Tensor):
    """logits: [..., V] -> (ids int32 [K], vals fp32 [K]) of the last position."""
    lg = logits.detach()
    if lg.dim() == 3:
        lg = lg[:, -1, :]
    vals, ids = lg.float().topk(min(_topk(), lg.shape[-1]), dim=-1)
    return ids.cpu().numpy().astype(np.int32), vals.cpu().numpy().astype(np.float32)


def _req_for_segment(bsz: int, seg_i: int, sp: int) -> dict:
    key = (bsz, seg_i)
    meta = _prefill_meta.get(key)
    if meta is not None and _reqs.get(meta["req_id"], {}).get("steps", 0) > 0:
        # The slot's previous request already decoded: this prefill is a NEW request.
        _write_meta(meta["req_id"])
        _prefill_meta.pop(key, None)
        meta = None
    if meta is None:
        _counter[0] += 1
        req_id = _counter[0]
        meta = {"req_id": req_id, "cached_len": sp, "token_ids": []}
        _prefill_meta[key] = meta
        _reqs[req_id] = {"table_rows": set(), "steps": 0, "seg_keys": {key}}
    return meta


def dump_prefill(batch, segments, logits: torch.Tensor) -> None:
    """Called once per prefill forward (after logits). `logits`: [n_segments, V]."""
    if not enabled() or _capturing() or not _is_primary() or segments is None:
        return
    with _lock:
        _flush_completed()
        token_ids = batch.input_ids.long().view(-1)
        for seg_i, (off, n, _ti, sp) in enumerate(segments):
            off, n, sp = int(off), int(n), int(sp)
            meta = _req_for_segment(0, seg_i, sp)
            req_id = meta["req_id"]
            parts = [_prefill_parts.pop((0, seg_i, li), None) for li in TARGET_LAYERS]
            tail = None
            if all(p is not None for p in parts):
                tail = torch.cat(parts, dim=-1)  # [rows, 3*dim]
            ids = token_ids[off : off + n].cpu().numpy().astype(np.int32)
            meta["token_ids"] = ids  # this chunk's tokens (single-chunk for <=40k prompts)
            meta["prompt_len"] = sp + n
            tk_ids, tk_vals = _logits_topk(logits[seg_i : seg_i + 1])
            meta["topk"] = (tk_ids[0], tk_vals[0])
            meta["last_start_pos"] = sp
            if tail is not None:
                meta["pending_tail"] = (tail, sp + n - tail.shape[0])
        # Persist what we have so far for every prefill we just saw (chunked prompts
        # write one prefill_*.npz per chunk; replay concatenates them).
        for key, meta in _prefill_meta.items():
            tail_info = meta.pop("pending_tail", None)
            if tail_info is None:
                continue
            tail, tail_start = tail_info
            req_dir = _req_dir(meta["req_id"])
            os.makedirs(req_dir, exist_ok=True)
            z: dict = {
                "token_ids": meta["token_ids"],
                "start_pos": meta["last_start_pos"],
                "tail_start": tail_start,
            }
            _store_hidden(tail, z, "tail")
            tk_ids, tk_vals = meta["topk"]
            z["topk_ids"] = tk_ids
            z["topk_vals"] = tk_vals
            np.savez(os.path.join(req_dir, f"prefill_{meta['last_start_pos']:09d}.npz"), **z)
            _write_meta(meta["req_id"])


def dump_step(batch, md, pos: torch.Tensor, input_ids: torch.Tensor, logits: torch.Tensor) -> None:
    """Called once per decode forward (after logits). Batched; bs=1 for the battery."""
    if not enabled() or _capturing() or not _is_primary():
        return
    rows = getattr(md, "table_rows", None)
    if rows is None:
        return
    tk_ids, tk_vals = _logits_topk(logits)
    pos_list = pos.tolist()
    fed_list = input_ids.view(-1).tolist()
    with _lock:
        for j, r in enumerate(rows.tolist()):
            triple = _stack_and_clear(int(r))
            if triple is None:
                continue
            _pending_pos[int(r)] = pos_list[j]
            req_id = _req_for_row(int(r))
            if req_id is None:
                continue
            req = _reqs[req_id]
            req["steps"] += 1
            req_dir = _req_dir(req_id)
            os.makedirs(os.path.join(req_dir, "steps"), exist_ok=True)
            z: dict = {
                "pos": pos_list[j],
                "fed": fed_list[j],
                "topk_ids": tk_ids[j],
                "topk_vals": tk_vals[j],
            }
            _store_hidden(triple, z, "mh")
            np.savez(os.path.join(req_dir, "steps", f"step_{pos_list[j]:09d}.npz"), **z)
            req["table_rows"].add(int(r))


def _req_for_row(row: int) -> int | None:
    """Attribute a decode table row to a request: the most recent request that has not
    yet seen this row (bs=1 sequential battery: unambiguous)."""
    candidates = [
        rid
        for rid, r in _reqs.items()
        if row not in r["table_rows"] and _prefill_meta_has(rid)
    ]
    if not candidates:
        # row reused across requests: fall back to the latest request overall
        return max(_reqs) if _reqs else None
    return max(candidates)


def _prefill_meta_has(req_id: int) -> bool:
    return any(m["req_id"] == req_id for m in _prefill_meta.values())


def _req_dir(req_id: int) -> str:
    return os.path.join(_out_dir(), f"req_{req_id:04d}")


def _write_meta(req_id: int) -> None:
    req = _reqs.get(req_id, {})
    metas = [m for m in _prefill_meta.values() if m["req_id"] == req_id]
    prompt_len = max((m.get("prompt_len", 0) for m in metas), default=0)
    cached_len = min((m.get("cached_len", 0) for m in metas), default=0)
    doc = {
        "req_id": req_id,
        "prompt_len": prompt_len,
        "cached_len": cached_len,
        "table_rows": sorted(req.get("table_rows", [])),
        "decode_steps": req.get("steps", 0),
    }
    req_dir = _req_dir(req_id)
    os.makedirs(req_dir, exist_ok=True)
    with open(os.path.join(req_dir, "meta.json"), "w") as f:
        json.dump(doc, f, indent=2)


def _flush_completed() -> None:
    """Flush step-count metadata for requests that already produced decode steps; the
    steps themselves are written eagerly in dump_step. Prefill parts are NOT cleared
    here: they belong to the request currently prefilling on a slot and are dropped
    by capture() itself when that slot's NEXT prompt starts (sp==0); clearing them
    here would wipe the parts captured for the prefill being dumped right now.
    """
    for req_id, req in _reqs.items():
        if req["steps"]:
            _write_meta(req_id)


def flush_all() -> None:
    """Best-effort flush (call at shutdown if convenient; safe to omit)."""
    if not enabled():
        return
    with _lock:
        for req_id in _reqs:
            _write_meta(req_id)
