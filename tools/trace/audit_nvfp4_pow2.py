#!/usr/bin/env python3
"""Full-model audit of the V4.1 NVFP4 per-32 power-of-two scale invariant.

The triton_dsfp4 conversion (moe/nvfp4_to_dsfp4.py) is lossless IFF every routed
expert scale tensor satisfies three properties:

1. every e4m3 scale byte is a positive exact power of two (mantissa zero);
2. every adjacent pair of 16-value groups (one 32-value MXFP4 group) shares the
   same scale byte -- i.e. the effective granularity is per-32, not per-16;
3. the per-projection fp32 global (weight_scale_2) is a power of two.

This tool reads the scale tensors of every routed expert (w1/w2/w3 per layer per
expert) straight from the safetensors shards and reports any violation. Run it
once per checkpoint before enabling --quant-backend moe.nvfp4=triton_dsfp4; a
clean bill means the load-time pack may convert with verify=False (the speed
path) and the disk tier's fetch-time repack is exact.

Exit code 0 = invariant holds everywhere (in the sampled scope). --layers and
--experts subsample for a quick check; the default is the FULL model.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))
from freetoken.moe.nvfp4_to_dsfp4 import E4M3_EXP_LUT, _INVALID  # noqa: E402

TENSORS_PER_EXPERT = ("w1", "w2", "w3")  # gate, down, up projections


def _shard_headers(model_path: str) -> dict[str, tuple[str, int, int, dict]]:
    """tensor name -> (shard path, data start, data end, dtype str)."""
    with open(os.path.join(model_path, "model.safetensors.index.json"), encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]
    headers: dict[str, tuple[str, int, int, tuple]] = {}
    per_shard: dict[str, list[str]] = {}
    for name, shard in weight_map.items():
        per_shard.setdefault(shard, []).append(name)
    for shard, names in per_shard.items():
        path = os.path.join(model_path, shard)
        with open(path, "rb") as f:
            (hlen,) = struct.unpack("<Q", f.read(8))
            meta = json.loads(f.read(hlen))
        base = 8 + hlen
        for name in names:
            info = meta[name]
            s, e = info["data_offsets"]
            headers[name] = (path, base + s, base + e, tuple(info["shape"]))
    return headers


def _read_range(path: str, start: int, end: int, fds: dict[str, int]) -> bytes:
    fd = fds.get(path)
    if fd is None:
        fd = os.open(path, os.O_RDONLY)
        fds[path] = fd
    return os.pread(fd, end - start, start)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model_path")
    ap.add_argument("--layers", type=int, default=40)
    ap.add_argument("--experts", type=int, default=384)
    ap.add_argument("--layer-ids", type=str, default="",
                    help="comma list; default all --layers")
    ap.add_argument("--expert-ids", type=str, default="",
                    help="comma list; default all --experts")
    ap.add_argument("--name-pattern", type=str,
                    default="layers.{layer}.ffn.experts.{expert}.{proj}.weight_scale",
                    help="scale tensor name template ({layer} {expert} {proj}=w1/w2/w3)")
    ap.add_argument("--global-pattern", type=str,
                    default="layers.{layer}.ffn.experts.{expert}.{proj}.weight_scale_2")
    args = ap.parse_args()

    layer_ids = ([int(x) for x in args.layer_ids.split(",") if x != ""]
                 or list(range(args.layers)))
    expert_ids = ([int(x) for x in args.expert_ids.split(",") if x != ""]
                  or list(range(args.experts)))

    t0 = time.time()
    headers = _shard_headers(args.model_path)
    print(f"[audit] index: {len(headers)} tensors, {time.time() - t0:.1f}s", flush=True)

    fds: dict[str, int] = {}
    bad_scale_bytes = bad_pairs = bad_globals = missing = 0
    checked = scale_bytes = 0
    t0 = time.time()
    try:
        for layer in layer_ids:
            for expert in expert_ids:
                for proj in TENSORS_PER_EXPERT:
                    sname = args.name_pattern.format(layer=layer, expert=expert, proj=proj)
                    gname = args.global_pattern.format(layer=layer, expert=expert, proj=proj)
                    ent = headers.get(sname)
                    gent = headers.get(gname)
                    if ent is None or gent is None:
                        missing += 1
                        continue
                    path, s, e, shape = ent
                    if len(shape) != 2 or shape[1] % 2:
                        print(f"[audit] SKIP {sname}: unexpected scale shape {shape}", flush=True)
                        missing += 1
                        continue
                    raw = np.frombuffer(_read_range(path, s, e, fds), dtype=np.uint8)
                    scale_bytes += raw.size
                    pairs = raw.reshape(shape[0], shape[1] // 2, 2)
                    # (1) every byte a positive pow2
                    exps = E4M3_EXP_LUT[raw]
                    n_bad = int((exps == _INVALID).sum())
                    if n_bad:
                        bad_scale_bytes += n_bad
                        if bad_scale_bytes <= 20:
                            idx = int(np.nonzero(exps == _INVALID)[0][0])
                            print(f"[audit] NON-POW2 scale byte 0x{raw[idx]:02x} at {sname} flat {idx}",
                                  flush=True)
                    # (2) adjacent 16-groups identical (pair = two adjacent bytes)
                    mm = pairs[:, :, 0] != pairs[:, :, 1]
                    mism = int(mm.sum())
                    if mism:
                        bad_pairs += mism
                        if bad_pairs <= 20:
                            r, c = [int(v) for v in np.nonzero(mm)]
                            r, c = r[0], c[0]
                            print(f"[audit] PAIR MISMATCH at {sname} row {r} pair {c}: "
                                  f"0x{pairs[r, c, 0]:02x} vs 0x{pairs[r, c, 1]:02x}", flush=True)
                    # (3) global is a power of two
                    _gp, gs, ge, _gdt = gent
                    (gv,) = struct.unpack("<f", _read_range(_gp, gs, ge, fds))
                    if not (gv > 0.0) or abs(math.log2(gv) - round(math.log2(gv))) > 1e-6:
                        bad_globals += 1
                        print(f"[audit] NON-POW2 global {gv!r} at {gname}", flush=True)
                    checked += 1
            done = (layer_ids.index(layer) + 1) / len(layer_ids)
            print(f"[audit] layer {layer}: checked={checked} "
                  f"({scale_bytes / 2**30:.2f} GiB scales) "
                  f"bad_bytes={bad_scale_bytes} bad_pairs={bad_pairs} "
                  f"bad_globals={bad_globals} missing={missing} "
                  f"[{time.time() - t0:.0f}s, {done:.0%}]", flush=True)
    finally:
        for fd in fds.values():
            os.close(fd)

    ok = not (bad_scale_bytes or bad_pairs or bad_globals or missing)
    print(f"[audit] {'PASS' if ok else 'FAIL'}: {checked} tensors, "
          f"non-pow2 bytes={bad_scale_bytes}, pair mismatches={bad_pairs}, "
          f"non-pow2 globals={bad_globals}, missing={missing}, "
          f"{time.time() - t0:.0f}s", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
