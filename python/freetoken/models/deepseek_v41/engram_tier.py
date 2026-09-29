"""The engram tables, served off NVMe.

Each engram layer ships a 94.37 GiB lookup table -- 384M rows of 256 fp8-e4m3 bytes plus an 8-byte
e8m0 block scale, 264 B a row -- so 188.7 GiB for the two of them. That is more than the host has
RAM and far more than the two L20s have VRAM, so the tables cannot be resident. Nothing about the
access pattern wants them to be: a decode step gathers one token's n-grams per layer, i.e. at most
``(max_ngram - 1) * n_heads`` = 24 rows (6 KB), and a 4096-token prefill gathers 24576 rows
(6.5 MB).

So a lookup reads its rows straight out of the checkpoint's own safetensors shards -- no repack, no
second copy on disk -- through ``kernel.row_store`` (io_uring or a pread pool, O_DIRECT where the
filesystem allows it), stages them into pinned host memory and copies them to the device
asynchronously; the block-scale dequant then runs on the GPU exactly as the reference's
``ParallelEngramEmbedding`` does it. Hashing is *not* done here: ``NgramHashState`` already turned
the n-gram into row ids, so this is a gather.

``RowStore`` has one ``rows_per_extent`` for all of its extents, and the two layers do not have the
same row count (384006168 vs 384016682), so each (layer, tensor) pair gets its own store and its
row ids are the layer's own.
"""

from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass
from typing import Sequence

import torch

_WEIGHT_SUFFIX = ".engram.embed.weight"
_SCALE_SUFFIX = ".engram.embed.scale"
_WEIGHT_DTYPE = "F8_E4M3"
_SCALE_DTYPE = "F8_E8M0"
_IO_URING_ENV = "FREETOKEN_ENGRAM_IO_URING"


def _safetensors_header(path: str) -> tuple[dict, int]:
    """The shard's JSON header plus the byte offset its data section starts at."""
    with open(path, "rb") as fh:
        (n,) = struct.unpack("<Q", fh.read(8))
        return json.loads(fh.read(n)), 8 + n


@dataclass(frozen=True)
class EngramTableLocation:
    """Where one engram layer's table lives: two tensors, as absolute byte ranges in two shards."""

    layer_id: int
    rows: int
    dim: int
    block: int
    weight_path: str
    weight_base: int
    scale_path: str
    scale_base: int

    @property
    def scale_cols(self) -> int:
        return self.dim // self.block


def locate_engram_tables(model_dir: str, layer_ids: Sequence[int]) -> list[EngramTableLocation]:
    """Resolve the engram tables' row geometry and byte offsets from the checkpoint index.

    Header-only: a few KiB per shard, so this stays cheap even though the tensors behind it are 94
    GiB each. Raises rather than guessing if a table is missing or shaped unlike the reference.
    """
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    with open(index_path, encoding="utf-8") as fh:
        weight_map = json.load(fh)["weight_map"]
    headers: dict[str, tuple[dict, int]] = {}
    located: list[EngramTableLocation] = []
    for layer_id in layer_ids:
        names = {
            "weight": f"layers.{layer_id}{_WEIGHT_SUFFIX}",
            "scale": f"layers.{layer_id}{_SCALE_SUFFIX}",
        }
        for what, name in names.items():
            if name not in weight_map:
                raise KeyError(f"{name} is not in {index_path}")
        weight_path = os.path.join(model_dir, weight_map[names["weight"]])
        scale_path = os.path.join(model_dir, weight_map[names["scale"]])
        for path in (weight_path, scale_path):
            if path not in headers:
                headers[path] = _safetensors_header(path)
        weight_meta = headers[weight_path][0][names["weight"]]
        scale_meta = headers[scale_path][0][names["scale"]]
        if weight_meta["dtype"] != _WEIGHT_DTYPE:
            raise ValueError(f"{names['weight']} has dtype {weight_meta['dtype']}, expected {_WEIGHT_DTYPE}")
        if scale_meta["dtype"] != _SCALE_DTYPE:
            raise ValueError(f"{names['scale']} has dtype {scale_meta['dtype']}, expected {_SCALE_DTYPE}")
        rows, dim = (int(x) for x in weight_meta["shape"])
        scale_rows, scale_cols = (int(x) for x in scale_meta["shape"])
        if scale_rows != rows:
            raise ValueError(f"{names['scale']} has {scale_rows} rows but {names['weight']} has {rows}")
        if scale_cols <= 0 or scale_cols * 32 != dim:
            raise ValueError(
                f"{names['weight']} is {dim} wide but {names['scale']} has {scale_cols} columns; "
                "the reference uses one e8m0 scale per 32 elements"
            )
        located.append(
            EngramTableLocation(
                layer_id=int(layer_id),
                rows=rows,
                dim=dim,
                block=dim // scale_cols,
                weight_path=weight_path,
                weight_base=headers[weight_path][1] + int(weight_meta["data_offsets"][0]),
                scale_path=scale_path,
                scale_base=headers[scale_path][1] + int(scale_meta["data_offsets"][0]),
            )
        )
    if not located:
        raise ValueError("no engram layers requested")
    dims = {loc.dim for loc in located}
    blocks = {loc.block for loc in located}
    if len(dims) != 1 or len(blocks) != 1:
        raise ValueError(f"engram layers disagree on geometry: dims={dims} blocks={blocks}")
    return located


class EngramTier:
    """``Engram.table``: gather rows of the 94 GiB tables off NVMe, dequantized on the device.

    ``gather(rows)`` matches ``ResidentEngramTable.gather`` -- bf16 ``[*rows.shape, head_dim]``,
    out-of-range rows read as 0 -- so the layer does not know which of the two it holds.
    """

    def __init__(
        self,
        model_dir: str,
        layer_ids: Sequence[int],
        *,
        device=None,
        max_rows: int = 16384,
        use_io_uring: bool | None = None,
    ) -> None:
        from freetoken.kernel.row_store import RowStore

        self.locations = locate_engram_tables(model_dir, layer_ids)
        self.layer_ids = [loc.layer_id for loc in self.locations]
        self._by_layer = {loc.layer_id: loc for loc in self.locations}
        self.dim = self.locations[0].dim
        self.block = self.locations[0].block
        self.head_dim = self.dim
        self.num_layers = len(self.locations)
        self.max_rows = int(max_rows)
        if use_io_uring is None:
            use_io_uring = os.getenv(_IO_URING_ENV, "1") != "0"
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._weight_stores: dict[int, "RowStore"] = {}
        self._scale_stores: dict[int, "RowStore"] = {}
        for loc in self.locations:
            self._weight_stores[loc.layer_id] = RowStore(
                [loc.weight_path], [0], [loc.weight_base], loc.rows, loc.dim, loc.dim, use_io_uring
            )
            self._scale_stores[loc.layer_id] = RowStore(
                [loc.scale_path], [0], [loc.scale_base], loc.rows, loc.scale_cols, loc.scale_cols, use_io_uring
            )
        self._allocate()

    # ------------------------------------------------------------------ staging

    def _pinned(self, nbytes: int) -> torch.Tensor:
        """Pinned (device-mapped) host memory when the extension is there; plain host memory otherwise."""
        if self._pinned_ok:
            from freetoken.kernel.pinned import alloc_pinned_tensor

            return alloc_pinned_tensor(nbytes, dtype=torch.uint8)
        return torch.empty(nbytes, dtype=torch.uint8)

    def _allocate(self) -> None:
        try:
            import torch.cuda

            self._pinned_ok = torch.cuda.is_available()
        except Exception:  # pragma: no cover - a CPU-only build
            self._pinned_ok = False
        scale_cols = self.dim // self.block
        self._pinned_w = self._pinned(self.max_rows * self.dim)
        self._pinned_s = self._pinned(self.max_rows * scale_cols)
        self._dev_w = torch.empty(self.max_rows * self.dim, dtype=torch.uint8, device=self.device)
        self._dev_s = torch.empty(self.max_rows * scale_cols, dtype=torch.uint8, device=self.device)
        self._scale_cols = scale_cols

    def to(self, device) -> "EngramTier":
        device = torch.device(device)
        if device != self.device:
            self.device = device
            self._dev_w = self._dev_w.to(device)
            self._dev_s = self._dev_s.to(device)
        return self

    # ------------------------------------------------------------------ gather

    @property
    def num_embeddings(self) -> int:
        return self.locations[0].rows

    def rows_of(self, layer_index: int) -> int:
        return self.locations[layer_index].rows

    @torch.inference_mode()
    def gather(self, layer_index: int, rows: torch.Tensor) -> torch.Tensor:
        loc = self.locations[layer_index]
        shape = rows.shape
        flat = rows.reshape(-1)
        n = int(flat.numel())
        if n > self.max_rows:
            raise ValueError(
                f"engram gather of {n} rows exceeds the {self.max_rows}-row staging buffer; "
                "raise EngramTier(max_rows=...) or gather in chunks"
            )
        if n == 0:
            return torch.zeros(*shape, self.dim, dtype=torch.bfloat16, device=self.device)
        local = flat.detach().to(device="cpu", dtype=torch.int64).contiguous()
        out_of_range = (local < 0) | (local >= loc.rows)
        ids = local.masked_fill(out_of_range, 0)

        weight_store = self._weight_stores[loc.layer_id]
        scale_store = self._scale_stores[loc.layer_id]
        staged_w = self._pinned_w[: n * self.dim]
        staged_s = self._pinned_s[: n * self._scale_cols]
        weight_store.stage_rows(ids.data_ptr(), n, staged_w.data_ptr(), 0)
        scale_store.stage_rows(ids.data_ptr(), n, staged_s.data_ptr(), 0)
        # one batched round trip for both tensors; the copies below then see finished host memory
        weight_store.flush(0)
        scale_store.flush(0)
        dev_w = self._dev_w[: n * self.dim].copy_(staged_w, non_blocking=True)
        dev_s = self._dev_s[: n * self._scale_cols].copy_(staged_s, non_blocking=True)

        values = dev_w.view(n, self.dim).view(torch.float8_e4m3fn).float()
        scales = dev_s.view(n, self._scale_cols).view(torch.float8_e8m0fnu).float()
        values = values.unflatten(-1, (-1, self.block)) * scales.unsqueeze(-1)
        values = values.flatten(-2).to(torch.bfloat16)
        values = values.masked_fill(out_of_range.to(self.device).unsqueeze(-1), 0)
        return values.view(*shape, self.dim)

    def describe(self) -> str:
        backend = self._weight_stores[self.layer_ids[0]].io_backend()
        rows = ", ".join(f"layer {loc.layer_id}: {loc.rows} rows" for loc in self.locations)
        total = sum(loc.rows * (loc.dim + loc.scale_cols) for loc in self.locations)
        return f"engram tier ({backend}, {self.device}): {rows}; {total / 2**30:.1f} GiB on disk"


__all__ = ["EngramTableLocation", "EngramTier", "locate_engram_tables"]
