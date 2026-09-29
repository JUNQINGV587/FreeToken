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
import threading
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
        graph_rows: int = 2048,
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
        self.graph_rows = int(graph_rows)
        # A RowStore keeps its pending batch in member state, so the engine thread (eager prefill)
        # and the doorbell's service thread (a replayed graph) must not stage into one at once.
        self._io_lock = threading.Lock()
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
        # The CUDA-graph path (see engram_fetch): a captured gather cannot D2H its row ids, so it
        # hand-shakes with a host service thread over a doorbell instead. Off unless there is a
        # device to capture on.
        self._graph_bridge = None
        if self.device.type == "cuda" and os.getenv("FREETOKEN_ENGRAM_FETCH", "1") != "0":
            from .engram_fetch import EngramGraphFetch

            self._graph_bridge = EngramGraphFetch(self, k_max=self.graph_rows)

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
        if n == 0:
            return torch.zeros(*shape, self.dim, dtype=torch.bfloat16, device=self.device)
        if (
            self._graph_bridge is not None
            and self.device.type == "cuda"
            and torch.cuda.is_current_stream_capturing()
        ):
            return self._gather_graphed(layer_index, shape, flat, n)
        if self.device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            # No bridge (FREETOKEN_ENGRAM_FETCH=0, or a non-CUDA device) while capturing: the
            # eager round trip below starts with a D2H, which CUDA refuses inside capture. Say
            # so here instead of letting the driver raise mid-graph.
            raise RuntimeError(
                "engram gather inside CUDA graph capture needs the doorbell bridge "
                "(models/deepseek_v41/engram_fetch.py), which FREETOKEN_ENGRAM_FETCH=0 "
                "disables; relaunch with --cuda-graph-max-bs 0 or re-enable the bridge"
            )
        # Chunked: one forward can hash more rows than the staging buffer holds (a 4096-token
        # prefill gathers 24576 of them), and the staging is what the round trip is sized by.
        out = torch.empty(n, self.dim, dtype=torch.bfloat16, device=self.device)
        for start in range(0, n, self.max_rows):
            stop = min(start + self.max_rows, n)
            out[start:stop] = self._gather_eager(loc, flat[start:stop])
        return out.view(*shape, self.dim)

    def _gather_eager(self, loc: EngramTableLocation, flat: torch.Tensor) -> torch.Tensor:
        """One synchronous round trip: D2H the row ids, preadv the rows, H2D, dequant."""
        n = int(flat.numel())
        local = flat.detach().to(device="cpu", dtype=torch.int64).contiguous()
        out_of_range = (local < 0) | (local >= loc.rows)
        ids = local.masked_fill(out_of_range, 0)

        weight_store = self._weight_stores[loc.layer_id]
        scale_store = self._scale_stores[loc.layer_id]
        staged_w = self._pinned_w[: n * self.dim]
        staged_s = self._pinned_s[: n * self._scale_cols]
        with self._io_lock:
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
        return values

    def _gather_graphed(
        self, layer_index: int, shape: torch.Size, flat: torch.Tensor, n: int
    ) -> torch.Tensor:
        """A captured replay's gather: hand the ids to the doorbell and pull staged rows.

        Nothing here may touch the host: the ids stay on the device (they are hashed inside the
        graph from the tokens being replayed), the request block reaches the service thread
        through a captured D2H node, and the rows come back through captured H2D nodes gated by
        the spin kernel. Rows the request marks out of range are zeroed host-side, so unlike the
        eager path there is no mask to apply.
        """
        bridge = self._graph_bridge
        bridge.stage_gather(layer_index, flat.to(torch.int64).contiguous(), n)
        dev_w = bridge.out_w[: n * self.dim]
        dev_s = bridge.out_s[: n * self._scale_cols]
        values = dev_w.view(n, self.dim).view(torch.float8_e4m3fn).float()
        scales = dev_s.view(n, self._scale_cols).view(torch.float8_e8m0fnu).float()
        values = values.unflatten(-1, (-1, self.block)) * scales.unsqueeze(-1)
        values = values.flatten(-2).to(torch.bfloat16)
        return values.view(*shape, self.dim)

    def describe(self) -> str:
        backend = self._weight_stores[self.layer_ids[0]].io_backend()
        rows = ", ".join(f"layer {loc.layer_id}: {loc.rows} rows" for loc in self.locations)
        total = sum(loc.rows * (loc.dim + loc.scale_cols) for loc in self.locations)
        return f"engram tier ({backend}, {self.device}): {rows}; {total / 2**30:.1f} GiB on disk"

    def layer_index(self, layer_id: int) -> int:
        return self.layer_ids.index(layer_id)

    def view(self, layer_index: int) -> "EngramTable":
        """One layer's slice, in the interface ``Engram.forward`` calls."""
        return EngramTable(self, layer_index)


class EngramTable:
    """One engram layer's slice of an :class:`EngramTier`.

    ``Engram`` asks its table for ``gather(rows)`` -- the signature the resident table has -- while
    one tier serves every engram layer at once, so binding a tier hands each layer a view. ``to``
    returns the view (the tier moves underneath it), which is what ``Engram.bind`` expects.
    """

    def __init__(self, tier: EngramTier, layer_index: int):
        self.tier = tier
        self.layer_index = layer_index

    @property
    def layer_id(self) -> int:
        return self.tier.locations[self.layer_index].layer_id

    @property
    def num_embeddings(self) -> int:
        return self.tier.rows_of(self.layer_index)

    @property
    def device(self):
        return self.tier.device

    def gather(self, rows: torch.Tensor) -> torch.Tensor:
        return self.tier.gather(self.layer_index, rows)

    def to(self, device) -> "EngramTable":
        self.tier.to(device)
        return self

    def __repr__(self) -> str:
        return f"EngramTable(layer {self.layer_id}, {self.num_embeddings} rows)"


__all__ = ["EngramTable", "EngramTableLocation", "EngramTier", "locate_engram_tables"]
