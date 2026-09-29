"""CUDA-graph-compatible engram fetches ("engram doorbell").

The eidetic-engram gather is host driven: the row ids are hashed on the DEVICE (they come from the
token ids the graph is currently replaying), and the 94 GiB tables live on NVMe, so the rows must
come back through the host. ``EngramTier.gather`` does that with a device->host sync, which is
illegal inside a replayed graph -- so an engram layer and CUDA graphs have been mutually exclusive
(the boot ran with ``--cuda-graph-max-bs 0`` to get around it).

This module borrows the MoE disk tier's doorbell split (see :mod:`freetoken.moe.graph_fetch`) and
the same transport facts, each probed on this platform:

- device kernel STORES to host-pinned sysmem: LOST inside a replay (the ack therefore never comes
  from the device);
- in-graph cudaMemcpyAsync D2H node: WORKS (this is how the row ids reach the host);
- in-graph cudaMemcpyAsync H2D node: WORKS (``research/runs/v41boot/h2d_graph_probe.py`` -- the
  graph re-reads the pinned staging on every replay, which a plain thread-issued H2D cannot do);
- device kernel READS of host-pinned sysmem: WORK (the spin polls the ack over PCIe);
- thread-issued CUDA work while a replay spins: NEVER SCHEDULED (the service thread makes ZERO
  CUDA calls -- preadv and CPU stores only).

Per capture-time gather, the graph records:

1. ``_ef_request_kernel``: copy this gather's row ids and layer into the device request block and
   bump its sequence (LAST field, so a sequence the host sees implies the ids landed).
2. a D2H memcpy node of that block into its pinned mirror;
3. ``_gf_spin_kernel``: wait until the host has acked that sequence (no-op during capture, when
   the doorbell is off, and for an empty request);
4. two H2D memcpy nodes: the staged fp8 rows and their e8m0 block scales.

and the host service thread, on seeing a new sequence, preadv's those rows into pinned staging and
releases the sequence with a plain CPU store. Serialisation is the same as the MoE bridge's: a
single replay stream plus the spin means requests are strictly ordered, so one request block and
one staging set are enough.
"""

from __future__ import annotations

import os
import threading
import time

import torch
import triton
import triton.language as tl

from ...moe.graph_fetch import _gf_spin_kernel

__all__ = ["EngramGraphFetch"]


@triton.jit
def _ef_request_kernel(
    ids_ptr, n_ptr, req_ptr, BLOCK_K: tl.constexpr, LAYER: tl.constexpr, TRACE: tl.constexpr
):
    """Record one gather's row ids in the device request block.

    Layout (int64): [0] = count, [1] = layer, [2:2+BLOCK_K] = row ids (slots past ``count`` hold
    garbage the host never reads), [2+BLOCK_K] = sequence. The sequence is written LAST and only
    for a non-empty request, so the captured D2H memcpy cannot expose a new sequence with stale
    ids -- the block is copied linearly.
    """
    off = tl.arange(0, BLOCK_K)
    n = tl.load(n_ptr)
    ids = tl.load(ids_ptr + off, mask=off < n, other=0)
    tl.store(req_ptr + 2 + off, ids.to(tl.int64))
    tl.store(req_ptr, n.to(tl.int64))
    tl.store(req_ptr + 1, (LAYER + 0 * n).to(tl.int64))
    if TRACE:
        tl.device_print("[ef-req] L", LAYER)
        tl.device_print("[ef-req] n=", n)
    if n > 0:
        seq_ptr = req_ptr + 2 + BLOCK_K
        tl.store(seq_ptr, tl.load(seq_ptr, volatile=True) + 1)


class EngramGraphFetch:
    """Per-tier doorbell + pinned staging + host service thread for :class:`EngramTier`."""

    def __init__(self, tier, k_max: int) -> None:
        self.tier = tier
        self.k_max = int(k_max)
        if self.k_max <= 0 or self.k_max & (self.k_max - 1):
            raise ValueError(f"engram doorbell k_max must be a power of two, got {k_max}")
        self.block_k = self.k_max
        self.seq_off = 2 + self.block_k
        device = tier.device
        self._device = device
        dim = int(tier.dim)
        scale_cols = int(tier._scale_cols)
        self.dim = dim
        self.scale_cols = scale_cols

        # The engine holds torch.inference_mode() around model construction; tensors born there
        # are inference tensors, and enable()/disable() writing doorbell_dev would raise.
        with torch.inference_mode(False):
            n_req = 3 + self.block_k
            self.req_dev = torch.zeros(n_req, dtype=torch.int64, device=device)
            self.req_host = torch.zeros(n_req, dtype=torch.int64, pin_memory=True)
            self.n_dev = torch.zeros(1, dtype=torch.int64, device=device)
            # The ack word: written by the thread with a plain CPU store, polled by the spin
            # kernel over PCIe. No device copy of it exists.
            self.resp_host = torch.zeros(1, dtype=torch.int64, pin_memory=True)
            self.doorbell_dev = torch.zeros(1, dtype=torch.int64, device=device)
            # Staging rows for the graph path, separate from the eager path's own pinned buffers:
            # an eager gather on the engine thread runs host-side staging without stream ordering,
            # so sharing them with a replay's H2D nodes would be a data race, not a slowdown.
            self.staging_w = torch.zeros(self.k_max * dim, dtype=torch.uint8, pin_memory=True)
            self.staging_s = torch.zeros(self.k_max * scale_cols, dtype=torch.uint8, pin_memory=True)
            # Device targets of those two H2D nodes, bridge-private: the eager path's own
            # _dev_w/_dev_s are shared with it, and a graph must own every buffer it reads.
            self.out_w = torch.zeros(self.k_max * dim, dtype=torch.uint8, device=device)
            self.out_s = torch.zeros(self.k_max * scale_cols, dtype=torch.uint8, device=device)
            # Row ids for the service thread: copied out of the request mirror so the stage call
            # never reads memory a later D2H can overwrite, and masked to a legal row id.
            self.ids_host = torch.zeros(self.k_max, dtype=torch.int64, pin_memory=True)

        self._dbg = os.environ.get("FT_ENGRAM_FETCH_DEBUG", "") == "1"
        self._trace = os.environ.get("FT_ENGRAM_FETCH_TRACE", "") == "1"
        self._io_lock = tier._io_lock
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._served = 0
        self._numpy_view = None
        self._warm()

    # -- capture-time entry points (python runs only while capturing) --------

    def _warm(self) -> None:
        # triton JIT-compiles on first launch with a given constexpr set, and compiling during
        # capture is illegal: warm every layer's request kernel outside capture.
        if os.environ.get("FT_ENGRAM_FETCH_OFF"):
            return
        self.n_dev.fill_(0)
        for layer_index in range(self.tier.num_layers):
            _ef_request_kernel[(1,)](
                self.req_dev, self.n_dev, self.req_dev,
                BLOCK_K=self.block_k, LAYER=layer_index, TRACE=False,
            )
        _gf_spin_kernel[(1,)](
            self.resp_host, self.req_dev, self.doorbell_dev, SEQ_OFF=self.seq_off, TRACE=False
        )
        torch.cuda.synchronize(self._device)

    def stage_gather(self, layer_index: int, ids: torch.Tensor, n: int) -> None:
        """Record request + memcpy + spin + staged-row pull for one engram gather."""
        if n > self.k_max:
            raise RuntimeError(
                f"engram doorbell: a graph that gathers {n} rows exceeds the {self.k_max}-row "
                "request block; raise EngramTier(graph_rows=...) to at least the rows one captured "
                "forward touches"
            )
        self.n_dev.fill_(n)
        _ef_request_kernel[(1,)](
            ids, self.n_dev, self.req_dev,
            BLOCK_K=self.block_k, LAYER=layer_index, TRACE=self._trace,
        )
        # Captured memcpy node: DMA the request block to the host-polled mirror (kernel stores to
        # sysmem are lost from a replay; DMA writes are visible by construction).
        self.req_host.copy_(self.req_dev, non_blocking=True)
        _gf_spin_kernel[(1,)](
            self.resp_host, self.req_dev, self.doorbell_dev, SEQ_OFF=self.seq_off,
            TRACE=self._trace,
        )
        # Captured H2D nodes: re-read the pinned staging every replay, after the spin.
        self.out_w[: n * self.dim].copy_(self.staging_w[: n * self.dim], non_blocking=True)
        self.out_s[: n * self.scale_cols].copy_(
            self.staging_s[: n * self.scale_cols], non_blocking=True
        )

    # -- doorbell lifecycle (host side) ---------------------------------------

    def disable(self) -> None:
        """Called before (re)capture: spin kernels become no-ops."""
        self.doorbell_dev.fill_(0)

    def enable(self) -> None:
        """Called after capture finished: start servicing replay requests."""
        if self._thread is None:
            # Snapshot: ignore every sequence written during capture and warm-up.
            self._served = int(self.req_host[self.seq_off])
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._serve, name="engram-fetch", daemon=True
            )
            self._thread.start()
        self.doorbell_dev.fill_(1)
        print(f"[engram-fetch] enabled dev={self._device} pid={os.getpid()} "
              f"served={self._served} k_max={self.k_max}", flush=True)

    def shutdown(self) -> None:
        self._stop.set()
        t = self._thread
        if t is not None:
            t.join(timeout=5)
            self._thread = None

    # -- host service thread ---------------------------------------------------
    # Zero CUDA calls: anything enqueued from here was empirically never scheduled while a graph
    # replay spun on the device. preadv + CPU stores only.

    def _view(self):
        if self._numpy_view is None:
            self._numpy_view = self.req_host.numpy()
        return self._numpy_view

    def _serve(self) -> None:
        print(f"[engram-fetch] thread-entry dev={self._device} pid={os.getpid()} "
              f"tid={threading.get_native_id()}", flush=True)
        try:
            self._serve_inner()
        except Exception:
            import traceback

            print(f"[engram-fetch] THREAD CRASHED dev={self._device}\n"
                  + traceback.format_exc(), flush=True)
            raise

    def _serve_inner(self) -> None:
        blk = self._view()
        seq_off = self.seq_off
        served = self._served
        polls = 0
        while not self._stop.is_set():
            seq = int(blk[seq_off])
            if seq == served:
                polls += 1
                if self._dbg and polls % 1000000 == 0:
                    print(f"[engram-fetch] alive dev={self._device}: served={served} "
                          f"req_seq={seq} resp={int(self.resp_host[0])}", flush=True)
                time.sleep(0)  # yield the GIL between polls (~µs cadence)
                continue
            count = int(blk[0])
            layer_index = int(blk[1])
            if count > self.k_max or not 0 <= layer_index < self.tier.num_layers:
                print(f"[engram-fetch] FATAL: request count {count} / layer {layer_index} out of "
                      f"range; refusing to serve (the server will hang and must be restarted)",
                      flush=True)
                return
            self._stage(layer_index, count)
            if self._dbg:
                print(f"[engram-fetch] served dev={self._device} seq={seq} layer={layer_index} "
                      f"count={count}", flush=True)
            # Ack: plain CPU store AFTER all staging writes (x86 TSO orders store->store; the
            # spin's sys-scope acquire pairs with it). No CUDA op from this thread.
            self.resp_host[0] = seq
            served = seq

    def _stage(self, layer_index: int, count: int) -> None:
        """preadv ``count`` rows of one layer into pinned staging (row i at staging row i)."""
        loc = self.tier.locations[layer_index]
        ids = self.ids_host[:count]
        ids.copy_(self.req_host[2 : 2 + count])
        bad = (ids < 0) | (ids >= loc.rows)
        ids.masked_fill_(bad, 0)
        with self._io_lock:
            weight_store = self.tier._weight_stores[loc.layer_id]
            scale_store = self.tier._scale_stores[loc.layer_id]
            weight_store.stage_rows(ids.data_ptr(), count, self.staging_w.data_ptr(), 0)
            scale_store.stage_rows(ids.data_ptr(), count, self.staging_s.data_ptr(), 0)
            # one batched round trip for both tensors; the copies below then see finished host memory
            weight_store.flush(0)
            scale_store.flush(0)
        if bool(bad.any()):
            # an out-of-range row reads as zero, exactly like ResidentEngramTable.gather
            self.staging_w.view(count, self.dim)[bad] = 0
            self.staging_s.view(count, self.scale_cols)[bad] = 0
