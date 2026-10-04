"""CUDA-graph-compatible disk fetch for the MoE disk tier ("graph-doorbell fetch").

Design: research/notes/engines/20260928-graph-doorbell-fetch-design.md

Whole-model CUDA graph capture means routing happens INSIDE the replayed graph
while preadv/io_uring are host syscalls that cannot be captured. This module
splits the fetch in two:

- two tiny triton kernels captured in the graph per MoE layer:
  ``_gf_request_kernel`` splits the ensure plan into disk rows (recorded in a
  request block, sequence bumped) and RAM rows (compacted in place so the
  existing fast_index_copy handles them), and ``_gf_spin_kernel`` waits until
  the host service thread acknowledges that the staging rows hold the
  requested experts.
- a host service thread per rank that polls the request block, reads the
  requested checkpoint rows into pinned staging (same preadv machinery as the
  eager path), and releases the sequence with a PLAIN CPU STORE. The thread
  makes ZERO CUDA calls.

TRANSPORT — what empirically works on this platform during a hung graph
replay (each direction probed separately, see .pytools/probe_sysmem_read.py):

- device kernel STORES to host-pinned sysmem: LOST (plain, .wt, atomic_xchg
  all vanished from inside a replayed graph; eager launches land fine).
- in-graph cudaMemcpyAsync D2H node: WORKS (the request block reaches the
  host mirror; seq at the block END makes a partially-completed copy unable
  to expose a new sequence with stale fields).
- thread-issued cudaMemcpyAsync H2D AND thread-launched kernels (fill_) on a
  private stream while the replay spins: NEVER SCHEDULED (tx_stream
  synchronize() never returned — copy engine and compute engine alike).
- device kernel READS of host-pinned sysmem (volatile load and sys-scope
  atomic): WORK — the host store is visible immediately.

So the graph PULLS and the host never touches CUDA:

- request: kernel writes a DEVICE request block; the captured D2H memcpy
  node right after it delivers the block to the pinned host mirror.
- ack: thread preadv's into pinned staging, then a plain CPU store bumps
  resp_host; the spin polls resp_host over PCIe (sys-scope acquire), which
  also orders the spin's subsequent staging reads after the store.
- install: fast_index_copy_jit reads the PINNED HOST staging directly —
  exactly the same zero-copy sysmem reads the eager path uses for host banks,
  and the direction the probe proved.

Serialisation: single decode stream + the spin means requests are strictly
ordered, so ONE request block and ONE staging set are enough: request N+1's
kernel (and its D2H node) executes after request N's install in stream order,
so staging rows are never overwritten while an install can still read them.
"""

from __future__ import annotations

import ctypes
import os
import threading
import time

import torch
import triton
import triton.language as tl

from ..kernel.fast_index_copy import fast_index_copy_jit


@triton.jit
def _gf_request_kernel(
    num_indices_ptr, evict_slots_ptr, src_indices_ptr,
    req_ptr, req_slots_ptr, req_count_dev_ptr,
    row_map_ptr,
    RAM_LOCAL: tl.constexpr, K_MAX: tl.constexpr, LAYER: tl.constexpr,
    TRACE: tl.constexpr,
):
    """Split the ensure plan: disk rows -> device request block, RAM rows ->
    compacted in place (num_indices := ram count). Bump req seq only when at
    least one disk row was requested (warm layers cost zero host round-trips).

    req block layout (int64, device): [0]=count (TRUE count, may exceed K_MAX
    so the thread can fail loud instead of silently dropping rows), [1]=layer,
    [2:2+K_MAX]=disk expert rows (LOCAL expert ids -- the doorbell resolves
    checkpoint rows from them), [2+K_MAX]=sequence (LAST: the captured D2H
    memcpy writes the block linearly, so a new seq implies all fields landed).

    ``row_map`` is the tier's pinned-row layout for LAYER (local id -> bank
    row; identity without a pin file): the RAM/disk split compares the BANK
    ROW against RAM_LOCAL, the request block keeps the local id, and the
    compacted RAM remainder stores the bank row for the host-bank PCIe copy.
    """
    n = tl.load(num_indices_ptr)
    w = 0
    c = 0
    for i in range(n):
        slot = tl.load(evict_slots_ptr + i)
        src = tl.load(src_indices_ptr + i)
        row = tl.load(row_map_ptr + src)
        if row >= RAM_LOCAL:
            if c < K_MAX:
                tl.store(req_slots_ptr + c, slot)
                tl.store(req_ptr + 2 + c, src.to(tl.int64))
            c += 1
        else:
            tl.store(evict_slots_ptr + w, slot)
            tl.store(src_indices_ptr + w, row)
            w += 1
    tl.store(num_indices_ptr, w.to(tl.int64))
    cc = tl.minimum(c, K_MAX)
    tl.store(req_ptr, c.to(tl.int64))
    tl.store(req_ptr + 1, (LAYER + 0 * c).to(tl.int64))
    tl.store(req_count_dev_ptr, cc.to(tl.int64))
    if TRACE:
        tl.device_print("[gf-req] L", LAYER)
        tl.device_print("[gf-req] n=", n)
        tl.device_print("[gf-req] c=", c)
    if c > 0:
        seq_ptr = req_ptr + 2 + K_MAX
        tl.store(seq_ptr, tl.load(seq_ptr, volatile=True) + 1)


@triton.jit
def _gf_spin_kernel(resp_ptr, req_ptr, doorbell_ptr, SEQ_OFF: tl.constexpr,
                    TRACE: tl.constexpr):
    """Wait until the host service thread acknowledges the request written by
    the (stream-ordered) preceding _gf_request_kernel. Skipped entirely during
    graph capture (doorbell off) and on warm layers (req count == 0). Polls
    HOST-PINNED resp over PCIe: device kernel sysmem reads are the one
    host->device direction that works while a graph replay is resident, and
    the sys-scope acquire read orders the install's staging reads after the
    observed ack."""
    if tl.load(doorbell_ptr, volatile=True) == 0:
        return
    if tl.load(req_ptr, volatile=True) == 0:
        return
    target = tl.load(req_ptr + SEQ_OFF, volatile=True)
    if TRACE:
        tl.device_print("[gf-spin] enter target=", target)
    while tl.atomic_add(resp_ptr, 0, sem="acquire", scope="sys") < target:
        pass
    if TRACE:
        tl.device_print("[gf-spin] exit resp=",
                        tl.atomic_add(resp_ptr, 0, sem="acquire", scope="sys"))


class GraphFetchBridge:
    """Per-rank doorbell + staging + host service thread for one DiskTier."""

    def __init__(self, tier, cache, k_max: int) -> None:
        self.tier = tier
        self.k_max = int(k_max)
        self.seq_off = 2 + self.k_max
        device = cache.evict_slots.device
        self._device = device

        # The bridge is created while the engine holds torch.inference_mode();
        # tensors born there are inference tensors, and later writes outside
        # that mode (enable/disable fill_) would raise. Allocate every bridge
        # tensor with inference mode explicitly off.
        with torch.inference_mode(False):
            n_req = 3 + self.k_max
            self.req_dev = torch.zeros(n_req, dtype=torch.int64, device=device)
            self.req_host = torch.zeros(n_req, dtype=torch.int64,
                                        pin_memory=True)
            # The ack word: written by the thread with a plain CPU store,
            # polled by the spin kernel over PCIe. No device copy of it exists.
            self.resp_host = torch.zeros(1, dtype=torch.int64,
                                         pin_memory=True)
            self.doorbell_dev = torch.zeros(1, dtype=torch.int64,
                                            device=device)

            # Staging rows, one pinned host set per bank: the thread preadv's
            # into it and the captured install kernel reads it over PCIe (same
            # zero-copy contract the eager path has with host banks). Same
            # dtype/shape as a host-bank row so fast_index_copy_jit sees
            # exactly the same contract.
            # tier._banks entries are (per-layer [host tensors], gpu_cache).
            self.staging_host = []
            for host_layers, _gpu_cache in tier._banks:
                row = host_layers[0][0]  # layer 0, expert 0 -> one row
                shape = (self.k_max,) + tuple(row.shape)
                self.staging_host.append(
                    torch.zeros(shape, dtype=row.dtype, pin_memory=True))

            # Bridge-private preadv bounce buffer. The eager path's
            # tier._staging_ring() guards buffer reuse with CUDA events
            # recorded on the calling thread's stream; in this thread that
            # event can end up queued behind the graph replay whose spin
            # kernel is itself waiting for our ack (observed deadlock on
            # TP rank>0: thread spinning in ev.synchronize forever).
            # The bounce buffer is host-only scratch — no events needed.
            self._bounce = torch.empty(self.tier._staging_size,
                                       dtype=torch.uint8, pin_memory=True)

        self.req_slots_dev = torch.zeros(self.k_max, dtype=torch.int32,
                                         device=device)
        self.req_count_dev = torch.zeros(1, dtype=torch.int64, device=device)
        self.staging_idx_dev = torch.arange(self.k_max, dtype=torch.int32,
                                            device=device)

        self._dbg = os.environ.get("FT_GRAPH_FETCH_DEBUG", "") == "1"
        self._trace = os.environ.get("FT_GRAPH_FETCH_TRACE", "") == "1"
        # PILOT from the doorbell thread (opt-in): the request block already holds the
        # layer's local disk row ids, so the identity-map prefetch for L+1 needs no extra
        # device->host copy. Off by default: it competes for NVMe bandwidth with the very
        # demand reads this thread is serving, so it is A/B-measured before shipping.
        self._issue_prefetch = os.environ.get("FT_GRAPH_FETCH_PREFETCH", "0") == "1"
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._numpy_view = None
        # Compile-warm both triton kernels OUTSIDE capture (first-call JIT
        # compilation does device work that is illegal inside graph capture).
        if not os.environ.get("FT_GRAPH_FETCH_OFF"):
            self._warm(cache)

    # -- capture-time entry points (python runs only while capturing) --------

    def _warm(self, cache) -> None:
        # Compile every triton specialization ahead of time: triton JIT-compiles
        # on first launch with a given constexpr set, and a compile during CUDA
        # graph capture is illegal. LAYER is a constexpr, so warm all layers.
        # cache.banks entries are (per-layer [HostBank], gpu_cache) pairs.
        num_layers = len(cache.banks[0][0])
        cache.num_indices.fill_(0)
        for layer in range(num_layers):
            self.stage_fetch(cache, layer)
        torch.cuda.synchronize()

    def stage_fetch(self, cache, layer_id: int) -> None:
        """Record request+memcpy+spin+install for one MoE layer (capture-time
        only). In eager mode (tests, warm-up) the same calls just execute."""
        _gf_request_kernel[(1,)](
            cache.num_indices, cache.evict_slots, cache.src_indices,
            self.req_dev, self.req_slots_dev, self.req_count_dev,
            self.tier._row_map_dev[layer_id],
            RAM_LOCAL=self.tier._ram, K_MAX=self.k_max, LAYER=layer_id,
            TRACE=self._trace,
        )
        # Captured memcpy node: DMA the request block to the host-polled
        # mirror. DMA writes to sysmem are host-visible by construction, which
        # kernel stores from a replayed graph empirically are not.
        self.req_host.copy_(self.req_dev, non_blocking=True)
        _gf_spin_kernel[(1,)](
            self.resp_host, self.req_dev, self.doorbell_dev,
            SEQ_OFF=self.seq_off, TRACE=self._trace)
        # Install reads PINNED HOST staging over PCIe — the probe-proven
        # host->device direction that needs zero CUDA calls from the thread.
        for bank_idx, (_host, gpu_cache) in enumerate(cache.banks):
            fast_index_copy_jit(gpu_cache, self.req_slots_dev,
                                self.staging_host[bank_idx],
                                self.staging_idx_dev, self.req_count_dev)

    # -- doorbell lifecycle (host side) ---------------------------------------

    def disable(self) -> None:
        """Called before (re)capture: spin kernels become no-ops."""
        self.doorbell_dev.fill_(0)

    def enable(self) -> None:
        """Called after capture finished: start servicing replay requests."""
        if self._thread is None:
            # Snapshot: ignore every sequence written during capture.
            self._served = int(self.req_host[self.seq_off])
            self._stop.clear()
            if not os.environ.get("FT_GRAPH_FETCH_NOTHREAD"):
                self._thread = threading.Thread(
                    target=self._serve, name="graph-fetch", daemon=True)
                self._thread.start()
        self.doorbell_dev.fill_(1)
        print(f"[graph-fetch] enabled dev={self._device} pid={os.getpid()} "
              f"served={self._served} k_max={self.k_max} "
              f"ram={self.tier._ram}", flush=True)

    def shutdown(self) -> None:
        self._stop.set()
        t = self._thread
        if t is not None:
            t.join(timeout=5)
            self._thread = None

    # -- host service thread ---------------------------------------------------
    # The thread makes ZERO CUDA calls: anything enqueued from here (memcpy or
    # kernel, any stream) was empirically never scheduled while a graph replay
    # spun on the device. preadv + CPU stores only.

    def _view(self):
        if self._numpy_view is None:
            self._numpy_view = self.req_host.numpy()
        return self._numpy_view

    def _serve(self) -> None:
        print(f"[graph-fetch] thread-entry dev={self._device} "
              f"pid={os.getpid()} tid={threading.get_native_id()}", flush=True)
        # A dead service thread hangs the whole engine: every replayed graph spins on an
        # ack that would never come. Observed 2026-10-03 -- an inference-mode RuntimeError
        # inside _read_into_staging killed the thread and froze a prefill for 22 minutes.
        # So: keep serving after a transient error, but give up loudly instead of
        # spinning forever on a request that can never be served -- the same terminal
        # contract as the count > k_max branch, which deliberately never acks.
        fails = 0
        while not self._stop.is_set():
            try:
                self._serve_inner()
                if not self._stop.is_set():
                    print(f"[graph-fetch] service thread stopping dev={self._device}: it "
                          f"refused a request; the server will hang and must be "
                          f"restarted", flush=True)
                return
            except Exception:
                import traceback
                fails += 1
                print(f"[graph-fetch] THREAD ERROR ({fails}) dev={self._device}\n"
                      + traceback.format_exc(), flush=True)
                if fails > 3:
                    print(f"[graph-fetch] service thread stopping dev={self._device} "
                          f"after {fails} consecutive failures; the server will hang "
                          f"and must be restarted", flush=True)
                    return
                time.sleep(0.05)

    def _serve_inner(self) -> None:
        print(f"[graph-fetch] thread-run dev={self._device} "
              f"pid={os.getpid()}", flush=True)
        blk = self._view()
        seq_off = self.seq_off
        served = self._served
        polls = 0
        while not self._stop.is_set():
            seq = int(blk[seq_off])
            if seq == served:
                polls += 1
                if self._dbg and polls % 1000000 == 0:
                    print(f"[graph-fetch] alive dev={self._device} pid={os.getpid()}: "
                          f"served={served} "
                          f"req_seq={seq} resp={int(self.resp_host[0])}",
                          flush=True)
                time.sleep(0)  # yield the GIL between polls (~µs cadence)
                continue
            count = int(blk[0])
            layer = int(blk[1])
            if count > self.k_max:  # impossible by construction; never ack
                print(f"[graph-fetch] FATAL: request count {count} > "
                      f"k_max {self.k_max}; refusing to serve (server will "
                      f"hang and must be restarted)", flush=True)
                return
            # Staging reuse needs no fence: requests are strictly serialized
            # by the spin, so the previous request's install kernel (stream
            # order) has consumed the rows before this request could exist.
            for j in range(count):
                self._read_into_staging(layer, int(blk[2 + j]), j)
            if self._dbg:
                print(f"[graph-fetch] served dev={self._device} seq={seq} layer={layer} "
                      f"count={count} rows={blk[2:2 + min(count, 8)].tolist()}",
                      flush=True)
            # Ack: plain CPU store, AFTER all staging writes (x86 TSO orders
            # store->store; the spin's sys-scope acquire pairs with it). The
            # graph polls resp_host over PCIe — no CUDA op from this thread.
            self.resp_host[0] = seq
            served = seq
            # Only after the ack is the graph unblocked, so neither of the two
            # follow-ups below sits on the request's critical path (both concern the
            # *next* layer anyway).
            if self._issue_prefetch and count:
                # PILOT on the captured path: the just-served disk rows ARE layer L's
                # routing (the device only asks for rows it routed to), so they predict
                # L+1's. Slab acquisition here never waits on a CUDA event -- see
                # DiskTier.prefetch_from_local_ids.
                self.tier.prefetch_from_local_ids(
                    layer, [int(blk[2 + j]) for j in range(count)])
            if layer == self.tier._index.num_layers - 1 and self.tier._telemetry_path:
                # Close the per-turn HITS bitmap from here: with graphs on the Python
                # decode path never runs, so nothing else would ever call this.
                self.tier.end_turn()

    def _read_into_staging(self, layer: int, expert: int, j: int) -> None:
        """preadv one expert's rows (all banks) into pinned staging row j.

        ``expert`` is the LOCAL (slot-cache) id; _group_runs/_fill_scalar_row
        apply the g0 translation themselves, exactly like the eager path.

        A stashed (PILOT-prefetched) row is served from its pinned slab instead of the
        device -- the one way the prefetch can save NVMe traffic on the captured path."""
        tier = self.tier
        entry = None
        # getattr, not attribute access: the unit tests drive this method with a stub
        # tier that has no PILOT state at all (tests/moe/test_graph_fetch.py::_FakeTier),
        # and a raise from here means no ack, i.e. a hung engine (2026-10-03).
        if getattr(tier, "_prefetch_window", 0) > 0 and hasattr(tier, "_stash"):
            with tier._stash_lock:
                entry = tier._stash.pop((layer, expert), None)
        if entry is not None:
            entry["future"].result()  # slab read done? (usually long done)
            tier.stash_to_staging(
                entry, layer, expert,
                [self.staging_host[b][j] for b in range(len(tier._banks))])
            tier._pf_hits += 1
            # mark_turn_src, not a raw write: this thread runs outside the engine's
            # inference mode, where writing the inference-mode _turn_src tensor raises
            # "Inplace update to inference tensor outside InferenceMode" (2026-10-03).
            tier.mark_turn_src(layer, expert, 1)  # served from stash (cf. fetch_pending)
            return
        bounce = self._bounce
        scalar_banks = getattr(tier._index, "scalar_banks", ())
        for bank_idx, (_host, _gpu) in enumerate(tier._banks):
            dst_row = self.staging_host[bank_idx][j]
            if bank_idx in scalar_banks:
                tier._fill_scalar_row(bank_idx, layer, expert, dst_row)
                continue
            for shard_idx, a0, a1, members, _end in tier._group_runs(
                    bank_idx, layer, expert):
                fd, _direct = tier._fd(shard_idx)
                slen = a1 - a0
                mv = (ctypes.c_char * slen).from_address(bounce.data_ptr())
                os.preadv(fd, [mv], a0)
                tier._preadv_calls += 1
                for d0, d1, off, nbytes in members:
                    src = bounce[off - a0:off - a0 + nbytes]
                    dst = dst_row[d0:d1]
                    dst.copy_(src.view(dst.dtype).view(dst.shape))
