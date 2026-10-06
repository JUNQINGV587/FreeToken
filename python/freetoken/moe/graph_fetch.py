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

Operational semantics (M1 hardening):

- k_max = cuda_graph_max_bs x topk bounds one graphed layer's disk misses, so
  a request count above it is a SIZING BUG (or a new multi-token decode step,
  e.g. spec decode, raising the bound), never legitimate traffic. There is no
  eager fallback inside a replayed graph: the service thread refuses (FATAL
  log), sets the engine-visible health flag, counts a doorbell_timeout, and
  poisons the ack so the replay releases into the engine's
  raise_if_unhealthy instead of hanging (the pre-watchdog behaviour was a
  22-minute silent freeze, 2026-10-03).
- The same loud path covers a wedged service thread: a watchdog thread spots
  a request unacked for more than FT_GRAPH_FETCH_TIMEOUT_S (default 10 s),
  counts it, and poisons the ack. A fired watchdog means the step's staging
  is incomplete; the engine raises before sampling and the server must be
  restarted (logged). This is the cpu_executor err[] pattern, host-side
  because the in-graph spin has no bounded wait of its own.
- doorbell == 0 (before enable(), i.e. during capture and the eager compile
  warm-up): the spin kernel no-ops and the install reads staging as-is. Safe
  because capture executes nothing and _warm runs with num_indices == 0 (the
  req-count == 0 short-circuit fires before the doorbell is even read).
  Graphs captured with FT_GRAPH_FETCH_OFF set contain NO fetch nodes at all
  (the capture branch in offload_cache.copy_missing is env-gated), so the env
  is effectively read once per capture set, not per replay.
- One bridge serves EVERY captured batch-size graph: requests are serialized
  by the single decode stream, k_max is sized to the LARGEST graph, and the
  spin's stream ordering serializes staging reuse, so switching between bs
  graphs mid-stream needs no extra coordination.
- Spin-wait telemetry: the spin kernel times itself (%globaltimer) into a
  device stats word [total_ns, waits, peak_ns]; one captured D2H node after
  the LAST layer's spin mirrors it to pinned host memory (kernel stores to
  sysmem from a replay are lost, DMA copies are not -- see the transport
  notes above), so /v1/stats reads it with zero CUDA calls.
- Miss-routing priority once the CPU tier (port item 1) lands: slot hit ->
  GPU-resident rows (today's fast path); RAM-pinned -> PCIe host-bank copy
  (the compacted RAM remainder the request kernel leaves behind); disk ->
  this doorbell. The row_map split in _gf_request_kernel is where the
  RAM/disk boundary is decided; CPU-tier rows must be classified BEFORE the
  doorbell request is built, or they would be fetched from disk needlessly.
"""

from __future__ import annotations

import ctypes
import os
import queue
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor

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
def _gf_spin_kernel(resp_ptr, req_ptr, doorbell_ptr, stats_ptr,
                    SEQ_OFF: tl.constexpr, TRACE: tl.constexpr):
    """Wait until the host service thread acknowledges the request written by
    the (stream-ordered) preceding _gf_request_kernel. Skipped entirely during
    graph capture (doorbell off) and on warm layers (req count == 0). Polls
    HOST-PINNED resp over PCIe: device kernel sysmem reads are the one
    host->device direction that works while a graph replay is resident, and
    the sys-scope acquire read orders the install's staging reads after the
    observed ack.

    The wait is timed with %globaltimer (the dsv41 ft_tier_cu_v.cu caliber)
    and accumulated into stats_ptr = [total_ns, waits, peak_ns] so the host
    can report mean/peak gpu-side doorbell wait without any CUDA call -- one
    captured D2H node after the last layer's spin mirrors the device word to
    pinned host memory (see GraphFetchBridge.stage_fetch)."""
    if tl.load(doorbell_ptr, volatile=True) == 0:
        return
    if tl.load(req_ptr, volatile=True) == 0:
        return
    target = tl.load(req_ptr + SEQ_OFF, volatile=True)
    if TRACE:
        tl.device_print("[gf-spin] enter target=", target)
    t0 = tl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", "=l", [],
                                   dtype=tl.int64, is_pure=False, pack=1)
    while tl.atomic_add(resp_ptr, 0, sem="acquire", scope="sys") < target:
        pass
    t1 = tl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", "=l", [],
                                   dtype=tl.int64, is_pure=False, pack=1)
    tl.atomic_add(stats_ptr, t1 - t0)
    tl.atomic_add(stats_ptr + 1, 1)
    tl.atomic_max(stats_ptr + 2, t1 - t0)
    if TRACE:
        tl.device_print("[gf-spin] exit resp=",
                        tl.atomic_add(resp_ptr, 0, sem="acquire", scope="sys"))


# W21 (2026-10-05): the doorbell service threads polled at ~µs cadence with
# time.sleep(0) -- millions of GIL acquisitions per second per thread, showing as
# 8-13% per thread on the production py-spy and churning the GIL under the main
# scheduler thread. The GPU->host direction forces polling (a device kernel writes
# the doorbell; there is no syscall on the waker side), so the only lever is
# backoff: stay hot for FT_FETCH_SPIN_HOT polls after the last served request
# (covers the inter-layer doorbell burst of a decode replay), then nap
# FT_FETCH_SPIN_NAP_S between polls. Worst added ack latency is one nap (~50 µs)
# on the first doorbell after a long idle -- ~0.02% of a 250 ms token.
_SPIN_HOT = int(os.environ.get("FT_FETCH_SPIN_HOT", "16384"))
_SPIN_NAP_S = float(os.environ.get("FT_FETCH_SPIN_NAP_S", "0.00005"))


@triton.jit
def _gf_scale_convert_install_kernel(
    dst_ptr,           # [S, ROWS*COLS_OUT] uint8 gpu scale bank (row = slot)
    slots_ptr,         # [K_MAX] int32 -> cache slot
    src_ptr,           # [K_MAX, ROWS*2*COLS_OUT] uint8 pinned native staging
    gexp_ptr,          # [K_MAX, 2] int32 pinned: per-half global exponents
    count_ptr,         # [1] int64
    ROWS: tl.constexpr, HALF: tl.constexpr, COLS_OUT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """DS-FP4 conversion-mode install for scale banks: staged NVFP4 per-16 e4m3
    bytes become per-32 e8m0 codes with the per-expert global folded into the
    exponent: out[j] = (in[2j] >> 3) + 120 + gexp (positive pow2 e4m3 bytes only;
    the model-wide audit gate in tools/trace/audit_nvfp4_pow2.py is what licenses
    skipping per-byte verification here). Reads pinned staging + pinned gexp over
    PCIe — the same zero-copy contract as the fast_index_copy_jit install. The
    gate|up halves of a row can carry DIFFERENT globals (two disk segments), so
    the exponent is per half; single-segment banks write both halves equal."""
    pid = tl.program_id(0)
    r = pid % ROWS
    i = pid // ROWS
    count = tl.load(count_ptr)
    if i < count:
        slot = tl.load(slots_ptr + i).to(tl.int64)
        g = tl.load(gexp_ptr + i * 2 + tl.where(r < HALF, 0, 1))
        cols = tl.arange(0, BLOCK)
        src_row = src_ptr + i * (ROWS * 2 * COLS_OUT) + r * (2 * COLS_OUT)
        dst_row = dst_ptr + slot * (ROWS * COLS_OUT) + r * COLS_OUT
        for c0 in range(0, COLS_OUT, BLOCK):
            idx = c0 + cols
            m = idx < COLS_OUT
            b = tl.load(src_row + idx * 2, mask=m, other=0)
            e = (b.to(tl.int32) >> 3) + 120 + g
            tl.store(dst_row + idx, e.to(tl.uint8), mask=m)


def _gf_scale_convert_install(dst_bank: torch.Tensor, slots: torch.Tensor,
                              src_staging: torch.Tensor,
                              gexp_host: torch.Tensor,
                              count: torch.Tensor) -> None:
    """Install wrapper: dst_bank = gpu scale cache (any 1-byte dtype, viewed as
    uint8), src_staging = [k_max, ROWS, 2*COLS_OUT] pinned native bytes."""
    if dst_bank.dtype != torch.uint8:
        dst_bank = dst_bank.view(torch.uint8)
    k_max, rows, cols2 = src_staging.shape
    cols = cols2 // 2
    _gf_scale_convert_install_kernel[(k_max * rows,)](
        dst_bank, slots, src_staging, gexp_host, count,
        ROWS=rows, HALF=rows // 2, COLS_OUT=cols, BLOCK=128,
        num_warps=1,
    )


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

            # Spin-wait telemetry [total_ns, waits, peak_ns]: accumulated
            # device-side by the spin kernel, mirrored to pinned host memory
            # by one captured D2H node after the LAST layer's spin in
            # stage_fetch (kernel stores to sysmem from a replay are lost,
            # DMA copies are not -- same transport asymmetry as the request
            # block above), so spin_stats() needs zero CUDA calls and stays
            # readable even while a replay is wedged.
            self.spin_stats_dev = torch.zeros(3, dtype=torch.int64,
                                              device=device)
            self.spin_stats_host = torch.zeros(3, dtype=torch.int64,
                                               pin_memory=True)

            # Staging rows, one pinned host set per bank: the thread preadv's
            # into it and the captured install kernel reads it over PCIe (same
            # zero-copy contract the eager path has with host banks). Same
            # dtype/shape as a host-bank row so fast_index_copy_jit sees
            # exactly the same contract — EXCEPT in DS-FP4 conversion mode,
            # where the scale banks stage the NATIVE NVFP4 per-16 bytes (twice
            # the cache row width, uint8) plus a per-half global-exponent side
            # buffer, and the install converts on device instead of copying.
            # tier._banks entries are (per-layer [host tensors], gpu_cache).
            self._convert = bool(getattr(tier, "_convert", False))
            self._scale_convert = dict(getattr(tier, "_scale_convert", None) or {})
            self._gexp_host: dict[int, torch.Tensor] = {}
            self.staging_host = []
            for bank_idx, (host_layers, _gpu_cache) in enumerate(tier._banks):
                row = host_layers[0][0]  # layer 0, expert 0 -> one row
                if bank_idx in self._scale_convert:
                    self.staging_host.append(torch.zeros(
                        (self.k_max, row.shape[0], row.shape[1] * 2),
                        dtype=torch.uint8, pin_memory=True))
                    self._gexp_host[bank_idx] = torch.zeros(
                        (self.k_max, 2), dtype=torch.int32, pin_memory=True)
                    continue
                shape = (self.k_max,) + tuple(row.shape)
                self.staging_host.append(
                    torch.zeros(shape, dtype=row.dtype, pin_memory=True))

            # Bridge-private preadv bounce slabs (O_DIRECT needs an aligned
            # buffer), PRE-ALLOCATED here — the eager path's tier._staging_ring()
            # guards buffer reuse with CUDA events recorded on the calling
            # thread's stream; in a service thread that event can end up queued
            # behind the graph replay whose spin kernel is itself waiting for
            # our ack (observed deadlock on TP rank>0: thread spinning in
            # ev.synchronize forever). The bounce is host-only scratch — no
            # events needed. Pre-allocation is not a luxury: cudaHostAlloc from
            # a service thread while a replay spins was empirically never
            # scheduled (same transport limit as thread-issued copies), so the
            # slabs must exist before enable(). One slab per row-pool worker
            # plus one for the service thread itself; _bounce_buf hands them
            # out via thread-local storage.
            self._bounce_tls = threading.local()
            self._bounce_slabs = queue.SimpleQueue()
            for _ in range(
                    1 + int(os.environ.get("FT_GRAPH_FETCH_WORKERS", "4"))):
                self._bounce_slabs.put(
                    torch.empty(self.tier._staging_size, dtype=torch.uint8,
                                pin_memory=True))

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
        self._watchdog_thread: threading.Thread | None = None
        # Engine-visible health flag (the cpu_executor err[] pattern, host
        # side): set by the watchdog on an unacked request past the timeout
        # and by the service thread on a count > k_max refusal; read once per
        # forward via raise_if_unhealthy. A plain bool is enough (GIL-atomic).
        self._err = False
        self._timeout_s = float(
            os.environ.get("FT_GRAPH_FETCH_TIMEOUT_S", "10"))
        # Row-level fetch pool: the W15 py-spy profile showed the doorbell
        # service thread is the decode bottleneck (one python thread doing
        # per-row preadv + member copies serially). Rows within one request
        # are independent; the ack still waits for ALL of them (see
        # _serve_inner), so the stream-ordering contract is unchanged.
        # Threads spawn lazily on first submit, so an unused pool costs nothing.
        self._row_workers = int(os.environ.get("FT_GRAPH_FETCH_WORKERS", "4"))
        self._row_pool = ThreadPoolExecutor(
            max_workers=self._row_workers,
            thread_name_prefix="graph-fetch-row")
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
            self.spin_stats_dev, SEQ_OFF=self.seq_off, TRACE=self._trace)
        if layer_id == len(cache.banks[0][0]) - 1:
            # One captured D2H node per replay mirrors the cumulative spin
            # stats to the host, stream-ordered after every layer's spin.
            self.spin_stats_host.copy_(self.spin_stats_dev, non_blocking=True)
        # Install reads PINNED HOST staging over PCIe — the probe-proven
        # host->device direction that needs zero CUDA calls from the thread.
        for bank_idx, (_host, gpu_cache) in enumerate(cache.banks):
            if bank_idx in self._scale_convert:
                _gf_scale_convert_install(
                    gpu_cache, self.req_slots_dev, self.staging_host[bank_idx],
                    self._gexp_host[bank_idx], self.req_count_dev)
                continue
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
            self._err = False
            if not os.environ.get("FT_GRAPH_FETCH_NOTHREAD"):
                # Pre-spawn the row workers and their pinned bounce slabs NOW:
                # a cudaHostAlloc issued by a service thread DURING a replay
                # spin is the same class of CUDA call that was empirically
                # never scheduled (2026-09-28 transport probe), so every slab
                # must exist before the first request can arrive.
                warm = [self._row_pool.submit(self._bounce_buf)
                        for _ in range(self._row_workers)]
                for f in warm:
                    f.result()
                self._thread = threading.Thread(
                    target=self._serve, name="graph-fetch", daemon=True)
                self._thread.start()
                self._watchdog_thread = threading.Thread(
                    target=self._watchdog, name="graph-fetch-watchdog",
                    daemon=True)
                self._watchdog_thread.start()
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
        w = self._watchdog_thread
        if w is not None:
            w.join(timeout=5)
            self._watchdog_thread = None
        self._row_pool.shutdown(wait=False)

    # -- observability + health (host side, zero CUDA calls) -------------------

    def spin_stats(self) -> dict:
        """GPU-side spin-wait ledger, read off the pinned mirror (the dsv41
        ``gpu_wait_ms`` caliber): ``doorbell_wait_ms`` is the mean device wait
        per doorbell request, ``_peak`` the worst single wait, both over the
        process lifetime. Zero CUDA calls, so /v1/stats stays readable even
        while a replay is wedged in its spin."""
        total_ns, waits, peak_ns = (int(x) for x in self.spin_stats_host[:3])
        return {
            "doorbell_spins": waits,
            "doorbell_wait_ms": total_ns / 1e6 / waits if waits else 0.0,
            "doorbell_wait_ms_peak": peak_ns / 1e6,
        }

    def raise_if_unhealthy(self) -> None:
        """Engine per-forward health check (one python bool; the doorbell
        analogue of cpu_executor's err[]). A fired watchdog or an over-k_max
        refusal means a replay was let out of its spin WITHOUT valid staging;
        the step must not ship its tokens, so the engine calls this between
        replay and sampling."""
        if self._err:
            raise RuntimeError(
                "graph-doorbell fetch failed: a request went unacked past "
                f"FT_GRAPH_FETCH_TIMEOUT_S={self._timeout_s}s or exceeded "
                f"k_max={self.k_max} (see the [graph-fetch] FATAL/WATCHDOG "
                "log); the step was aborted before sampling but the server "
                "must be restarted -- slot contents are no longer trustworthy")

    def _watchdog(self) -> None:
        """Loud-failure path for a wedged service thread (the cpu_executor
        err[] pattern, host-side because the in-graph spin has no bounded
        wait of its own).

        The spin kernel polls resp_host forever; a dead service thread would
        hang every replay with zero diagnostics (observed 2026-10-03: a
        22-minute frozen prefill). This thread watches for a request that
        stays unacked past FT_GRAPH_FETCH_TIMEOUT_S, then counts it
        (doorbell_timeouts), sets the health flag, and poisons the ack so the
        replay completes and the engine's raise_if_unhealthy turns the step
        into a loud error before its tokens are sampled. The poisoned replay
        installs incomplete staging, which is why a fired watchdog means
        restart-required (logged)."""
        base = self._served
        pending_seq = None
        pending_since = 0.0
        while not self._stop.wait(0.05):
            req_seq = int(self.req_host[self.seq_off])
            if req_seq <= base or req_seq <= int(self.resp_host[0]):
                pending_seq = None
                continue
            if pending_seq != req_seq:
                pending_seq = req_seq
                pending_since = time.monotonic()
                continue
            if time.monotonic() - pending_since < self._timeout_s:
                continue
            count = int(self.req_host[0])
            layer = int(self.req_host[1])
            print(f"[graph-fetch] WATCHDOG TIMEOUT dev={self._device}: "
                  f"request seq={req_seq} layer={layer} count={count} unacked "
                  f"for >{self._timeout_s}s (service thread wedged); failing "
                  f"the step loudly -- the server must be restarted",
                  flush=True)
            self._err = True
            rec = getattr(self.tier, "record_doorbell_timeout", None)
            if rec is not None:
                rec()
            # Poison the ack AFTER setting err: the spin exits, the replay
            # completes, and raise_if_unhealthy fails the step.
            self.resp_host[0] = req_seq
            base = req_seq  # never refire on the poisoned sequence

    def _refuse_overcount(self, count: int, seq: int) -> None:
        """count > k_max is impossible by construction (k_max = max graph bs x
        topk bounds one graphed layer's disk misses); seeing it means the
        bridge was sized under the captured graphs, or a future multi-token
        decode step (e.g. spec decode) raised the bound. There is NO eager
        fallback inside a replayed graph, so fail loud: flag the engine
        health check, count a doorbell_timeout, and poison the ack so the
        spin releases into the engine's raise -- the pre-watchdog behaviour
        (never ack) was the 22-minute silent freeze of 2026-10-03, and
        silently serving min(count, k_max) rows would be silent corruption."""
        print(f"[graph-fetch] FATAL: request count {count} > "
              f"k_max {self.k_max} dev={self._device}; refusing to serve -- "
              f"the engine raises on this step and the server must be "
              f"restarted", flush=True)
        self._err = True
        rec = getattr(self.tier, "record_doorbell_timeout", None)
        if rec is not None:
            rec()
        self.resp_host[0] = seq

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
        # Own bounce slab up front (see enable(): cudaHostAlloc is unsafe
        # once a replay can be spinning).
        self._bounce_buf()
        # A dead service thread hangs the whole engine: every replayed graph spins on an
        # ack that would never come. Observed 2026-10-03 -- an inference-mode RuntimeError
        # inside _read_into_staging killed the thread and froze a prefill for 22 minutes
        # (the watchdog added after that incident now bounds the same failure to
        # FT_GRAPH_FETCH_TIMEOUT_S + a loud engine error). So: keep serving after a
        # transient error, but give up loudly instead of spinning forever on a request
        # that can never be served -- the same terminal contract as the count > k_max
        # branch.
        fails = 0
        while not self._stop.is_set():
            try:
                self._serve_inner()
                if not self._stop.is_set():
                    print(f"[graph-fetch] service thread stopping dev={self._device}: it "
                          f"refused a request; the engine raises per forward and the "
                          f"server must be restarted", flush=True)
                return
            except Exception:
                import traceback
                fails += 1
                print(f"[graph-fetch] THREAD ERROR ({fails}) dev={self._device}\n"
                      + traceback.format_exc(), flush=True)
                if fails > 3:
                    print(f"[graph-fetch] service thread stopping dev={self._device} "
                          f"after {fails} consecutive failures; the watchdog now fails "
                          f"the next request loudly and the server must be restarted",
                          flush=True)
                    return
                time.sleep(0.05)

    def _serve_inner(self) -> None:
        print(f"[graph-fetch] thread-run dev={self._device} "
              f"pid={os.getpid()}", flush=True)
        blk = self._view()
        seq_off = self.seq_off
        served = self._served
        polls = 0
        idle = 0
        while not self._stop.is_set():
            seq = int(blk[seq_off])
            if seq == served:
                polls += 1
                idle += 1
                if idle > _SPIN_HOT:
                    time.sleep(_SPIN_NAP_S)  # quiet period: back off the GIL storm
                    continue
                if self._dbg and polls % 1000000 == 0:
                    print(f"[graph-fetch] alive dev={self._device} pid={os.getpid()}: "
                          f"served={served} "
                          f"req_seq={seq} resp={int(self.resp_host[0])}",
                          flush=True)
                time.sleep(0)  # yield the GIL between polls (~µs cadence)
                continue
            idle = 0
            t_serve0 = time.perf_counter_ns()
            count = int(blk[0])
            layer = int(blk[1])
            if count > self.k_max:
                self._refuse_overcount(count, seq)
                return
            # Staging reuse needs no fence: requests are strictly serialized
            # by the spin, so the previous request's install kernel (stream
            # order) has consumed the rows before this request could exist.
            if count == 1:
                self._read_into_staging(layer, int(blk[2]), 0)
            else:
                # Parallel rows: one row's preadv (DMA, GIL released) overlaps
                # another row's memmove (CPU). f.result() re-raises a worker
                # failure into _serve's error counter; the ack below only
                # fires after every row of this request is staged.
                futs = [self._row_pool.submit(
                            self._read_into_staging, layer,
                            int(blk[2 + j]), j)
                        for j in range(count)]
                for f in futs:
                    f.result()
            if self._dbg:
                print(f"[graph-fetch] served dev={self._device} seq={seq} layer={layer} "
                      f"count={count} rows={blk[2:2 + min(count, 8)].tolist()}",
                      flush=True)
            # Ack: plain CPU store, AFTER all staging writes (x86 TSO orders
            # store->store; the spin's sys-scope acquire pairs with it). The
            # graph polls resp_host over PCIe — no CUDA op from this thread.
            self.resp_host[0] = seq
            served = seq
            # Doorbell telemetry (W22 blind spot): rows + full host-side
            # latency of this request. CPU-only accounting, off the ack path.
            self.tier.record_doorbell(
                count, (time.perf_counter_ns() - t_serve0) // 1000)
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
        bounce = self._bounce_buf()
        scalar_banks = getattr(tier._index, "scalar_banks", ())
        _row_groups = getattr(tier, "_row_groups", None)
        if _row_groups is not None and not getattr(self, "_convert", False):
            # R1x native path: one cross-bank merged plan (6 -> 2 preadv/row
            # on the v41 layout, identical bytes). getattr: the CPU staging
            # tests drive this with stub tiers that only know _group_runs.
            for bank_idx in scalar_banks:
                tier._fill_scalar_row(bank_idx, layer, expert,
                                      self.staging_host[bank_idx][j])
            for shard_idx, a0, a1, members, _end in _row_groups(
                    layer, expert):
                fd, _direct = tier._fd(shard_idx)
                slen = a1 - a0
                mv = (ctypes.c_char * slen).from_address(bounce.data_ptr())
                os.preadv(fd, [mv], a0)
                tier._preadv_calls += 1
                src_base = bounce.data_ptr() - a0
                for bank_idx, d0, d1, off, nbytes in members:
                    ctypes.memmove(
                        self.staging_host[bank_idx][j][d0:d1].data_ptr(),
                        src_base + off, nbytes)
            return
        for bank_idx, (_host, _gpu) in enumerate(tier._banks):
            dst_row = self.staging_host[bank_idx][j]
            if not getattr(self, "_convert", False) and bank_idx in scalar_banks:
                # Native mode only: cache bank == disk bank. In conversion mode
                # the four cache banks collide with the NVFP4 scalar ids {2, 5}
                # -- every staged row here is a real disk read (same guard as
                # DiskTier._fetch_expert_inner). getattr default: the CPU
                # staging tests build the bridge with object.__new__.
                tier._fill_scalar_row(bank_idx, layer, expert, dst_row)
                continue
            sc = getattr(self, "_scale_convert", {}).get(bank_idx)
            if sc is not None:
                # DS-FP4 conversion: stage the NATIVE per-16 scale extents
                # (double-width staging) and publish the per-half global
                # exponents for the device-side convert-install kernel. The
                # globals come from the host-resident preload blob, not disk.
                from .nvfp4_to_dsfp4 import global_exponent
                disk_bank, blob_bank = sc
                blob = tier._scalar_blob(layer, blob_bank)
                nseg = len(tier._dst_slices[bank_idx])
                vals = struct.unpack_from(f"<{nseg}f", blob, expert * 4 * nseg)
                gh = self._gexp_host[bank_idx][j]
                gh[0] = global_exponent(vals[0])
                gh[1] = global_exponent(vals[-1])
                groups = tier._group_runs(bank_idx, layer, expert,
                                          disk_bank=disk_bank)
            else:
                # Native mode keeps the exact old call signature (unit stubs
                # implement _group_runs without the disk_bank kwarg).
                groups = (tier._group_runs(bank_idx, layer, expert,
                                           disk_bank=tier._disk_bank[bank_idx])
                          if getattr(self, "_convert", False)
                          else tier._group_runs(bank_idx, layer, expert))
            for shard_idx, a0, a1, members, _end in groups:
                fd, _direct = tier._fd(shard_idx)
                slen = a1 - a0
                mv = (ctypes.c_char * slen).from_address(bounce.data_ptr())
                os.preadv(fd, [mv], a0)
                tier._preadv_calls += 1
                src_base = bounce.data_ptr() - a0
                for d0, d1, off, nbytes in members:
                    # Raw memmove, NOT torch copy_: the staging row is pinned
                    # HOST memory, and copy_ routes float8 banks through a slow
                    # elementwise kernel instead of memcpy (the doorbell service
                    # thread spent ~34% of its samples in that copy_ -- W15
                    # py-spy). ctypes releases the GIL, so this also overlaps
                    # with the row pool's preadv DMA.
                    ctypes.memmove(dst_row[d0:d1].data_ptr(),
                                   src_base + off, nbytes)

    def _bounce_buf(self) -> torch.Tensor:
        """This thread's pinned preadv bounce slab, handed out from the pool
        pre-allocated in __init__ (cudaHostAlloc from a service thread is
        unsafe once a replay can spin — see __init__)."""
        b = getattr(self._bounce_tls, "buf", None)
        if b is None:
            try:
                b = self._bounce_slabs.get_nowait()
            except queue.Empty:
                raise RuntimeError(
                    "graph-fetch bounce slab exhaustion: more fetch threads "
                    "than pre-allocated slabs (1 + FT_GRAPH_FETCH_WORKERS)")
            self._bounce_tls.buf = b
        return b
