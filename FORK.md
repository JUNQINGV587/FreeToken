# FORK: sm_89 (Ada) production line

This fork carries a production branch for **FreeToken on NVIDIA Ada Lovelace (sm_89)** —
developed and validated on 2×L20 48GB (PCIe P2P, no NVLink), TP=2 + owner-local expert
parallelism, serving Qwen3.8-Flash-Next-NVFP4.

The default branch **`sm89-moe-offload`** is the production mainline. Upstream is tracked as
`origin/main` and merged on every sync. Experiments live on `exp/*`.

## What this branch adds over upstream

- `perf(distributed)`: **custom all-reduce donor** (vLLM one-stage kernel) for small decode
  tensors — 22–95 µs → 4–9 µs per [bs≤8, 2560] all-reduce on PCIe-only 2×GPU, bit-identical
  (fp32 accumulation in rank order). Production-measured: single-stream +4–6%, bs=8 +52%.
  `FREETOKEN_CUSTOM_ALL_REDUCE=0` opts out.
- `perf(moe)`: **owner route map fusion** — one kernel replaces the ~10-op owner-EP admission
  chain (22.2 → 3.5 µs/layer). CPU devices keep the torch reference chain.
- `feat(moe)`: owner-local **hybrid** decode path (GPU/CPU exactly-once split under owner EP)
  — present but inert in production (per-layer handshake cost makes it a regression on this
  box; kept for other platforms).
- `feat(moe)`: graph-safe decode alignment kernel for >1024-expert slot spaces (marlin path
  prerequisite, not yet wired).
- `feat(kvcache)`: **host KV tier** — a spilled prefix stays matchable by keeping its pages in
  host RAM instead of dropping them, with the geometry cross-checked against the pool's own
  buffers and the QSA index shadow plus mrope positions moving with each page. Opt in with
  `FREETOKEN_KV_HOST_TIER_PAGES=<pages>` (12,288 pages ≈ 9.8 GB of host RAM at the production
  page size); unset or 0 keeps the tier out of the manager entirely and the tree evicting as it
  always did. It does not free device memory -- it is host RAM buying prefix reuse, so the arm
  that matters is hit ratio/TTFT, and the arm that must not move is output bits.
- `feat(moe)`: **opt-in prefill staging profile** — one log line per prefill chunk attributing
  its host wall time to entry / PLE / attention (mix, core, combine) / MoE staging (classify,
  hit compact, hit gather, batch memcpy, wait, release) / expert GEMM. `FREETOKEN_PREFILL_PROFILE=1`
  turns it on; unset, every call site is a no-op method call and no timestamp is read. Built to
  attribute the ~1.2 s host-CPU TTFT floor of a prefill batch on this box, which is flat in prompt
  length (7 -> 4343 prompt tokens: rank-0 CPU 1200 -> 1210 ms) and charged per batch, not per
  request. Method and numbers: `research/notes/freetoken/202609-freetoken-speed-report.md` §6.
- `feat(moe)`: the same profile, one level deeper. Attention splits into proj / norm+rope /
  indexer / QSA / o_proj; the batch memcpy splits into miss list / plan+tensors / driver call /
  event record; buffer invalidation and the route-mask elementwise pair get their own phases, and
  an `ops=` counter section reports how many host operations each region ran (a region that costs
  milliseconds for three calls is a different problem from one that costs the same for three
  hundred). `FREETOKEN_PREFILL_RELAUNCH=1` additionally relaunches the idempotent hit-compaction
  kernel back to back and times the second launch: that separates "this launch is slow" from "the
  host was slow around it". All of it stays off unless `FREETOKEN_PREFILL_PROFILE=1` is also set.
- `feat(moe)`: **opt-in per-layer device timeline** for the same chunk (`FREETOKEN_PREFILL_TIMELINE=1`,
  again only with the profile on). The host phases above say which region the time showed up in, and
  that region moves between staging configurations (the same attention code costs 13.7 ms/layer in
  the ring configuration and 1.06 ms/layer in the active-only one), so they cannot name the
  bottleneck. This records four reused CUDA events per layer plus the host clock over the same span -
  `copy_begin`/`copy_end` on the copy stream, `wait_done`/`gemm_end` on the compute stream - and
  writes one CSV row per layer per chunk (`FREETOKEN_PREFILL_TIMELINE_OUT`). The compute stream is
  FIFO, so `wait_done[i] - gemm_end[i-1]` is the device time spent blocked on layer *i*'s staging;
  comparing it against the host gap over the same span separates "the device waited for the copy"
  from "the device was idle because the host had not enqueued the work yet". The dump is deferred and
  never blocks (`Event.query`, one chunk of lag) so the chunk's `host` number stays comparable with
  every other arm. The same flag adds a per-layer `attn_core`/`moe_total` series and a `batch=`
  section (entry count, bytes, size histogram, driver-call wall time) to the profile line.
- `feat(moe)`: the `batch=` section carries a **byte breakdown by staging origin** (`src=`), because
  the total alone cannot say whether the PCIe bytes are avoidable. `src=miss:<entries>/<bytes>` counts
  the coalesced miss runs of the large banks, `src=small:<entries>/<bytes>` counts the banks below
  `_SMALL_BANK_FEAT_BYTES`, which are copied whole-layer **even at 100% residency** to keep every batch
  entry above the driver's async floor and to cover the hit rows the D2D gather skips for them. A small
  bank's bytes are therefore pure overhead at steady state, and this split is the only way to price
  them before changing the copy plan. Both counters come from `bank_byte_split`, the same rule the plan
  loop applies.
- The profile line also reports `| hit=<rows>/<bytes>`: the volume the hit-D2D gather moves from the cache to
  the layer buffer. It prints outside the `batch=` section on purpose, because a fully resident layer stages
  nothing yet still gathers. Measured at steady state it is larger than the PCIe batch (19.7 GB vs 14.3 GB per
  chunk), which is the point: the gather trades link bytes for HBM bytes, and only the link is saturated.
- `feat(moe)`: `FREETOKEN_PREFILL_SMALL_GATHER=1` (default off) lets the hit-D2D gather cover the small banks
  too, so they stage only their miss runs instead of the whole layer. The accounting above prices those
  whole-layer copies at ~24% of the prefill PCIe bytes, paid even at full cache residency where every row is
  a hit; the gather moves the same bytes over HBM instead of PCIe. At matched cache state that is 16.41 -> 14.20
  GB per chunk and TTFT 1.49 -> 1.30 s (-13%, 4.3K prompt 1.83 -> 1.64 s), with bit-identical outputs. Every row
  still arrives exactly once (hit -> gather, miss -> copy run), which is what the GPU test checks; the flag is
  read at cache construction so a test can flip it.
- TP2/owner-EP serving stack from PR #447 lineage + vision TP sharding.

## Measurement protocol

A number from this branch is only comparable inside the same boundary and the same
output window; each rule below was learned by getting a wrong number first.

- **Decode needs an output window of >= 512 tokens, 1024+ preferred.** Client-side SSE
  under-reads by 11% at a 128-token window and 22% at 64 (sampling ramp plus the first
  graph steps after prefill). The `gen throughput` field in the engine log is a
  per-step instantaneous value - do not average it.
- **`reasoning_content` counts as output.** Counting only `content` reports the first
  token at end-of-reasoning instead of first token, and shortens `ct`.
- **Prefer prompts that cannot stop early.** A summarize instruction returns ~110 tokens
  whatever the budget (EOS), so every depth carries the same ramp penalty and the depth
  curve looks flat; an enumerate instruction runs to the budget and exposes the real slope.
- **Discard the first request after a long-context one.** Releasing a 255K sequence costs
  the next request a one-off slowdown, observed anywhere from 0 to -25% over three samples.
- **Warm the engine with short-request and decode traffic before measuring long prefill.**
  On a freshly booted engine long prefill reads ~28% low (2690 vs 3641 tok/s), and repeated
  long prefills do not clear it - a batch of short requests does.
- **Read concurrent decode in the saturated phase**, when all streams are decoding. An
  end-to-end average over a concurrent run mixes in queue starvation (0.7-1.4 t/s while a
  later prefill holds the GPU) and reads far too low.
- **Take KV occupancy from the engine log `token usage`**, not from polling `/v1/stats`:
  under saturation the stats request itself queues and the sampler under-reads (47.9%
  sampled against the 97% the engine reported for the same run).
- **A per-call `cuda.Event` around a small op measures the Python launch gap, not the GPU.**
  At bs=1 the QSA sparse-attend call reads 86.9 us that way while its two kernels take 8.14 us
  and a one-element `fill_` in the same harness reads 24.6 us. Use torch profiler device time,
  or replay the call inside a CUDA graph, to get the number production actually pays.

## Measured and deliberately not changed

- **QSA sparse-attention tier ladder** (`kernel/triton/qsa/attend.py`, the `(block_n,
  target_splits, partial_warps)` table tuned on GB300). Measured on 2xL20 at production shapes
  (1 local KV head, group_size 12, head_dim 256, selection width 2051, page 64): inside the
  captured decode graphs the call costs 19.9-23.0 us/layer, i.e. ~2.0% of a 12.2 ms step at
  bs<=4 and ~2.2% at bs=8, of which only 8.1-14.7 us are the two kernels. The ladder only
  chooses block_n and the split count, so re-tuning it cannot reach the +0.5% e2e threshold in
  force here; the visible residue is the split-K workspace, not the tiling. Numbers and method:
  `research/runs/202609-freetoken-port-batch1/sweep-summary.md`. The QSA *indexer* chain
  (`_qsa_mqa_paged_kernel` and the top-k kernels) was not measured and is a separate question.

- **TP-sharded experts**: every rank holds half of every routed expert along the intermediate
  axis and the MoE layer all-reduces the partial sums (upstream #385, community `tp4_5060ti`),
  instead of this fork's owner-local EP where a rank owns whole, disjoint experts. Ported onto
  this tree (`exp/tp-shard-eval`) and A/B'd against production on 2026-09-23: same model, same
  image apart from the five ported files, same flags except `--moe-ep-size`, both engines freshly
  started, warmed and driven by the same probes. At **equal cache bytes** (17,600 half-width slots
  for the sharded arm against production's 8,800 full-width ones) the two are indistinguishable:
  cold 262K prefill 3173 vs 3149 tok/s, six concurrent 8K prompts 3.80 vs 3.91 s median TTFT, miss
  rate 0.083 vs 0.084 and predicted routing hit 0.9971 vs 0.9989, while production itself scatters
  3149-3368 tok/s across two identical runs. An earlier pass that gave both arms the same
  `--moe-cache-size` rather than the same bytes put EP ahead by 16-28%; that is a slot-width
  artifact, not a property of the layouts, and the working-set numbers quoted from it compared
  different cache warm-up stages and are withdrawn. Functionally equal in both passes (smoke,
  penalties, logprobs, geometry, 0 Xid). Declined: no measurable benefit at equal memory, against a
  kernel-contract change (`tp_ok` plus the Triton layout/pack moving to `local_intermediate`), a
  fork-only predicate, and marlin/b12x excluded under sharding. The mechanism is sound and verified
  on its own (slices partition the checkpoint exactly; two half-width banks sum to the full-width
  output on real kernels). Numbers:
  `research/runs/202609-freetoken-upstream-sync/arm-tpshard-slots8800/` (slot-count pass) and
  `arm-tpshard-eqbytes/` (equal-bytes pass).

## Precision contract

Zero precision degradation vs upstream: kernel swaps are verified bit-identical or within
1 bf16 ulp (fp32 accumulation-order noise only), with output A/B on the production model.
Deterministic decode (no atomic-order-dependent kernels).

## Upstream sync log

Merges use `git merge origin/pr/<N>` so upstream SHAs are preserved and a later real merge
into `origin/main` is recognised as already applied. Survey, full classification and the
open decision list: `research/notes/freetoken/202609-freetoken-upstream-research.md`.

**2026-09-23** (base `271e8be4`, 8 commits):

- `origin/main` `cab110ec` - glm4 router correction bias in fp32 (#469). Routine sync.
- **#505** preserve the pending hybrid checkpoint across prefill chunks: a short
  continuation chunk cannot mint a new mamba snapshot, so `mamba_last_track_seqlen` has to
  survive to the final prefill commit, otherwise the hybrid prefix cache can match a
  recurrent state that does not belong to the cached prefix. This is the prefill and cache
  path the 2026-09-22 crash RCA pointed at
  (`research/runs/202609-freetoken-ttft-window/incident-20260922-xid31.md`).
- **#464** match stop strings with incremental decoding (same decode path as the frontend,
  terminal EOS excluded) instead of decoding a suffix every step.
- **#495** wait for `size-1` rank subscribers before the first broadcast (PUB to XPUB).
  Live risk here: production runs TP=2 and a lost first broadcast deadlocks both ranks.
- **#527** reject non-finite sampling penalties at the API boundary. The substantive half of
  that PR (applying presence and frequency penalties) already existed in this fork together
  with the logprobs plumbing, so ours was kept.
- **#500** single-launch MoE prefill buffer invalidation. The boolean-mask form hid a
  device-to-host synchronization per call, twice per chunk across 48 layers, which
  serialized the prefill-overlap pipeline on long cached contexts. Took upstream's kernel
  revision (identical body plus a bounds guard and a wrapper-level CPU fallback) and kept
  this fork's profiler phase and caller-side CPU fallback. **A first GPU run of that kernel
  still owes a compute-sanitizer memcheck and the Xid-delta check before it serves traffic.**
- `origin/fix/prefill-jit-warmup` - warm prefill triton kernels at startup and stop l2norm
  recompiling per token count. Both take work off the first request of a backend, where the
  1.03-1.04 s cache-warm TTFT floor lives.
- **#531** clamp `/v1/models` context_length to the allocated KV pool. Both sides clamped
  from different sources, so both intents were merged (enforced value first, then clamped by
  `kv_pool_geometry()`); that combination is what makes upstream's two tests pass.

**2026-09-23** (base `7f227d9a`, 3 commits, community survey round):

- **#85** round onto the e4m3 grid before the native downcast, at the two call sites this
  fork's own 2026-09-23 fix missed (`dsv4/fp8_linear.py`'s two quant kernels and
  `fp8_pertensor_linear.py::_static_quant_kernel`), with its tests. The author left the diff
  for anyone to pick up after maintainers declined AI-heavy contributions; the commit keeps
  his authorship and both measurements (his 0.38% of 2**22 samples, this fork's 0.34% of 50k)
  are in the merged comment. Not on the production hot path (that model is pure NVFP4).
- **#466** tolerate coalesced msgpack frames in the zmq pull queues. One frame carrying two
  packed objects raised `ExtraData` in `msgpack.unpackb` and killed the worker process; the
  buffered unpacker decodes the first object and holds the rest. Same file as this fork's
  #495 change, so the merge only touched the import line. Its async test is a bare `async def`
  while this repo declares only `pytest` (no pytest-asyncio, no `asyncio_mode`), so it failed
  collection here; rewritten to drive the coroutine with `asyncio.run`, same assertions.
- Survey of all 969 branches and 251 `pr/*` refs, with per-candidate verdicts: 21 community
  picks turned out to be already carried, `#198`'s KV reserve already exists here as a superset
  (ours adds the `num_token_override` branch), and `#484` is byte-identical to this tree.
  Full table: `research/notes/freetoken/202609-freetoken-upstream-research.md` section 11.

Not merged, with reasons: #499 (its commits depend on `kv_host_offload.py`, a feature this
tree does not carry), #525 (collides with this fork's own host tier, and the community's own
stack ends up refusing the host KV tiers on an NVFP4 pool, which is this fork's combination),
#491 (large refactor of the hybrid decode path this fork already owns), #385/#104/#507
(alternative TP implementations, measured against owner-local EP on 2026-09-23 and declined:
see "Measured and deliberately not changed"), #502/#292/#327 and the GGUF-in-FTW work on
`vektory79` (model directions this box does not serve; revisit when one is actually served).

Flag naming: upstream renamed the backend switch to `--moe-backend`; this branch keeps
`--moe-strategy` as the public name (`config.py` folds the old name in `__post_init__`) and
production argv plus `ops/` scripts depend on it.

**2026-09-23 round 2** (base `d95ca203`, 7 commits; community re-survey after the morning
round - full detail in the research note's section 12):

- `origin/main` unchanged (`cab110ec`); the 143 open PRs unchanged; a branch-tip diff of all
  13 community forks found exactly one fork with new in-domain work: **vektory79**'s hybrid
  cache "fix3" campaign (09-21/22, branch-only, no PR). gberasmus87's fork (first time in the
  local ref set) was fully classified: the qwen4_exp o_proj row-parallel fix (#429) is already
  in this tree, the block-FP8 dense reader exists here in a TP-aware superset, and the rest is
  models this box does not serve.
- `44046f5f` mamba eviction ordered by a per-node snapshot_lru stamp (vektory79 `9732be02`):
  walks re-stamp every on-path node with one shared tic, so without a per-snapshot stamp all
  on-path victims tie and eviction is arbitrary. Conflict with this fork's host tier resolved
  by keeping both intents (their heap key + our host-resident filter).
- `17635d8c` `--linear-state-cache-ratio` CLI flag (vektory79 `e5730e0e`): the engine config
  field already existed; this exposes it and fails fast on ratio <= 0, plus `int` -> `ceil`
  in the pool sizing.
- `daa17df0`/`243e5f86` invariant comments + cross-conversation snapshot-eviction test.
- `ba109809` **donate per-chunk boundary snapshots in chunked prefill** (vektory79 `70808240`):
  intermediate ChunkedReq chunks never called cache_req, so every mid-history divergence
  re-prefilled in full (the author's hardware measurement: 60184 -> 11480 tokens on turn 2).
  This tree still carried the pre-fix drain skip, so the bug was live here. Their test suite
  pins transfer semantics; `f231927d` adapts the slot-id assertions to this fork's
  copy-on-donate (#287) - pool conservation and scenario coverage unchanged.
- `02e4d2c9` **no-sync port of the donation** (this fork's rewrite, the reason it could merge):
  the donated barrier (`torch.cuda.synchronize`) at continuation-creation time would have
  drained the overlap pipeline once per prefill chunk. The schedule-time call site is already
  stream-ordered in both loops (overlap asserts the scheduler runs on Scheduler.stream,
  bracketed by the loop's wait pair; normal_loop's predecessor batch is drained), so
  `cache_req(schedule_time=True)` skips the barrier; drain/finish commits keep it.
- Deploy precondition for the next image: the donation path is the hybrid prefill hot path -
  before it serves traffic, run a GPU saturation correctness probe (wrong-answer detector
  under concurrent load) and a TTFT A/B against the previous image, per the GPU watchdog
  rules (Xid delta check after).
