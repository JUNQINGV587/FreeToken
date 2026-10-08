# FORK: sm_89 (Ada) production line

This fork carries a production branch for **FreeToken on NVIDIA Ada Lovelace (sm_89)** —
developed and validated on 2×L20 48GB (PCIe P2P, no NVLink), TP=2 + owner-local expert
parallelism, serving Qwen3.8-Flash-Next-NVFP4.

The default branch **`sm89-moe-offload`** is the production mainline. Upstream is tracked as
`origin/main` and merged on every sync. Experiments live on `exp/*`.

## What this branch adds over upstream

- `fix(utils)`: **msgpack IPC hardening** (`69fce2ba`) — the shared `_CoalescedUnpacker`
  gets an explicit 1 GiB `max_buffer_size` (`FREETOKEN_MSGPACK_MAX_BUFFER` overrides;
  invalid/non-positive falls back to the default) and `feed()` catches `BufferFull`,
  logs the oversized frame, and swaps in a fresh unpacker: the frame is dropped, the
  worker process survives. Root-caused from a production incident (2026-09-26): a
  ~258k-token request built a >100 MiB IPC frame, past msgpack's default 100 MiB
  ceiling, and the uncaught `BufferFull` killed `tokenize_worker`. The same DoS exists
  upstream. 4 unit tests (`tests/utils/test_mp_coalesced.py::TestOversizedFrames`).
- `fix(moe)`: **offload stats semantics for non-hybrid decode targets** (`557b6fdd`) —
  `stat_fetched` only accumulates on the two hybrid decode paths, so with
  `decode_target=gpu` (the production default) `/v1/stats` reported
  `fetched_per_layer` as a constant 0 and the idle log showed the misleading
  `cpu_per_layer == missing_per_layer`. New `_effective_fetched(missing)`: gpu target
  counts every miss as one H2D fetch (`fetch_rate` 1.0), cpu target counts none,
  hybrid keeps the real PCIe/CPU split. 3 unit tests.
- `feat(qwen4_exp)`: **SmoothQuant-style attention alpha consumer** (KV8k-style
  checkpoints): the engine scans safetensors headers for `self_attn.k_alpha` /
  `v_alpha` and the layer binds them iff present (strict loading fails loudly on a
  mismatch either way). `v_alpha` scales the raw v_proj output; `k_alpha` applies at
  `--attn-alpha-apply {pre_norm,post_norm,post_rope}` (default pre_norm, exactly
  foldable into the k_proj weight rows offline). Consumer side only -- the
  producer/calibration side waits for real KV8k artifacts. 8 synthetic-alpha tests.
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
- `feat(moe)`: **prefill data-source split by pin layout** (dsv41 port plan ②a). The overlap prefill
  fetch plan now names its two sources explicitly: RAM-pinned rows (HostBank pinned-row H2D) vs disk
  rows (preadv), split by the pinned-row layout (`row_map`) rather than by expert id, so a learned pin
  set and the default prefix pin go through the same test. `FREETOKEN_PREFILL_PIN_SOURCE=1` (default
  off; off is byte-identical to the previous plan) moves the pin share's H2D from the ring's blanket
  prefix copy into `fetch_routed_into` itself -- routed pinned rows only, gathered from the host banks
  and scattered into the borrowed buffer, while the ring skips its prefix copy. Unrouted rows stay
  stale either way: the grouped GEMM never gathers them, same argument as the unrouted disk tail.
  `/v1/stats` gains the split of the routed demand: `prefill_pin_rows`/`prefill_pin_bytes` (H2D
  payload) vs `prefill_disk_rows`/`prefill_disk_bytes` (preadv payload), plus `prefill_pin_source`
  echoing the flag. Validation (GPU battery, deferred): with the pin file mounted (~31% pin share at
  budgets 64+56/384) the cold24k "盘读 GiB" column must drop by that share vs the 141.5 GiB baseline;
  targets cold24k <= 80 s (>= 300 tok/s).
- TP2/owner-EP serving stack from PR #447 lineage + vision TP sharding.
- `feat(moe)`: **graph-doorbell fetch hardening — observability + loud failure, default-on**
  (`f9947feb60`/`915beb65ff`) — the disk-fetch bridge inside CUDA graphs gets the
  dsv41-caliber telemetry and the cpu_executor-style health contract. (1) The spin
  kernel times itself with `%globaltimer` into a device stats word, mirrored to
  pinned host memory by one captured D2H node per replay, so `/v1/stats`'s
  `disk_tier` block carries `doorbell_spins` / `doorbell_wait_ms` /
  `doorbell_wait_ms_peak` (GPU-side wait, the dsv41 `gpu_wait_ms` caliber) next to
  the W22 host-side `doorbell_requests/rows/bytes/host_ms`, plus
  `doorbell_timeouts`. (2) A host watchdog (`FT_GRAPH_FETCH_TIMEOUT_S`, default
  10 s) turns a wedged service thread into a counted timeout + an engine-visible
  `raise_if_unhealthy` error between replay and sampling (poison the ack, fail the
  step) instead of the 22-minute silent freeze of 2026-10-03; count > k_max
  (= cuda_graph_max_bs x topk) takes the same loud path — refuse, count, poison,
  raise — because no eager fallback exists inside a replayed graph. **Default-on
  decision: ON.** Production (`/data/build/start_ftprod_full0913tp2.sh`:
  `--moe-disk-tier auto --cuda-graph-max-bs 8`) has never set
  `FT_GRAPH_FETCH_OFF`, and the 20261002 battery's graphs4 arm ran with the
  doorbell hot, so the doorbell IS the graph-mode disk path; the env stays as a
  debug escape hatch (eager `fetch_pending` fallback), read once per capture set.
  Baseline anchor (runs/20261002-tune/results/_table.md; decode300 median t/s,
  engine-cumulative t/s, wall): graphs4 2.73 / 1.989 / 147.1 s vs graphs0
  2.16 / 1.678 / 180.3 s → +26.4% decode300; cold-24K wall 156.2 s vs 148.5 s
  (spin overhead visible on cold prefill); disk read volume identical at 141.9
  GiB, as expected — the doorbell changes WHERE reads are served, not how many.

## Working on this fork (git)

- **`origin` = upstream FreeToken (read-only for us). `fork` = our fork — the only
  remote we push to.** Repo `AGENTS.md` forbids pushing or PR-ing to upstream on the
  user's behalf; upstream-bound contributions are the user's own decision.
- Production mainline branch: **`sm89-moe-offload`** on `fork`. **It is the only working
  branch (2026-09-27 user decision): work directly on it and push to `fork`; do not create
  `exp/*` or other work branches unless the user explicitly asks.** The legacy `exp/*`
  branches still on `fork` are frozen history, not active lines.
- Cherry-picks keep the original author; our own commits use the fork identity
  (see /data/AGENTS.md for the exact `user.name`/`user.email` and the vault-backed
  credential helper).
- Never rewrite pushed history on `sm89-moe-offload` — production images pin commit
  markers (`/opt/FREETOKEN-DEPLOYED-COMMIT`) that must stay reachable.

## Testing on this fork

- **Test images**: since 2026-09-27 pytest ships **in** the image (base
  `/data/build/ftcontainer/full0913/Dockerfile` installs it; decision: merge the
  test toolbox and Xid watchdog roles into the production image chain instead of
  maintaining separate images). Until the next full rebuild, overlays keep
  building a `*-test` tag (same image plus `pip install pytest`, e.g.
  `freetoken:sm89-delta28-test`) and the production tag stays pytest-free.
  Run suites inside the image that will ship, not against a host checkout.
- **CPU suites** (`tests/models/qwen4_exp tests/engine tests/server tests/moe
  tests/utils`, `CUDA_VISIBLE_DEVICES=`): expect **all green** (1425 passed at delta26).
  `tests/models` on CPU skips the model-building weight tests — the
  `create_model → sgl_kernel common_ops` import chain has no CPU variant — so
  **210 passed / 293 skipped / 0 failed** is the expected CPU shape there.
- **GPU suites**: wrap every GPU run in `/data/ops/gpu_test_guard.sh docker run --rm
  --gpus all ...` (co-residency with production is fine) and check
  `dmesg | grep -icE "Xid"` afterwards — **0** is the only acceptable value.
- **Full `tests/models` GPU baseline**: **323 passed / 2 known-failed**
  (`test_glm_dsa.py::test_indexer_matches_hf_reference` ×2 — the failure is inside the
  transformers HF reference, not this tree; GLM is not served here). Anything beyond
  those two is a regression.
- **Ad-hoc smoke**: the `freetoken-testbox` container (renamed from freetoken-p0test 2026-09-27; currently `sm89-delta28-test`,
  repo mounted at `/ft`) is the standing tool box; `docker exec freetoken-testbox ...`
  with `PYTHONPATH=/ft/python` tests the live checkout (note: the bare binary is
  `python3`, no `python` alias).
- **Xid watchdog channel**: `/data/ops/gpu_test_guard.sh` reads host dmesg via a
  throwaway `--privileged` container from the **production image** (resolved
  dynamically from `freetoken`, fallback newest `sm89-delta*`) — no
  separate watchdog container or foreign image since 2026-09-27.

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

**2026-09-23 evening - fix3 donation REVERTED (production incident)**: delta21 (the image
carrying `ba109809`/`f231927d`/`02e4d2c9`) passed every gate - image self-tests (2165 CPU+GPU
green), divergence probe (the fix works: cached-token 0 -> 16,320, re-prefill -34%, TTFT
-27%), floor/262K/concurrency A/B (zero regression over 4 samples) - and then **died under
the saturation correctness probe**: 15 concurrent 12K-token divergent prefills plus 2
background decode streams killed both TP scheduler ranks with CUDA illegal memory access
(Xid 31 MMU Fault, VIRT_READ, same address pattern on both ranks = deterministic code path),
surfaced at the drain-time donate barrier. Production rolled back to delta20 (verified green)
within minutes. The donation trio is reverted from this branch (`41185f51`, `ce503db1`,
`20b6cd32`); the work lives on `exp/vektory-fix3-eval` pending RCA (sanitizer repro on an
experiment arm; the barrier-vs-logic discrimination experiment is designed). The four
low-risk picks (snapshot_lru eviction order, --linear-state-cache-ratio, invariant comments,
cross-conversation test) REMAIN merged - they are not implicated: the crash path (donation +
hit-restore of donated boundaries) does not exist without the reverted commits.
Detail: `research/notes/freetoken/202609-freetoken-upstream-research.md` section 12.4.

**2026-09-26** (base `d1713b75`, merge `origin/main` `0d652e73` + two PRs):

- `origin/main` `0d652e73` (4 commits). **#546** (overlap token limits / cache safety) was
  already carried in this fork -- scheduler `hit_length` used the `max_device_len` form,
  `OffloadMoeCache` already zero-filled `bank_caches` (with the more detailed
  ``0 * NaN`` rationale, kept), and `cache_status._swa` already reported the usable
  floors; the merge conflicts there were context drift and resolved to our side.
  **#548** extracts the PLE disk row store into `kernel/row_store.py` and renames the
  compiled extension `_ple_store` -> `_row_store` (behavior-preserving; rebuilt
  in-place, `tests/kernels/test_row_store.py` + `test_ple_disk.py` pass on GPU).
  **#132** ROCm RDNA3/RDNA4 runtime foundation: taken now that it is upstream, mostly
  additive and guarded (user decision). Two conflicts resolved by keeping ours:
  `pynccl.py` keeps the NCCL-wheel absolute-path logic over their bare `-lnccl`, and
  `test_pinned_tensor.py` keeps the measured 2xL20 UVA contract (the raw
  `host_device_ptr` rejects pageable memory even under UVA) over #132's
  degenerate-identity assumption. #545 (pin flashinfer) came along with the merge.
- **#551** higher uv install timeout/retries. Install-only, one line.
- **#553** cumulative prefill/decode timing on `/v1/stats` (cherry-picked, `stats.py`
  conflict resolved as a union): a poller can now diff two polls into speeds, which the
  5 s sliding windows cannot express. Its `cached_prompt_tokens_total` mirrors this
  fork's existing `cached_tokens_total` (same `UserReply.cached_tokens` source); both
  names are emitted so fork tooling and upstream alignment both hold.

Tests after the sync: CPU 10 passed (`test_rocm_arch`, `test_stats_timing`,
`test_rocm_launch_kwargs` skips off-ROCm); GPU 15 passed (`test_row_store`,
`test_ple_disk`, `test_pinned_tensor`), 0 Xid; regression `tests/moe+engine+utils+server`
1316 passed / 128 skipped / 0 failed.

## Port-branch entries pending merge (2026-10-07)

Entries for the four `port/*` experiment branches, written at the file tail on
purpose: the `m1-doorbell` and `m3-prefill-source` entries sit mid-file in their
own branches, so keeping these in a separate tail section keeps the six-way merge
conflict-free (union-append either way). Validation status for all four: CPU
suites green on their branches; GPU validation is the deferred battery
(`research/notes/engines/202610-port-branches-review.md` §6).

- `feat(sched)`: **decode-share QoS two-parter** (`34b99ef223`, `3baea0f7e7`,
  `port/m1-qos`) — dsv41's time-based decode share: under contention the
  scheduler reserves a configurable share of each scheduling window for decode
  so long prefills cannot starve running decodes, plus a contention chunk cap
  that shrinks prefill chunks while requests wait. Switches:
  `FREETOKEN_DECODE_SHARE` (fraction; unset = off = byte-identical scheduling),
  `FREETOKEN_DECODE_SHARE_CAP_S` (allowance cap, default 10 s),
  `FREETOKEN_LONG_PREFILL_WHEN_WAITING` (contention chunk cap in tokens).
  Newly arriving requests bypass the throttle (`chunked_req is None`), so a
  fresh prompt never queues behind its own allowance. Snapshot published on
  `/v1/stats` under `sched_qos`. 22 CPU tests on branch; latency A/B defers to
  the battery.
- `feat(moe)`: **CPU tier -- RAM-resident miss compute** (`6c7809c253`,
  `46ccb798c1`, `481fe8570f`, `677be89cb9`, `ccb259a4c6`, `4fe59ce9b1`,
  `port/m2-cpu-tier`) — when the disk tier stages a decode-step miss whose row
  already sits in the pinned RAM set, a Triton split kernel classifies the
  routed entries and hands the CPU-ok experts to a C++ `CpuTierService`
  (pinned-pool GEMV, ISA-tiered) instead of re-fetching them over PCIe; a
  combine kernel spin-waits on a doorbell and folds the CPU partials back onto
  the GPU GEMM output. Switch: `--moe-cpu-tier` (default off); cost model via
  `FREETOKEN_CT_*` envs. The split kernel also records the cumulative [L,E]
  per-expert route/miss counters that feed item ⑤'s online admission
  (`port/m4-perexpert-counts`). Pure-CPU protocol/accounting tests on branch;
  kernel-vs-mirror cross-check and the end-to-end decode win defer to the
  battery.
- `feat(moe)`: **prefill fat-GEMM experiment arm (②c)** (`f86be0fd81`,
  `f3aa6bfc8c`, `d684bafc0b`, `7a4a1a13cc`, `port/m3-fat-gemm`) — prefill apply
  can route through a fat-GEMM dispatch that dequantizes contiguous expert
  banks and runs one wide GEMM instead of per-expert gathers, behind a decision
  boundary (`FREETOKEN_PREFILL_FAT_GEMM`, default 0 = off; row threshold
  `MIN_ROWS` default 32 to re-tune on the target). Ships with a fidelity gate
  (`python -m freetoken.moe.prefill_fat_gemm`): bf16 top-1 agreement ≥ 0.98
  and relative error ≤ 0.10 must hold before any default-on is considered, and
  the run ledger surfaces in `/v1/stats`. Note for the A/B battery: the ②c
  dequant scratch peaks at ~150 MB per expert (V4.1 geometry) — measure the
  real peak in the battery arm; it sizes whether ②c can coexist with the
  borrowed prefill ring on 48 GB cards.
- `feat(moe)`: **VRAM-elastic guard + online admission skeleton (item ⑤)**
  (`aae62c2f8b`, `58d3c7093c`, `d8acd60050`, `port/m4-cache-admission`;
  per-expert signal: `port/m4-perexpert-counts`) — pure host-side policy for
  elastic release/rewarm of MoE cache slots under prefill pressure and
  route-count online admission that re-pins hot experts in the RAM set.
  Switches: `FREETOKEN_VRAM_ELASTIC`, `FREETOKEN_ONLINE_ADMISSION` (both
  default 0 = off = byte-identical cache behavior). Policy-only skeleton: the
  guard construction, pin-layout handoff, stats passthrough and (as of the
  per-expert-counts branch) the [L,E] route/miss signal are landed; the
  engine/scheduler call sites and the CUDA apply (unmap + empty_cache +
  re-add) are battery-#12 wiring. Until then `/v1/stats` carries
  `wired: false` next to `enabled: true` so monitoring cannot misread an idle
  policy as a working one. CPU tests cover release scope/gates, admission
  boundaries (strict threshold, hysteresis, batch cap, period, decay) and the
  self-sourced counter feed.

## Port2-branch entries (2026-10-07, merged into sm89-moe-offload)

Entries for the three `port2/*` branches, kept in their own tail section for
the same union-append reason as the batch above. Validation status for all
three: CPU suites green (`research/notes/engines/20261007-port2-branches-review.md`;
349 passed / 9 pre-existing environment failures on the merge tip); every GPU
item is deferred to the combined battery (§G of that review) -- do not enable
any flag before it.

- `feat(moe)`: **CPU tier rescue -- pre-ensure claim + reuse filter** (`559d95330e`,
  `3b35fe8aa6`, `1403201bb0`, `port2/cpu-tier-rescue`) — regression rescue for
  the decode collapse (2.23 t/s vs the >=8 target): a `split_pre` kernel claims
  CPU-ok experts BEFORE `ensure_experts` (slot-0 sentinel protocol kills the
  eviction churn), a reuse-score filter keeps one-shot misses on the PCIe
  path, and `gpu_wait_ms` is measured with `%globaltimer` instead of the ~300x
  off cross-step clock. Switches (all default off = byte-identical legacy
  split): `FREETOKEN_CT_PRE_ENSURE` (0), `FREETOKEN_CT_REUSE_MIN` (-1 = no
  filter), `FREETOKEN_CT_XSTEP_W` (1.0 = neutral weight). Note for the A/B
  battery: the B9 ghost-pick fix changes the pick PUBLICATION order to pure
  routed order, so the C++ fp32 accumulation order over multi-pick tokens
  changes -- output is exact-arithmetic equivalent but NOT bitwise identical;
  compare with rel-RMS / `torch.allclose`, never `torch.equal`. 28 CPU tests
  on branch (incl. TRITON_INTERPRET kernel twins); sanitizer + decode wins
  defer to the battery. **B10c follow-up** (`5c120c091b`, `92b58deccd`,
  `3464e23c57`, `port2/owner-pre-ensure`): the pre-ensure claim is now wired
  into the OWNER-EP decode path too (`_decode_owner` previously fell through
  to the legacy split, so `PRE_ENSURE=1` was a no-op in production ep>1
  geometry). `split_pre_owner` runs the same kernel on the raw global route
  BEFORE `ensure_route[_graph]` with a `GLOBAL_START` ownership filter --
  only this rank's owned experts are classified/claimed (remote entries feed
  the cost model's GEMM term like hits but are never claimed, sentinel'd,
  weight-zeroed, or counter-bumped); the reuse filter (piece C) applies
  identically. Same env, default off = byte-identical legacy owner split.
- `feat(moe)`: **learned block bulk prefetch for cold prefill** (`5efd1173d7`,
  `e0c9fd93e3`, `port2/bulk-prefetch`) — cold prefill leaves the disk idle in
  the GEMM/attention windows (~6.4% honest ceiling); a learned per-layer
  expert score distribution (never an identity predictor) is rolled into
  top-k candidates per layer, chunked into cross-expert read blocks, and read
  into refcounted pinned slabs behind the running GEMM. Switches (master
  default 0 = off = byte-identical; no prediction file or DS-FP4 convert mode
  force it off with a log line): `FREETOKEN_PREFILL_BULK_PREFETCH` (0),
  `FREETOKEN_PREFILL_BULK_PREDICT` (unset = disabled),
  `FREETOKEN_PREFILL_BULK_DEPTH` (8), `FREETOKEN_PREFILL_BULK_TOPK` (32),
  `FREETOKEN_PREFILL_BULK_MIN_FRAC` (0), `FREETOKEN_PREFILL_BULK_BLOCK` (4),
  `FREETOKEN_PREFILL_BULK_SLABS` (4), `FREETOKEN_PREFILL_BULK_WORKERS` (2).
  22 new CPU cases on branch; cold24k TIMELINE-gap win defers to the battery.
- `feat(moe)`: **admission apply -- hot-swap RAM rows in place** (`75a4d00c1e`,
  `becc0b8962`, `port2/admission-apply`) — wires the online-admission skeleton
  to a real apply: each evaluation's swap plan reads the challenger row from
  disk (prepare, per-pair failure keeps the old layout), drains the device
  once, then memcpys into the incumbent's bank row and trades the two rows'
  slots in the pin maps in place (captured graphs baked the pointers, so
  rebinding is never used; GPU slot contents stay valid because weights are
  immutable). Switches (both default off = evaluation-only skeleton,
  byte-identical cache behavior): `FREETOKEN_ADMISSION_APPLY` (0),
  `FREETOKEN_ADMISSION_WARMUP_STEPS` (64; warmup steps evaluate but never
  commit, counted as skips). 8 new CPU tests on branch (row content,
  failure rollback, warmup gate); APPLY-on cache1100 A/B and the graph-replay
  race check defer to the battery.
