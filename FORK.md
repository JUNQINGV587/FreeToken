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

## Precision contract

Zero precision degradation vs upstream: kernel swaps are verified bit-identical or within
1 bf16 ulp (fp32 accumulation-order noise only), with output A/B on the production model.
Deterministic decode (no atomic-order-dependent kernels).
