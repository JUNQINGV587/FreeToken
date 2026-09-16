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
- TP2/owner-EP serving stack from PR #447 lineage + vision TP sharding.

## Precision contract

Zero precision degradation vs upstream: kernel swaps are verified bit-identical or within
1 bf16 ulp (fp32 accumulation-order noise only), with output A/B on the production model.
Deterministic decode (no atomic-order-dependent kernels).
