<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/FlashML-org/FreeToken/main/assets/freetoken-logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/FlashML-org/FreeToken/main/assets/freetoken-logo-light.svg">
    <img alt="FreeToken" src="https://raw.githubusercontent.com/FlashML-org/FreeToken/main/assets/freetoken-logo.svg" width=65%>
  </picture>
</div>

<p align="center">
| <a href="https://www.flashml.ai/"><b>Download</b></a> | <a href="https://arxiv.org/abs/2608.16157"><b>Paper</b></a> | <a href="https://join.slack.com/t/flashml/shared_invite/zt-3zpdh5j10-9dwTXrgLiqpVxizhA9KVbA"><b>Developer Slack</b></a> | <a href="https://discord.gg/MsA277cJzZ"><b>Community Discord</b></a> | <a href="https://github.com/FlashML-org/FreeToken/issues/482"><b>Community WeChat</b></a> |
</p>

> **Fork operations branch (`sm89-moe-offload`) — not upstream content**
>
> This branch tracks upstream main plus production customizations for **2x NVIDIA L20 (sm_89, PCIe P2P)**: TP2 + expert offload, serving Qwen3.8-Flash-Next-NVFP4 (qwen4_exp, NVFP4 + PLE disk backend). See git history for the full change list (cherry-picked upstream PRs keep their original authors).
>
> Measured performance. Every row states **where** it was measured: the same engine
> reads 81 or 68 t/s on the same workload depending on the measurement boundary and the
> output window, so a boundary-less number is not reproducible. Protocol in [FORK.md](FORK.md).
>
> | Scenario | Value | Measured at | Notes |
> |---|---|---|---|
> | Single-stream decode, short context | **81 t/s** | server-side gen throughput | +19% vs first deployment (68.07 t/s); client-side SSE on the same condition reads 77-80 |
> | Single-stream decode, 255K context | **71.7 t/s** | client-side SSE, 2047-token window | step time 12.74 -> 13.57 ms, i.e. -6.7% vs short context |
> | 8-way concurrent aggregate | **278-282 t/s** | server-side steady state | client-side end-to-end window reads 198-207 (includes queue ramp + drain, not comparable) |
> | 3-way concurrent aggregate, KV pool at 97% | **113-117 t/s** | client-side SSE, saturated phase | 38-39 t/s per stream; 1.7x single-stream |
> | Cold prefill, 70K tokens | **3641 tok/s** (TTFT 19.1s) | client-side TTFT | 3.6x the ~1025 tok/s first-deployment baseline (1025 -> 2132 -> 3641) |
> | Cold prefill, 255K tokens | **3180-3380 tok/s** (TTFT 75.5-80.3s) | client-side TTFT | prefill peaks near 70K tokens, then declines ~15% by 255K |
> | Prefix-cache-hit TTFT | 1.26s (17K tokens) | client-side | hybrid radix reuse |
> | Short-interaction TTFT | 1.27-1.35s | client-side | prefill JIT already paid at boot warmup |
> | Long-context KV ceiling | 3 x 262144 tokens | engine log `token usage` | a 4th full-context request queues for a free slot; no error, no dropped request |
> | Expert-cache reload under domain churn | **no measurable cost** | - | 6-domain rotation x2 rounds; revisit round identical to first (72.2 vs 72.2 t/s median) |
>
> Server hardware this deployment runs on:
>
> | Component | Spec |
> |---|---|
> | GPU | 2x NVIDIA L20 48GB (sm_89, PCIe P2P, no NVLink) |
> | CPU | Intel Core i9-7980XE (18C/36T; PCIe Gen3 root complex - device links run at 8.0 GT/s) |
> | RAM | 128 GB (hosts the offloaded MoE expert bank) |
> | Storage | 2 TB NVMe (PLE disk backend) |
> | OS | Unraid 7 (Docker) |
>
> Note the platform constraint: the L20 is Gen4-capable, but the i9-7980XE root complex caps links at PCIe Gen3 - measured 12.3 GB/s pinned H2D per GPU, 24 GB/s aggregate across both. All numbers above were measured under this constraint.
>
> Production configuration on this box (2x L20 48GiB, PCIe P2P):
>
> ```bash
> ft serve \
>   --model Qwen3.8-Flash-Next-NVFP4 \
>   --tensor-parallel-size 2 --moe-ep-size 2 --gpu 0,1 \
>   --moe-strategy offload \
>   --ple-backend disk --quant-backend moe.nvfp4=triton \
>   --expert-load serial --memory-ratio 0.90 \
>   --moe-cache-size 8800 \
>   --num-tokens 786432 \
>   --max-running-requests 8 \
>   --cuda-graph-max-bs 8 \
>   --image-max-tokens 4096 \
>   --moe-prefill-hit-d2d
> # env: FREETOKEN_NVFP4_PREFILL_WIDE=1  (wide-load prefill kernel)
> ```
>
> Notes: `--moe-cache-size 8800` holds the top ~8800 of 48x512 experts per GPU in VRAM (measured miss cost ~4 rows/step, so a bigger cache buys nothing); `--num-tokens 786432` sizes the KV pool to ~768K tokens; `--expert-load serial` trades load time for lower peak host RAM; `--image-max-tokens 4096` caps per-image vision tokens.

> Decode progression on this box (same model, same hardware): 68.07 t/s at first deployment -> 69.7 after merging upstream main -> 72.5-74.2 with custom all-reduce -> 78.6-79.9 with admission fusion -> **81 t/s** now (PLE hash fusion + D2H-stall-free GDN conv + prefill warmup batch).
>
> Key customizations: all-backend prefill JIT warmup (#169), wide-load NVFP4 prefill MoE kernel for M>=2048 (int32 wide loads + register unpacking, 3.4-3.6x per layer, bit-identical, env-gated), lm_head projecting only the rows the sampler reads (last-token gather), D2H-stall-free varlen GDN conv (#339), single-kernel PLE n-gram hash (#338), disconnect abort delivery (#222), and more.

Unlock datacenter-class intelligence on the hardware you already own — Run 290B+ frontier MoE models locally on your gaming PC at blistering interactive speeds.

## About

FreeToken is an edge-native Mixture-of-Experts (MoE) serving engine designed for running frontier-scale open-weight models on personal and consumer hardware. It treats heterogeneous edge resources—GPUs, CPUs, host memory, and interconnects—as a unified, elastic inference platform. Its core features include:  

- **Fast Edge-Native Runtime**: Provides efficient MoE serving with bandwidth-adaptive CPU–GPU co-execution ($q^\star$ policy), full-layer double-buffered prefill streaming, global LRU expert caching, graph-compatible execution, and the FTW fast weight format.  
- **Semantic-Aware Caching**: Features semantic anchor checkpoints for recurrent state and KV caches, allowing agentic context edits (e.g., tool calls, thinking blocks) to avoid redundant context recomputation.  
- **Elastic Memory Management**: Supports dynamic, runtime VRAM re-allocation between expert caches and KV memory without engine restarts or weight reloading.  
- **Broad MoE & Ecosystem Support**: Supports frontier open-weight MoE models (e.g., DeepSeek-V4-Flash, Qwen3.6-35B-A3B, GLM-5.2) across various parameter scales and quantization formats (e.g., MXFP4, NVFP4, FP8, BF16), with Anthropic/OpenAI-compatible APIs for seamless integration with real-world coding and tool-calling agents (e.g., Codex, Claude Code, OpenCode, OpenClaw, DeepSeek Harness). 
- **Diverse Consumer Hardware**: Scales across consumer laptops, gaming desktops, and workstation GPUs, with native support for NVIDIA RTX 30, RTX 40, and RTX 50 series GPUs.  

## Getting Started

### Desktop app

Download FreeToken for Windows or Linux at [flashml.ai](https://www.flashml.ai/). It sets the engine up for you and gives you a GUI for running models, chatting, and tuning the engine.

<div align="center">
  <img alt="FreeToken Desktop" src="https://raw.githubusercontent.com/FlashML-org/FreeToken/main/assets/desktop-console.png" width=92%>
</div>

### CLI

Install FreeToken with [uv](https://docs.astral.sh/uv/) (recommended) or pip:

```bash
uv pip install "freetoken[accel]"
```

Or build from source:

```bash
git clone https://github.com/FlashML-org/FreeToken.git && cd FreeToken
uv venv && source .venv/bin/activate
uv pip install -e ".[accel]"
```

For More details:

- [Install FreeToken](https://github.com/FlashML-org/FreeToken/blob/main/docs/install.md)
- [Quick start](https://github.com/FlashML-org/FreeToken/blob/main/docs/quickstart.md)
- [Supported models](https://github.com/FlashML-org/FreeToken/blob/main/docs/models.md)
- [CLI reference](https://github.com/FlashML-org/FreeToken/blob/main/docs/cli.md)
- [Repairing old FTW checkpoints](https://github.com/FlashML-org/FreeToken/blob/main/docs/ftw-hotfix.md)

## Citation

If you use FreeToken for your research, please cite our [paper](https://arxiv.org/abs/2608.16157):

```bibtex
@article{yang2026freetoken,
  title={FreeToken: Efficient Edge-Native MoE Serving with Bandwidth-Adaptive Execution},
  author={Yang, Shuo and Fan, Xiaoze and Pan, Melissa and Xi, Haocheng and Wang, Zhe and Sun, Shanlin and Keutzer, Kurt and Han, Song and Zaharia, Matei and Xu, Chenfeng and Stoica, Ion},
  journal={arXiv preprint arXiv:2608.16157},
  year={2026}
}
```

## Acknowledgment

FreeToken was deeply inspired by [mini-sglang](https://github.com/sgl-project/mini-sglang), and
learned the design and reused code from the following projects:
[SGLang](https://github.com/sgl-project/sglang),
[vLLM](https://github.com/vllm-project/vllm),
[FlashInfer](https://github.com/flashinfer-ai/flashinfer),
[flash-linear-attention](https://github.com/fla-org/flash-linear-attention),
[LightLLM](https://github.com/ModelTC/lightllm) and [llama.cpp](https://github.com/ggml-org/llama.cpp).

## License

[Apache License 2.0](https://github.com/FlashML-org/FreeToken/blob/main/LICENSE).
