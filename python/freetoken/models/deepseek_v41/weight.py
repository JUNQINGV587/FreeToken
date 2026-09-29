"""Weight loading for DeepSeek-V4.1-Flash (engine path).

  - :func:`iter_weights` streams resident (non-expert) tensors keyed by the model's
    attribute paths (``model.`` + the checkpoint name). ``wo_a`` is dequantized to
    bf16 to match the reference bf16 einsum. The engram embedding TABLES (94 GiB
    each) are not resident weights -- they belong to the EngramTier (NVMe-backed)
    and are skipped here. MTP (``mtp.*``) and vision (``vision.*`` / ``aligner.*``)
    tensors are never touched: the reader enumerates explicitly.
  - :func:`nvfp4_expert_spec` describes the routed ModelOpt NVFP4 experts (U8 pairs
    + per-16 E4M3 scales + fp32 global) for the expert banks / disk tier, the same
    dialect the Qwen production models use.
"""

from __future__ import annotations

import json
import os
import re

import safetensors
import torch

from freetoken.models.loader import drop_page_cache
from freetoken.models.nvfp4_banks import Nvfp4ExpertSourceSpec

from .args import load_args


class _ShardReader:
    def __init__(self, folder: str, weight_map: dict, device):
        self._folder = folder
        self._weight_map = weight_map
        self._device = str(device)
        self._handles: dict[str, object] = {}

    def has(self, name: str) -> bool:
        return name in self._weight_map

    def get(self, name: str) -> torch.Tensor:
        shard = self._weight_map[name]
        handle = self._handles.get(shard)
        if handle is None:
            handle = safetensors.safe_open(
                os.path.join(self._folder, shard), framework="pt", device=self._device
            ).__enter__()
            self._handles[shard] = handle
        return handle.get_tensor(name)

    def close(self) -> None:
        for shard, handle in self._handles.items():
            try:
                handle.__exit__(None, None, None)
            except Exception:
                pass
            drop_page_cache(os.path.join(self._folder, shard))
        self._handles.clear()


def _weight_map(model_path: str) -> dict:
    with open(os.path.join(model_path, "model.safetensors.index.json")) as f:
        return json.load(f)["weight_map"]


def _dequant_fp8_block(weight: torch.Tensor, scale: torch.Tensor, block: int = 32) -> torch.Tensor:
    """Dequantize block-scaled FP8 (e4m3) to bf16.

    V4.1 fp8 linears carry e8m0 scales over 32x32 blocks (V4 used 128x128);
    ``value = 2^(code-127)`` (Triton FP8 GEMM convention). Used for ``wo_a``
    to match the reference's bf16 einsum.
    """
    n, k = weight.shape
    codes = scale.view(torch.uint8).to(torch.float32)
    s = torch.exp2(codes - 127.0)
    s = s.repeat_interleave(block, dim=0).repeat_interleave(block, dim=1)[:n, :k]
    return (weight.to(torch.float32) * s).to(torch.bfloat16)


def iter_weights(
    model_path: str,
    device,
    *,
    include_moe_experts: bool = True,
    include_non_moe: bool = True,
):
    """Stream resident (non-expert) weights as ``(name, tensor)`` keyed to engine params.

    Routed NVFP4 experts come from the offload cache, so ``include_moe_experts`` must be
    False (DeepSeek-V4.1 only runs ``--moe-strategy offload``). Tensors yielded in
    checkpoint dtype (fp8 + e8m0 preserved); ``wo_a`` dequantized to bf16 to match the
    reference einsum.
    """
    if include_moe_experts:
        raise ValueError(
            "DeepSeek-V4.1 routed experts are served from the offload cache; "
            "run with --moe-strategy offload (include_moe_experts must be False)."
        )
    if not include_non_moe:
        return

    args = load_args(model_path, max_batch_size=1)
    reader = _ShardReader(model_path, _weight_map(model_path), device)

    def get(name: str) -> torch.Tensor:
        return reader.get(name)

    def linear(src: str, dst: str):
        yield f"{dst}.weight", get(f"{src}.weight")
        # fp8 linears declare the e8m0 block scale under the quant method's role name
        if reader.has(f"{src}.scale"):
            yield f"{dst}.weight_scale_inv", get(f"{src}.scale")

    try:
        yield "model.embed.weight", get("embed.weight")
        yield "model.norm.weight", get("norm.weight")
        yield "model.head.weight", get("head.weight")
        # NB: V4.1 has no top-level hc_head_* (V4 had them): the final pre-mix is the
        # fixed one-hot identity and the last layer's hc_pre does the contraction.

        for L in range(args.n_layers):
            a = f"layers.{L}.attn"
            m = f"model.{a}"
            yield from linear(f"{a}.wq_a", f"{m}.wq_a")
            yield f"{m}.q_norm.weight", get(f"{a}.q_norm.weight")
            yield from linear(f"{a}.wq_b", f"{m}.wq_b")
            yield from linear(f"{a}.wkv", f"{m}.wkv")
            yield f"{m}.kv_norm.weight", get(f"{a}.kv_norm.weight")
            # wo_a: FP8 in the checkpoint, dequantized to bf16 (reference bf16 einsum).
            yield f"{m}.wo_a", _dequant_fp8_block(
                get(f"{a}.wo_a.weight"), get(f"{a}.wo_a.scale")
            )
            yield from linear(f"{a}.wo_b", f"{m}.wo_b")
            yield f"{m}.attn_sink", get(f"{a}.attn_sink")

            # Compressor: only the kv-source layers own one. ratio 2 (L 2/8/14) is a
            # gated softmax pool (wkv + wgate + norm); ratio 1 (L 20) is a plain
            # projection (wkv + norm, NO wgate).
            if args.is_kv_source(L):
                c = f"{a}.compressor"
                yield f"model.{c}.wkv.weight", get(f"{c}.wkv.weight")
                if args.layer_ratio(L) == 2:
                    yield f"model.{c}.wgate.weight", get(f"{c}.wgate.weight")
                yield f"model.{c}.norm.weight", get(f"{c}.norm.weight")

            # Indexer: every index-source layer scores with its own wq_b (from the
            # attention q-latent) + weights_proj; only kv-source layers also carry the
            # K projection (wk + k_norm) that fills the shared index-K cache.
            if args.is_index_source(L):
                idx = f"{a}.indexer"
                yield from linear(f"{idx}.wq_b", f"model.{idx}.wq_b")
                yield f"model.{idx}.weights_proj.weight", get(f"{idx}.weights_proj.weight")
                if args.is_kv_source(L):
                    yield f"model.{idx}.wk.weight", get(f"{idx}.wk.weight")
                    yield f"model.{idx}.k_norm.weight", get(f"{idx}.k_norm.weight")

            # Engram (layers 1/14): the small resident tensors only. The 94 GiB
            # fp8 embedding table (engram.embed.{weight,scale}) is served by the
            # EngramTier from NVMe, not materialized as a resident parameter.
            if args.is_engram_layer(L):
                e = f"layers.{L}.engram"
                yield from linear(f"{e}.wkv", f"model.{e}.wkv")
                yield f"model.{e}.q_weight", get(f"{e}.q_weight")
                yield f"model.{e}.k_weight", get(f"{e}.k_weight")

            yield f"model.layers.{L}.attn_norm.weight", get(f"layers.{L}.attn_norm.weight")
            yield f"model.layers.{L}.ffn_norm.weight", get(f"layers.{L}.ffn_norm.weight")

            g = f"layers.{L}.ffn.gate"
            yield f"model.{g}.weight", get(f"{g}.weight")
            yield f"model.{g}.bias", get(f"{g}.bias")
            yield f"model.{g}.bias_vl", get(f"{g}.bias_vl")
            for proj in ("w1", "w2", "w3"):
                src = f"layers.{L}.ffn.shared_experts.{proj}"
                yield from linear(src, f"model.{src}")

            for nm in (
                "hc_attn_fn", "hc_ffn_fn", "hc_attn_base",
                "hc_ffn_base", "hc_attn_scale", "hc_ffn_scale",
            ):
                yield f"model.layers.{L}.{nm}", get(f"layers.{L}.{nm}")
    finally:
        reader.close()


# --------------------------------------------------------------------------------------
# Routed ModelOpt NVFP4 experts (U8 pairs + per-16 E4M3 scales + fp32 global).
# --------------------------------------------------------------------------------------
_EXPERT_KEY_RE = (
    r"^layers\.(?P<layer>\d+)\.ffn\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>w1|w2|w3)\.(?P<kind>{kinds})$"
)
_PROJ_ROLE = {"w1": "gate", "w3": "up", "w2": "down"}


def nvfp4_expert_spec(model_path: str, config) -> Nvfp4ExpertSourceSpec:
    """The per-expert NVFP4 layout: ``layers.N.ffn.experts.E.{w1,w2,w3}.{kind}``.

    The checkpoint already uses the canonical ModelOpt tensor names
    (weight / weight_scale / weight_scale_2); every backbone layer is MoE, and the
    MTP experts (old MXFP4 dialect) are out of scope here. The names are hardcoded
    rather than read off the process quant config: the top-level
    ``quantization_config`` declares ``quant_method=fp8`` (the dense dialect
    claims it) while only the expert containers are NVFP4, so the global config
    is the wrong source for the expert layout.
    """
    # canonical ModelOpt stored name -> the expert bank reader's tensor kind
    kind_map = {"weight": "weight", "weight_scale": "weight_scale", "weight_scale_2": "weight_scale_2"}
    return Nvfp4ExpertSourceSpec(
        key_pattern=re.compile(_EXPERT_KEY_RE.format(kinds="|".join(map(re.escape, kind_map)))),
        proj_to_role=dict(_PROJ_ROLE),
        layer_to_bank=lambda layer, config: layer,  # every backbone layer is MoE
        desc="DeepSeek-V4.1 NVFP4 experts (modelopt)",
        kind_map=kind_map,
        global_reciprocal=False,
    )


__all__ = ["iter_weights", "nvfp4_expert_spec"]
