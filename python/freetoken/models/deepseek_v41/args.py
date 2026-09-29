"""DeepSeek-V4.1-Flash hyperparameters.

Field names mirror the authors' ``inference/config.json`` (consumed by the
reference ``ModelArgs``) so the port stays 1:1 with the reference. ``load_args``
reads that file from the checkpoint directory (it ships alongside the weights);
the few runtime knobs (batch / sequence length) are overlaid by the runner.

V4.1 differences from V4: 40 backbone layers (+ 3 MTP/DSpark layers appended in
``compress_ratios``, which therefore has 43 entries), compressor ratios are
{0, 1, 2} (V4: {0, 4, 128}), compressed-KV caches are SHARED across layers
(``kv_source_layers``), the indexer exists on ``index_source_layers`` (a
superset) and reuses the attention q-latent, and two engram layers (1, 14)
carry 384M-row FP8 lookup tables.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, fields
from typing import Literal, Tuple


@dataclass
class DeepseekV41Args:
    # ----- runtime -----
    max_batch_size: int = 1
    max_seq_len: int = 4096
    dtype: Literal["bf16", "fp8"] = "fp8"
    expert_dtype: Literal[None, "fp4"] = "fp4"

    # ----- shape -----
    vocab_size: int = 129280
    dim: int = 5120
    moe_inter_dim: int = 2304
    n_layers: int = 40
    n_mtp_layers: int = 3
    n_heads: int = 64

    # ----- moe -----
    n_routed_experts: int = 384
    n_shared_experts: int = 1
    n_activated_experts: int = 6
    score_func: Literal["softmax", "sigmoid", "sqrtsoftplus"] = "sqrtsoftplus"
    route_scale: float = 1.5
    swiglu_limit: float = 10.0

    # ----- mla -----
    q_lora_rank: int = 1280
    head_dim: int = 512
    rope_head_dim: int = 64
    norm_eps: float = 1e-20
    o_groups: int = 8
    o_lora_rank: int = 1024
    window_size: int = 128
    compress_ratios: Tuple[int, ...] = ()

    # ----- shared-cache attention topology (CSA2) -----
    kv_source_layers: Tuple[int, ...] = (2, 8, 14, 20)
    index_source_layers: Tuple[int, ...] = (2, 8, 14, 20, 24, 28, 32, 36)
    candidate_source_layer: int = 20
    candidate_topk_blocks: int = 2048
    candidate_block_size: int = 8

    # ----- rope / yarn -----
    compress_rope_theta: float = 160000.0
    original_seq_len: int = 65536
    rope_theta: float = 10000.0
    rope_factor: float = 16
    beta_fast: int = 32
    beta_slow: int = 1

    # ----- lightning indexer -----
    index_n_heads: int = 32
    index_head_dim: int = 128
    index_topk: int = 512

    # ----- hyper-connections -----
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6

    # ----- engram -----
    engram_layer_ids: Tuple[int, ...] = (1, 14)
    engram_vocab_size: int = 16000000
    engram_num_embeddings: Tuple[int, ...] = (384006168, 384016682)
    engram_max_ngram_size: int = 4
    engram_pad_id: int = 2
    engram_compressed_vocab_size: int = 99092
    engram_n_heads: int = 8
    engram_head_dim: int = 256

    # ----- dspark (MTP speculative decoding; ported last) -----
    dspark_block_size: int = 5
    dspark_noise_token_id: int = 128799
    dspark_target_layer_ids: Tuple[int, ...] = (37, 38, 39)
    dspark_markov_rank: int = 256
    dspark_n_routed_experts: int = 128
    dspark_n_activated_experts: int = 3

    # ----- vision (not ported; kept so no config key is silently dropped) -----
    vision_n_layers: int = 32
    vision_dim: int = 1024
    vision_n_heads: int = 16
    vision_inter_dim: int = 2816
    vision_patch_size: int = 14
    vision_downsample_ratio: int = 3
    vision_max_n_token: int = 1024
    vision_min_pixels: int = 295936
    vision_max_wh_ratio: float | None = None
    vision_rope_theta: float = 10000.0
    image_token_id: int = 129264

    def __post_init__(self) -> None:
        # JSON lists -> tuple so the dataclass stays hashable / immutable-ish.
        for name in (
            "compress_ratios", "kv_source_layers", "index_source_layers",
            "engram_layer_ids", "engram_num_embeddings", "dspark_target_layer_ids",
        ):
            value = getattr(self, name)
            if isinstance(value, list):
                setattr(self, name, tuple(value))

    @property
    def nope_head_dim(self) -> int:
        return self.head_dim - self.rope_head_dim

    def layer_ratio(self, layer_id: int) -> int:
        """Compressor ratio of a backbone layer (0 = window-only)."""
        return self.compress_ratios[layer_id] if layer_id < len(self.compress_ratios) else 0

    def is_kv_source(self, layer_id: int) -> bool:
        return layer_id in self.kv_source_layers

    def is_index_source(self, layer_id: int) -> bool:
        return layer_id in self.index_source_layers

    def is_engram_layer(self, layer_id: int) -> bool:
        return layer_id in self.engram_layer_ids

    def kv_source_for(self, layer_id: int) -> int | None:
        """The kv-source layer whose shared compressed cache ``layer_id`` reads
        (the most recent source at or before it), or None for window-only layers."""
        source = None
        for s in self.kv_source_layers:
            if s <= layer_id:
                source = s
        return source

    def engram_table_rows(self, layer_id: int) -> int:
        """Row count of the engram table at ``layer_id`` (per-layer prime bucket total)."""
        idx = self.engram_layer_ids.index(layer_id)
        return self.engram_num_embeddings[idx]


def _config_path(model_path: str) -> str:
    """Locate the authors' ModelArgs JSON inside the checkpoint directory."""
    candidates = [
        os.path.join(model_path, "inference", "config.json"),
        os.path.join(model_path, "model_args.json"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"No DeepSeek-V4.1 ModelArgs JSON found under {model_path} "
        f"(looked for inference/config.json)"
    )


def load_args(model_path: str, **overrides) -> DeepseekV41Args:
    """Build :class:`DeepseekV41Args` from the checkpoint's ``inference/config.json``.

    ``overrides`` (e.g. ``max_seq_len``, ``max_batch_size``) take precedence over the
    file, letting the runner size the per-request caches.
    """
    with open(_config_path(model_path)) as f:
        raw = json.load(f)
    valid = {f.name for f in fields(DeepseekV41Args)}
    kwargs = {k: v for k, v in raw.items() if k in valid}
    kwargs.update(overrides)
    return DeepseekV41Args(**kwargs)


__all__ = ["DeepseekV41Args", "load_args"]
