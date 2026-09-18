"""Decode-time GEMV weight concat fusion (Qwen3.8-Flash-Next), gated by FREETOKEN_GEMV_CONCAT.

Two load-time fusions share this module's second-stage fuser:

* F1 ``mlp.router_gate_up``: router [512, 2560] (TP-replicated) + shared-expert gate|up
  (TP column-parallel) -> one [512 + 2*local_inter, 2560] buffer per rank.
* F2 ``self_attn.qkv_index_proj``: qkv (TP-sharded heads) + indexer index_qk [640, 2560]
  (TP-replicated) -> one [local_qkv + 640, 2560] buffer per rank.

Bitwise contract (measured on torch 2.11.0+cu130, L20): a row-concatenated F.linear is
torch.equal to the separate GEMVs only at M==1; at M>=2 cuBLAS switches algorithm and the
reduction order changes (~4 bf16 ulp). The fused modules therefore dispatch on M: M==1
runs the single fused GEMV, M>1 runs the original two GEMVs on contiguous row slices of
the fused buffer, which is structurally identical to the unfused baseline.

The fuser runs AFTER _DenseFuser (shard-then-fuse order is unchanged): the router and
indexer segments are replicated on every rank while the gate/up and q/k/v segments arrive
already rank-local, so the concatenated buffer is exactly what the fused module builds.

FTW checkpoints replay whatever iter_weights emitted at conversion time, so
FREETOKEN_GEMV_CONCAT must be set identically for `ft convert` and for the serve that
loads the FTW; a mismatch fails loudly in load_state_dict (missing/unexpected keys).
"""

from __future__ import annotations

import os

import torch

_ENV = "FREETOKEN_GEMV_CONCAT"


def gemv_concat_env() -> bool:
    return os.getenv(_ENV, "0").strip().lower() in ("1", "true", "yes", "on")


def f1_enabled(config) -> bool:
    """F1 needs the shared expert served in bf16 (quantized dense layouts keep their own kernels)."""
    return gemv_concat_env() and getattr(config, "dense_quant", "none") == "none"


def f2_enabled(config) -> bool:
    return gemv_concat_env() and getattr(config, "attn_quant", "none") == "none"


# (emitted-name suffix, fused-name suffix, segment index). The fused name shares the
# layer stem of the part name, so both parts of a pair map to one buffer.
_PARTS = (
    (".mlp.gate.weight", ".mlp.router_gate_up.weight", 0),
    (".mlp.shared_expert.gate_up_proj.weight", ".mlp.router_gate_up.weight", 1),
    (".self_attn.qkv_proj.weight", ".self_attn.qkv_index_proj.weight", 0),
    (".self_attn.indexer.index_qk_proj.weight", ".self_attn.qkv_index_proj.weight", 1),
)


class _GemvConcatFuser:
    """Second-stage, cross-parent concat of _DenseFuser's emitted projections.

    ``f1``/``f2`` gate the mlp / self_attn pairs independently, matching the modules the
    model built (a quantized dense path keeps its native kernels and stays unfused).
    """

    def __init__(self, *, f1: bool, f2: bool) -> None:
        self.parts = tuple(
            (part, fused, idx)
            for part, fused, idx in _PARTS
            if (f1 if fused.startswith(".mlp.") else f2)
        )
        self.buf: dict[str, dict[int, torch.Tensor]] = {}

    def fuse(self, name: str, tensor: torch.Tensor) -> list[tuple[str, torch.Tensor]] | None:
        """Buffer a part; return the fused ``[(name, tensor)]`` once its pair is complete, ``[]`` while incomplete, ``None`` if ``name`` is not a part."""
        for part_suffix, fused_suffix, idx in self.parts:
            if not name.endswith(part_suffix):
                continue
            if tensor.dtype is not torch.bfloat16:
                raise ValueError(
                    f"{name} is {tensor.dtype}: FREETOKEN_GEMV_CONCAT supports the bf16 "
                    "dense path only; unset it for quantized-dense checkpoints"
                )
            fused = name[: -len(part_suffix)] + fused_suffix
            slots = self.buf.setdefault(fused, {})
            slots[idx] = tensor
            if len(slots) < 2:
                return []
            del self.buf[fused]
            return [(fused, torch.cat([slots[0], slots[1]], dim=0))]
        return None

    def finish(self) -> None:
        assert not self.buf, f"Incomplete GEMV concat fusions: {sorted(self.buf)}"


__all__ = ["gemv_concat_env", "f1_enabled", "f2_enabled", "_GemvConcatFuser"]
