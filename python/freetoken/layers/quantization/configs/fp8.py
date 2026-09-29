from __future__ import annotations

from typing import Any, ClassVar

from ..names import is_routed_expert, name_set, substr_set
from ..registry import register_dialect
from ..scheme import QuantKind, QuantScheme
from ..scheme import FP8_BLOCK, FP8_BLOCKS, fp8_block_scheme, fp8_tensor_scheme, mxfp4_scheme, nvfp4_scheme
from .base import QuantConfig, Stored, cfg_get


@register_dialect
class Fp8BlockConfig(QuantConfig):
    """HF ``quant_method: fp8`` (DeepSeek-V3 style 128x128 block scales) plus the DeepSeek-V4 e8m0 / fp4-expert variant.

    DeepSeek-V4.1 ships the same dialect one block smaller: ``weight_block_size [32, 32]``
    with ``scale_fmt ue8m0`` and, in the same file, ModelOpt NVFP4 routed experts
    (``moe_quant_algo NVFP4`` / ``quantized_layers.*.ffn.experts.quant_algo NVFP4``).
    Its ``ignore`` list names the modules kept OUT of that NVFP4 group -- they are the
    fp8 ones, so it is deliberately not read as ``modules_to_not_convert``."""

    dialect = "fp8"

    SCHEMES: ClassVar[dict[str, QuantScheme]] = {
        "BLOCK": fp8_block_scheme("float"),
        "BLOCK_E8M0": fp8_block_scheme("e8m0"),
        # HF ``modules_to_convert``: a table (Qwen3.8-Flash-Next PLE) stored e4m3 with one scalar scale
        "TABLE": fp8_tensor_scheme("float"),
        "EXPERT_MXFP4": mxfp4_scheme(),
        # DeepSeek-V4.1's routed experts are a ModelOpt NVFP4 build inside an fp8 config
        "EXPERT_NVFP4": nvfp4_scheme(input_scale=True),
    }
    # transformers' fp8 names; DeepSeek-V4's e8m0 export calls every scale ``scale`` (see storage)
    STORAGE: ClassVar[dict[QuantKind, dict[str, str | Stored]]] = {
        QuantKind.FP8_BLOCK: {"weight": "weight", "weight_scale_inv": "weight_scale_inv"},
        QuantKind.FP8_TENSOR: {"weight": "weight", "weight_scale": "weight_scale"},
        QuantKind.MXFP4: {"weight": "weight", "weight_scale": "scale"},
        QuantKind.NVFP4: {
            "weight": "weight",
            "weight_scale": "weight_scale",
            "weight_global": "weight_scale_2",
            "input_scale": "input_scale",
        },
    }

    def __init__(self, q: dict[str, Any], hf_config: Any = None, *, name_map=None, unquantized=()):
        super().__init__(name_map, unquantized)
        block = tuple(int(x) for x in (q.get("weight_block_size") or ()))
        if q.get("weight_per_tensor") or len(set(block)) > 1 or (block and block[0] not in FP8_BLOCKS):
            raise NotImplementedError(f"fp8 checkpoint with weight_block_size={block} per_tensor={q.get('weight_per_tensor')} is not supported; only square {FP8_BLOCKS} blocks are")
        if not block:
            raise NotImplementedError("fp8 checkpoint without weight_block_size (per-tensor scales) is not supported; only square 32x32 / 128x128 blocks are")
        self.fp8_block = block[0]
        # transformers skips lm_head when the checkpoint gives no list
        not_convert = tuple(q.get("modules_to_not_convert") or ("lm_head",))
        self.not_convert = name_set(not_convert)
        self.not_convert_substr = substr_set(not_convert)
        self.convert_tables = name_set(tuple(q.get("modules_to_convert") or ()))
        self.e8m0 = str(q.get("scale_fmt") or "").lower() == "ue8m0"
        # the export keeps it inside the quant dict (V4.1); older files hoist it to the model config
        self.expert_fp4 = str(q.get("expert_dtype") or cfg_get(hf_config, "expert_dtype") or "").lower() == "fp4"
        # DeepSeek-V4.1: fp8 everywhere, ModelOpt NVFP4 experts -- ``moe_quant_algo`` is the
        # top-level default, ``quantized_layers`` the explicit per-module list (an export
        # carries either). The list wins where it speaks: it names the 40 backbone expert
        # blocks, and leaves the MTP draft experts (the older per-32 e8m0 layout) alone.
        algo = str(q.get("moe_quant_algo") or "").upper()
        layers = q.get("quantized_layers") or {}
        self.expert_algo_global = algo
        self.quantized_experts = {
            str(name): str((entry or {}).get("quant_algo") or "").upper()
            for name, entry in (layers.items() if isinstance(layers, dict) else ())
            if "experts" in str(name)
        }
        self._expert_algo_cache: dict[str, str] = {}
        # the block size is per checkpoint, so the block schemes are instance-level
        self.SCHEMES = dict(type(self).SCHEMES)
        self.SCHEMES["BLOCK"] = fp8_block_scheme("float", self.fp8_block)
        self.SCHEMES["BLOCK_E8M0"] = fp8_block_scheme("e8m0", self.fp8_block)

    def storage(self, scheme: QuantScheme) -> dict[str, Stored]:
        names = super().storage(scheme)
        if self.e8m0 and scheme.kind is QuantKind.FP8_BLOCK:
            names["weight_scale_inv"] = Stored("scale")
        return names

    def _expert_algo(self, name: str) -> str:
        """The routed-expert algo for one module: the explicit per-module entry, else the global default.

        An export that ships ``quantized_layers`` is explicit -- a block it does not name is
        not the NVFP4 build (DeepSeek-V4.1's MTP draft experts keep the older per-32 layout).
        """
        cached = self._expert_algo_cache.get(name)
        if cached is not None:
            return cached
        algo = self.expert_algo_global
        if self.quantized_experts:
            algo = ""
            for key, listed in self.quantized_experts.items():
                if name == key or name.startswith(key + ".") or name.endswith("." + key) or ("." + key + ".") in name:
                    algo = listed
                    break
        self._expert_algo_cache[name] = algo
        return algo

    def scheme_for_name(self, name: str) -> QuantScheme | None:
        if self.convert_tables(name):
            return self.SCHEMES["TABLE"]
        if self.not_convert(name) or self.not_convert_substr(name):
            return None
        if self.expert_fp4 and is_routed_expert(name):
            return self.SCHEMES["EXPERT_NVFP4" if self._expert_algo(name) == "NVFP4" else "EXPERT_MXFP4"]
        return self.SCHEMES["BLOCK_E8M0" if self.e8m0 else "BLOCK"]
