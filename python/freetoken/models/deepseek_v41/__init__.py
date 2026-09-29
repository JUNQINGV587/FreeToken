"""DeepSeek-V4.1-Flash support for FreeToken.

Ports the official ``inference/model.py`` reference (CSA2: ratio {0,1,2}
compressors + SHARED compressed-KV caches + two-level Lightning Indexer,
manifold-constrained Hyper-Connections, engram lookup on layers 1/14) onto
FreeToken's primitives, reusing the DSV4 machinery where it fits and the
production ModelOpt NVFP4 expert path (offload cache + optional disk tier).

M0 (this commit): args parsing (inference/config.json -> DeepseekV41Args),
parse_config -> ModelConfig (expert_quant="nvfp4", DSV41 attention group),
iter_weights (fp8/e8m0-32 linears, wo_a dequant, compressor/indexer/engram/
gate/hc tensors; engram tables and experts excluded), nvfp4_expert_spec.
The model class exists for registry resolution; forward lands in M1.
"""

from .args import DeepseekV41Args, load_args
from .config import parse_config
from .model import DeepseekV41ForCausalLM
from .weight import iter_weights, nvfp4_expert_spec

__all__ = [
    "DeepseekV41Args",
    "load_args",
    "parse_config",
    "DeepseekV41ForCausalLM",
    "iter_weights",
    "nvfp4_expert_spec",
]
