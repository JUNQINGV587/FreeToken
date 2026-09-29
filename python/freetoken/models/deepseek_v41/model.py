"""DeepSeek-V4.1-Flash model class (engine path).

M0: class exists so the registry resolves and weights/config can be validated.
Construction raises NotImplementedError until the M1 attention/CSA2/engram port
lands (see /data/research/notes/v41-port/05-port-design.md).
"""

from __future__ import annotations

from freetoken.models.blocks import BaseLLMModel


class DeepseekV41ForCausalLM(BaseLLMModel):
    """DeepSeek-V4.1-Flash (model_type=deepseek_v41).

    CSA2 (ratio {0,1,2} compressor + shared compressed caches + two-level
    Lightning Indexer) attention, manifold-constrained Hyper-Connections (4
    residual streams), engram lookup (layers 1/14, NVMe-backed 192 GB tables),
    NVFP4 routed experts via the offload cache. Vision tower and DSpark MTP are
    out of scope.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "DeepseekV41ForCausalLM: weight loading / config parsing are wired "
            "(M0); the forward implementation lands with the CSA2/engram port (M1)."
        )
