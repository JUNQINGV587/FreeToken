"""DSV4 MoE: sqrtsoftplus/hash router, shared SwiGLU expert, offloaded MXFP4 routed
experts (GPU slot-cache / cpu / hybrid decode paths)."""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F

from freetoken.core import get_global_ctx
from freetoken.kernel.triton.dsv4.bf16_linear import bf16_linear_fp32
from freetoken.kernel.triton.dsv4.swiglu import fused_swiglu
from freetoken.layers import BaseOP, LinearColParallelMerged, LinearRowParallel, OffloadMoELayer
from freetoken.layers.moe import owner_ep_expert_tp_size
from freetoken.mm import MM_PAD_SHIFT_VALUE

from .args import DeepseekV4Args


class Gate(BaseOP):
    """MoE router: sqrtsoftplus scoring + hash routing (first ``n_hash_layers``).

    A vision checkpoint gives every layer a ``bias_vl`` too: the expert-selection bias for
    image rows, which never touches the routing weights. Hash layers keep routing text rows
    through ``tid2eid`` and pick image rows by ``(scores + bias_vl).topk``, mirroring the
    reference ``Gate.forward``.
    """

    def __init__(self, layer_id: int, args: DeepseekV4Args):
        self.topk = args.n_activated_experts
        self.score_func = args.score_func
        self.route_scale = args.route_scale
        self.hash = layer_id < args.n_hash_layers
        vl = args.vision_n_layers > 0
        self.weight = torch.empty(args.n_routed_experts, args.dim, dtype=torch.bfloat16)
        if self.hash:
            self.tid2eid = torch.empty(args.vocab_size, args.n_activated_experts, dtype=torch.int64)
            # present on hash layers of a vision checkpoint, loaded but never applied
            self.bias = torch.empty(args.n_routed_experts, dtype=torch.float32) if vl else None
        else:
            self.bias = torch.empty(args.n_routed_experts, dtype=torch.float32)
        self.bias_vl = torch.empty(args.n_routed_experts, dtype=torch.float32) if vl else None

    def forward(self, x: torch.Tensor, input_ids: torch.Tensor):
        scores = bf16_linear_fp32(x, self.weight)
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1)
        elif self.score_func == "sigmoid":
            scores = scores.sigmoid()
        else:
            scores = F.softplus(scores).sqrt()
        original_scores = scores
        # image rows are the content pad ids of an expanded image span
        image_mask = input_ids >= MM_PAD_SHIFT_VALUE if self.bias_vl is not None else None
        if self.hash:
            if image_mask is None:
                indices = self.tid2eid[input_ids]
            else:
                indices = self.tid2eid[torch.where(image_mask, 0, input_ids)]
                vl_indices = (scores + self.bias_vl).topk(self.topk, dim=-1)[1]
                indices = torch.where(image_mask.unsqueeze(-1), vl_indices.to(indices.dtype), indices)
        else:
            if image_mask is None:
                scores = scores + self.bias
            else:
                scores = scores + torch.where(image_mask.unsqueeze(-1), self.bias_vl, self.bias)
            indices = scores.topk(self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        if self.score_func != "softmax":
            weights = weights / weights.sum(dim=-1, keepdim=True)
        weights = weights * self.route_scale
        return weights, indices


class Expert(BaseOP):
    """Dense SwiGLU expert (the shared expert; routed experts are offloaded FP4)."""

    def __init__(self, dim: int, inter_dim: int, swiglu_limit: float, *, quant_config=None, prefix: str = ""):
        self.w1 = LinearColParallelMerged(dim, [inter_dim], has_bias=False, quant_config=quant_config, prefix=f"{prefix}.w1")
        self.w2 = LinearRowParallel(inter_dim, dim, has_bias=False, quant_config=quant_config, prefix=f"{prefix}.w2")
        self.w3 = LinearColParallelMerged(dim, [inter_dim], has_bias=False, quant_config=quant_config, prefix=f"{prefix}.w3")
        self.swiglu_limit = swiglu_limit

    def forward(self, x: torch.Tensor, *, reduce: bool = True) -> torch.Tensor:
        h = fused_swiglu(self.w1.forward(x), self.w3.forward(x), self.swiglu_limit, x.dtype)
        return self.w2.forward(h, reduce=reduce)


class DSV4OffloadMoELayer(OffloadMoELayer):
    """Routed experts (MXFP4 author dialect or NVFP4 0731 banks) on the shared offload
    cache: the base whole-layer streaming prefill (grouped inline-dequant GEMM for
    dense chunks, GEMV below the route crossover) and slot-cache / cpu / hybrid decode
    paths (per-route dequant GEMV). Under owner-local EP (``moe_ep_size > 1``) each rank
    owns a slice of experts whole (NVFP4 banks reject TP > 1), so the expert GEMM runs
    unsharded via ``expert_tp_size``."""

    def __init__(self, layer_id: int, args: DeepseekV4Args, *, strategy: str = "offload", decode_target: str = "gpu", quant_config=None, prefix: str = ""):
        super().__init__(
            layer_id=layer_id,
            num_experts=args.n_routed_experts,
            top_k=args.n_activated_experts,
            hidden_size=args.dim,
            intermediate_size=args.moe_inter_dim,
            renormalize=True,
            activation="silu",
            limit=args.swiglu_limit,
            strategy=strategy,
            decode_target=decode_target,
            quant_config=quant_config,
            prefix=prefix,
            expert_tp_size=owner_ep_expert_tp_size(args),
        )

    def _prefill_routed(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        # Whole-layer streaming moves all num_experts rows per layer; a small
        # chunk touches at most T*top_k of them, so below that crossover the
        # decode-style on-demand slot path strictly moves fewer bytes (and
        # keeps short-prompt slot residency -- hence hybrid decode's GPU/CPU
        # route split -- unchanged). Mixing modes across chunks is safe: the
        # streaming buffers disown their borrowed slots on invalidation.
        cache = self.offload_cache
        assert cache is not None
        if self.owner_cache is not None:
            # The owner adapter must see the original global route. Its local-row remap,
            # borrowed-buffer lifecycle, and remote zeroing are handled by the base owner
            # implementation; the short-prefill optimization is global-cache-only.
            return super()._prefill_routed(hidden_states, topk_weights, topk_ids)
        # unpinned (LOCKED) layers must take the base materialize path: their copy_missing is the whole-layer pageable branch with position == expert id, which ensure_experts's LRU slot remap would contradict (the GEMM would gather other experts' weights)
        if (
            hidden_states.shape[0] * self.top_k >= self.num_experts
            or cache.is_unpinned_layer(self.layer_id)
        ):
            return super()._prefill_routed(hidden_states, topk_weights, topk_ids)
        cache.ensure_experts(self.layer_id, topk_ids)  # in-place expert-id -> slot
        cache.copy_missing()
        if cache.collect_stats:
            cache.record_decode_stats(self.layer_id)
        return self._expert_gemm(
            cache,
            hidden_states,
            topk_weights,
            topk_ids,
            views=cache.bank_views(),
            n=None,
            alphas=cache.alphas_for_slots(self.layer_id),
            is_prefill=True,
        )


class MoE(BaseOP):
    """Sparse MoE: hash/score router -> offloaded MXFP4 routed experts + shared expert."""

    def __init__(self, layer_id: int, args: DeepseekV4Args, *, strategy: str = "offload", decode_target: str = "gpu", quant_config=None, prefix: str = ""):
        self.dim = args.dim
        self.layer_id = layer_id
        # One rendezvous per layer instead of two: shared and routed expert outputs
        # are partial sums over the same TP group, so their adds commute with the
        # all-reduce (qwen4_exp/moe.py precedent). Not bit-identical to the split
        # path (one bf16 add moves across the fp32-accumulated reduce).
        self._merge_shared_ar = os.environ.get("FREETOKEN_DSV4_MERGE_SHARED_AR", "1") != "0"
        self.gate = Gate(layer_id, args)
        self.shared_experts = Expert(args.dim, args.moe_inter_dim, args.swiglu_limit, quant_config=quant_config, prefix=f"{prefix}.shared_experts")
        self.experts = DSV4OffloadMoELayer(layer_id, args, strategy=strategy, decode_target=decode_target, quant_config=quant_config, prefix=f"{prefix}.experts")

    def forward(self, x: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices = self.gate.forward(x, input_ids.flatten())
        # Route-trace token-id sidecar: one marker per decode step at layer 0,
        # so the offline lookahead analysis can pair routing with the token
        # that produced it. Only active under --moe-trace-route.
        if self.layer_id == 0:
            rec = getattr(getattr(self.experts, "offload_cache", None),
                          "route_recorder", None)
            if rec is not None and not get_global_ctx().batch.is_prefill:
                rec.record_tokens(input_ids)
        # Shared expert enqueued before routed_forward: hybrid decode blocks on the
        # CPU pool inside routed_forward, so this GEMM must already be on the stream
        # to overlap the CPU overflow compute.
        shared = self.shared_experts.forward(x, reduce=not self._merge_shared_ar)
        # routed_forward may mutate the ids in place (offload decode slot remap);
        # indices.to(int32) always copies (int64 source), so no clone needed here.
        routed = self.experts.routed_forward(
            x, weights.float().contiguous(), indices.to(torch.int32).contiguous(),
            reduce=not self._merge_shared_ar,
        )
        if self._merge_shared_ar:
            return self.experts._maybe_all_reduce(routed + shared).view(shape)
        return (routed + shared).view(shape)
