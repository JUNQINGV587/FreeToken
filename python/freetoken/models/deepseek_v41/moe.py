"""DSV4.1 MoE: sqrtsoftplus router (no hash layers), shared SwiGLU expert, offloaded NVFP4
routed experts (ModelOpt per-16 E4M3 scales + a per-tensor ``scale_2``)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from freetoken.kernel.triton.dsv4.bf16_linear import bf16_linear_fp32
from freetoken.kernel.triton.dsv4.swiglu import fused_swiglu
from freetoken.layers import BaseOP, LinearColParallelMerged, LinearRowParallel, OffloadMoELayer
from freetoken.layers.moe import owner_ep_expert_tp_size

from .args import DeepseekV41Args


class Gate(BaseOP):
    """MoE router: sqrtsoftplus scoring with a selection-only bias, and no hash layers.

    V4.1 keeps V4's scoring but drops hash routing entirely and carries a second bias for
    image-span tokens (``bias_vl``); plain ``bias`` serves text. The bias steers *selection*
    only -- the routing weights come from the unbiased scores, matching training.
    """

    def __init__(self, layer_id: int, args: DeepseekV41Args):
        self.topk = args.n_activated_experts
        self.score_func = args.score_func
        self.gate_temp = args.gate_temp
        self.norm_topk_prob = args.norm_topk_prob
        self.route_scale = args.route_scale
        self.weight = torch.empty(args.n_routed_experts, args.dim, dtype=torch.bfloat16)
        self.bias = torch.empty(args.n_routed_experts, dtype=torch.float32)
        # The checkpoint always ships bias_vl (it was trained with the VL tower); it is only
        # consulted when an image mask is supplied, so keep the tensor unconditionally.
        self.bias_vl = torch.empty(args.n_routed_experts, dtype=torch.float32)

    def forward(
        self, x: torch.Tensor, image_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores = bf16_linear_fp32(x, self.weight)
        if self.gate_temp != 1.0:
            scores = scores / self.gate_temp
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1)
        elif self.score_func == "sigmoid":
            scores = scores.sigmoid()
        else:
            scores = F.softplus(scores).sqrt()
        bias = self.bias
        if image_mask is not None and self.bias_vl is not None:
            bias = torch.where(image_mask.unsqueeze(-1), self.bias_vl, bias)
        indices = (scores + bias).topk(self.topk, dim=-1)[1]
        weights = scores.gather(1, indices)
        if self.norm_topk_prob and self.topk > 1:
            # 1e-20, not norm_eps: this is the constant training used.
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        return weights * self.route_scale, indices


class Expert(BaseOP):
    """Dense SwiGLU expert (the shared expert; routed experts are offloaded NVFP4)."""

    def __init__(self, dim: int, inter_dim: int, swiglu_limit: float, *, quant_config=None, prefix: str = ""):
        self.w1 = LinearColParallelMerged(dim, [inter_dim], has_bias=False, quant_config=quant_config, prefix=f"{prefix}.w1")
        self.w2 = LinearRowParallel(inter_dim, dim, has_bias=False, quant_config=quant_config, prefix=f"{prefix}.w2")
        self.w3 = LinearColParallelMerged(dim, [inter_dim], has_bias=False, quant_config=quant_config, prefix=f"{prefix}.w3")
        self.swiglu_limit = swiglu_limit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = fused_swiglu(self.w1.forward(x), self.w3.forward(x), self.swiglu_limit, x.dtype)
        return self.w2.forward(h)


class DSV41OffloadMoELayer(OffloadMoELayer):
    """Routed NVFP4 experts on the shared offload cache.

    Identical strategy to V4's MXFP4 layer (whole-layer streaming prefill, slot-cache / cpu / hybrid
    decode); only the stored expert format differs, and that comes from ``quant_config``.
    """

    def __init__(self, layer_id: int, args: DeepseekV41Args, *, strategy: str = "offload", decode_target: str = "gpu", quant_config=None, prefix: str = ""):
        # V4.1 trained the routed experts with swiglu_limit=10 (silu(min(gate, L)) * clamp(up, +-L)),
        # which the shared NVFP4 epilogue now computes through ``swiglu_clamp``; the marlin and b12x
        # kernels still cannot express it and are rejected by ``gated_epilogue_reason``, so these
        # layers select the Triton backend.
        #
        # The experts are NVFP4, and no NVFP4 expert kernel accepts TP > 1: the kernel has to be told
        # the GEMM is unsharded, which is exactly true under owner-local EP (every rank holds whole,
        # disjoint experts) -- the layer still all-reduces once at its output. Without it every
        # candidate kernel is skipped and building the model dies with KernelSelectionError
        # ("TP > 1 is not supported for this expert format"). This seam constructs the offload layer
        # directly instead of going through ``make_moe_layer``, so it asks the same helper that
        # factory uses rather than re-deriving the rule.
        super().__init__(
            layer_id=layer_id,
            num_experts=args.n_routed_experts,
            top_k=args.n_activated_experts,
            hidden_size=args.dim,
            intermediate_size=args.moe_inter_dim,
            renormalize=True,
            activation="silu",
            limit=args.swiglu_limit if args.swiglu_limit > 0 else None,
            expert_tp_size=owner_ep_expert_tp_size(args),
            strategy=strategy,
            decode_target=decode_target,
            quant_config=quant_config,
            prefix=prefix,
        )

    def _prefill_routed(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        # Whole-layer streaming moves all num_experts rows per layer; a small chunk touches at
        # most T*top_k of them, so below that crossover the decode-style on-demand slot path
        # strictly moves fewer bytes. Mixing modes across chunks is safe: the streaming buffers
        # disown their borrowed slots on invalidation.
        #
        # This path routes on *slot* ids, whose namespace is the whole pool (1675 rows, shared
        # across layers), and hands that id space to moe_align_block_size. sglang's align kernel
        # silently drops anything past ~1024 experts (it wrote ntpp = -1694418143 and
        # expert_ids ~ 1e9 here, after which the inline-NVFP4 GEMM's
        # `pid_m * BLOCK_SIZE_M >= num_tokens_post_padded` guard never fired and indexed a wild
        # bank address). fused.py now keeps the sgl kernel to id spaces it is known to handle and
        # falls back to FreeToken's own triton align above that -- its histogram is sized from the
        # id space (HIST = next_pow2(1675 + 2) = 2048), which is what makes this shortcut legal.
        cache = self.offload_cache
        assert cache is not None
        if self.owner_cache is not None:
            # The owner adapter must see the original global route; its local-row remap,
            # borrowed-buffer lifecycle and remote zeroing stay in the base implementation.
            return super()._prefill_routed(hidden_states, topk_weights, topk_ids)
        # Unpinned (LOCKED) layers must take the base materialize path: their copy_missing is the
        # whole-layer pageable branch with position == expert id, which ensure_experts's LRU slot
        # remap would contradict.
        if (
            hidden_states.shape[0] * self.top_k >= self.num_experts
            or cache.is_unpinned_layer(self.layer_id)
        ):
            return super()._prefill_routed(hidden_states, topk_weights, topk_ids)
        cache.ensure_experts(self.layer_id, topk_ids)  # in-place expert-id -> slot
        cache.copy_missing()
        if cache.collect_stats:
            cache.record_decode_stats(self.layer_id)
        # ``n`` is the align kernel's num_experts, so it must be the id space these rows are drawn
        # from: the bank's row count (the slot pool), not the model's 384 experts. The base path's
        # invariant is the same one -- there n == num_experts because its ids are expert ids.
        views = cache.bank_views()
        return self._expert_gemm(
            cache,
            hidden_states,
            topk_weights,
            topk_ids,
            views=views,
            n=views[0].shape[0],
            alphas=cache.alphas_for_slots(self.layer_id),
            is_prefill=True,
        )


class MoE(BaseOP):
    """Sparse MoE: score router -> offloaded NVFP4 routed experts + one shared expert."""

    def __init__(self, layer_id: int, args: DeepseekV41Args, *, strategy: str = "offload", decode_target: str = "gpu", quant_config=None, prefix: str = ""):
        self.dim = args.dim
        self.gate = Gate(layer_id, args)
        self.shared_experts = Expert(args.dim, args.moe_inter_dim, args.swiglu_limit, quant_config=quant_config, prefix=f"{prefix}.shared_experts")
        self.experts = DSV41OffloadMoELayer(layer_id, args, strategy=strategy, decode_target=decode_target, quant_config=quant_config, prefix=f"{prefix}.experts")

    def forward(self, x: torch.Tensor, image_mask: torch.Tensor | None = None) -> torch.Tensor:
        shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices = self.gate.forward(x, image_mask)
        # Shared expert enqueued before routed_forward: the hybrid decode path blocks on the CPU
        # pool inside routed_forward, so this GEMM must already be on the stream to overlap it.
        shared = self.shared_experts.forward(x)
        # routed_forward may mutate the ids in place (offload decode slot remap);
        # indices.to(int32) always copies, so no clone is needed here.
        routed = self.experts.routed_forward(
            x, weights.float().contiguous(), indices.to(torch.int32).contiguous()
        )
        # The reference accumulates routes in fp32 and casts the sum back to the activation dtype
        # (``y.type_as(x)``); the shared expert is added before that cast because a bf16 add would
        # round twice.
        return (routed + shared.float()).to(x.dtype).view(shape)


__all__ = ["Gate", "Expert", "DSV41OffloadMoELayer", "MoE"]
