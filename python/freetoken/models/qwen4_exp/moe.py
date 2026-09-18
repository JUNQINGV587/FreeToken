from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from freetoken.distributed import get_tp_info
from freetoken.kernel.triton.moe_shared_gate import shared_gate_mul_add, shared_gate_sigmoid
from freetoken.layers import BaseOP, silu_and_mul
from freetoken.models.qwen3_5_moe.moe import Qwen3_5MoE
from freetoken.models.qwen4_exp.gemv_concat import f1_enabled
from freetoken.utils import div_even

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


class _RouterGateUpProj(BaseOP):
    """Load-time concat of the MoE router and the shared expert's gate|up projection.

    Row layout: ``[router (TP-replicated) | gate (rank-local) | up (rank-local)]``. The
    single fused GEMV is bitwise identical to the two separate GEMVs only at M==1 (cuBLAS
    switches algorithm with N at M>=2), so forward dispatches on M: M>1 runs the original
    pair on contiguous row slices of the fused buffer -- same shapes and values as the
    unfused weights, hence the same kernels and the same bits.
    """

    def __init__(self, hidden_size: int, num_experts: int, intermediate_size: int) -> None:
        self.num_router_rows = num_experts
        self.local_inter = div_even(intermediate_size, get_tp_info().size)
        self.weight = torch.empty(num_experts + 2 * self.local_inter, hidden_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n = self.num_router_rows
        if x.shape[0] == 1:
            y = F.linear(x, self.weight)
            return y[:, :n], y[:, n:]
        return F.linear(x, self.weight[:n]), F.linear(x, self.weight[n:])


class Qwen4ExpMoE(Qwen3_5MoE):
    """Qwen3_5MoE with the shared-expert gate on triton instead of gemv + sigmoid + mul + add.

    Same weights, same state dict. The gate reduction stays ahead of the routed experts, which may write into ``hidden_states`` in place.

    With FREETOKEN_GEMV_CONCAT the router and the shared expert's gate|up projection load
    as one ``router_gate_up`` buffer (see gemv_concat.py); the state dict then carries
    ``mlp.router_gate_up.weight`` instead of ``mlp.gate.weight`` +
    ``mlp.shared_expert.gate_up_proj.weight``.
    """

    def __init__(self, config: ModelConfig, layer_id: int | None = None, *, prefix: str = ""):
        super().__init__(config, layer_id, prefix=prefix)
        if f1_enabled(config):
            self.router_gate_up = _RouterGateUpProj(
                config.hidden_size,
                config.num_experts,
                config.shared_expert_intermediate_size,
            )
            del self.gate
            del self.shared_expert.gate_up_proj
        else:
            self.router_gate_up = None

    def _fused_shared(self, gate_up: torch.Tensor) -> torch.Tensor:
        # Same silu dispatch as the unfused arm, so the kernels (and bits) match: at
        # M==1 the gate_up slice of the fused projection is trivially contiguous (the
        # row stride of a 1-row tensor does not count), at M>1 it is a fresh GEMV output.
        return silu_and_mul(gate_up)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        owner_ep = getattr(self.experts, "owner_cache", None) is not None
        if self.router_gate_up is not None:
            router_logits, gate_up = self.router_gate_up.forward(hidden_states)
            if owner_ep:
                if not hasattr(self.shared_expert.down_proj, "_tp_size"):
                    raise NotImplementedError(
                        "owner EP shared+routed fusion requires a row-parallel shared projection"
                    )
                # Compute the gate before routed experts: a fused routed kernel is allowed to
                # mutate its hidden input in-place. The non-owner path has the same ordering
                # contract; owner mode must not rely on the current NVFP4 kernel being benign.
                gate = shared_gate_sigmoid(
                    hidden_states, self.shared_expert_gate.weight.view(-1)
                )
                shared = self.shared_expert.down_proj.forward(
                    self._fused_shared(gate_up), reduce=False
                )
                routed = self.experts.forward(
                    hidden_states=hidden_states, router_logits=router_logits, reduce=False
                )
                merged = shared_gate_mul_add(routed, shared, gate)
                return self.experts._maybe_all_reduce(merged).view(num_tokens, hidden_dim)
            shared = self.shared_expert.down_proj.forward(self._fused_shared(gate_up))
            gate = shared_gate_sigmoid(hidden_states, self.shared_expert_gate.weight.view(-1))
            routed = self.experts.forward(hidden_states=hidden_states, router_logits=router_logits)
            return shared_gate_mul_add(routed, shared, gate).view(num_tokens, hidden_dim)

        router_logits = self.gate.forward(hidden_states)
        if owner_ep:
            if not hasattr(self.shared_expert.down_proj, "_tp_size"):
                raise NotImplementedError(
                    "owner EP shared+routed fusion requires a row-parallel shared projection"
                )
            # Compute the gate before routed experts: a fused routed kernel is allowed to
            # mutate its hidden input in-place. The non-owner path has the same ordering
            # contract; owner mode must not rely on the current NVFP4 kernel being benign.
            gate = shared_gate_sigmoid(
                hidden_states, self.shared_expert_gate.weight.view(-1)
            )
            shared = self.shared_expert.down_proj.forward(
                silu_and_mul(self.shared_expert.gate_up_proj.forward(hidden_states)),
                reduce=False,
            )
            routed = self.experts.forward(
                hidden_states=hidden_states, router_logits=router_logits, reduce=False
            )
            merged = shared_gate_mul_add(routed, shared, gate)
            return self.experts._maybe_all_reduce(merged).view(num_tokens, hidden_dim)

        shared = self.shared_expert.forward(hidden_states)
        gate = shared_gate_sigmoid(hidden_states, self.shared_expert_gate.weight.view(-1))
        routed = self.experts.forward(hidden_states=hidden_states, router_logits=router_logits)
        return shared_gate_mul_add(routed, shared, gate).view(num_tokens, hidden_dim)


__all__ = ["Qwen4ExpMoE"]
