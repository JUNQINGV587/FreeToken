"""Qwen3.8-Flash-Next decoder stack (text-only).

The residual state is ``R [T, hc_count*hidden]`` end to end: the embedding is repeated over the
``hc_count`` streams, every layer mixes them down to one ``[T, hidden]`` block input and injects
its output back, and the top-level mixer collapses them once before ``lm_head``. There is no
input/post layernorm and no final ``model.norm`` -- the hyper-connection norms are the only ones.

Layer contract (frozen): ``forward(R [T, hc*hidden], batch) -> R' [T, hc*hidden]`` with an
immediate combine::

    R  = R + ple(R, batch)                 # zero-based layer 1 only
    x, s = attn_hc.mix(R); y = (GDN | QSA)(x); R = attn_hc.combine(R, y, s)
    x, s = mlp_hc.mix(R);  y = MoE(x);        R = mlp_hc.combine(R, y, s)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

import torch
from freetoken.core import get_global_ctx
from freetoken.layers import BaseOP, OPList, ParallelLMHead, VocabParallelEmbedding
from freetoken.models.blocks import BaseLLMModel
from freetoken.utils import nvtx_annotate

from .attention import Qwen4ExpAttention
from .hc import GatedResidual
from .moe import Qwen4ExpMoE
from .ple import PLELayer
from freetoken.models.blocks import embed_input_ids
from freetoken.models.qwen3_vl.vision import Qwen3VLVisionModel, QwenVLVisionMixin

if TYPE_CHECKING:
    from freetoken.core import Batch
    from freetoken.models.config import ModelConfig


def build_linear_mixer(config: ModelConfig, layer_id: int, prefix: str) -> BaseOP:
    """GDN mixer of a linear_attention layer (Qwen3.5's GDN with a configurable output gate)."""
    from .gdn import Qwen4ExpGatedDeltaNet

    g = config.linear_attention_group()
    return Qwen4ExpGatedDeltaNet(
        hidden_size=config.hidden_size,
        num_k_heads=g.num_key_heads,
        num_v_heads=g.num_value_heads,
        head_k_dim=g.key_head_dim,
        head_v_dim=g.value_head_dim,
        conv_kernel_size=g.conv_kernel_dim,
        rms_norm_eps=config.rms_norm_eps,
        layer_id=layer_id,
        output_gate=g.output_gate,
        quant_config=config.quant,
        prefix=prefix,
    )


class Qwen4ExpDecoderLayer(BaseOP):
    """One decoder layer over the hyper-connection streams (see the module docstring for the flow)."""

    def __init__(self, config: ModelConfig, layer_id: int, *, prefix: str = "") -> None:
        self._layer_id = layer_id
        self._is_linear = config.is_linear_layer(layer_id)
        if self._is_linear:
            self.linear_attn = build_linear_mixer(config, layer_id, f"{prefix}.linear_attn")
        else:
            self.self_attn = Qwen4ExpAttention(config, layer_id, prefix=f"{prefix}.self_attn")
        self.mlp = Qwen4ExpMoE(config, layer_id, prefix=f"{prefix}.mlp")
        self.attn_hyper_connection = GatedResidual(config, prefix=f"{prefix}.attn_hyper_connection")
        self.mlp_hyper_connection = GatedResidual(config, prefix=f"{prefix}.mlp_hyper_connection")
        self.ple = (
            PLELayer(config, layer_id, prefix=f"{prefix}.ple") if layer_id in config.qwen4_args.ple_layer_ids else None
        )

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(self, hidden: torch.Tensor, batch: Batch) -> torch.Tensor:
        if self.ple is not None:
            hidden = hidden + self.ple.forward(hidden, batch)
        block_input, inject = self.attn_hyper_connection.mix(hidden)
        if self._is_linear:
            block_output = self.linear_attn.forward(block_input)
        else:
            block_output = self.self_attn.forward(block_input, batch)
        hidden = self.attn_hyper_connection.combine(hidden, block_output, inject)
        block_input, inject = self.mlp_hyper_connection.mix(hidden)
        return self.mlp_hyper_connection.combine(hidden, self.mlp.forward(block_input), inject)

    def forward_fused(
        self,
        hidden: torch.Tensor,
        rn: torch.Tensor | None,
        batch: Batch,
        next_norm,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """FREETOKEN_HC_PREFILL_FUSED path: mix consumes pre-normed streams, combine
        returns the next block's normed streams fused into the same launch. Bit-identical
        to forward (spec in kernel/triton/hc_prefill_fused.py). ``rn`` is None at model
        entry and after a PLE add; ``next_norm`` is the norm that consumes this layer's
        mlp combine output (None when a PLE add or nothing follows)."""
        from freetoken.kernel.triton.hc import grouped_gemma_rmsnorm

        if self.ple is not None:
            hidden = hidden + self.ple.forward(hidden, batch)
            rn = None
        attn_hc, mlp_hc = self.attn_hyper_connection, self.mlp_hyper_connection
        if rn is None:
            rn = grouped_gemma_rmsnorm(hidden, attn_hc.hc_norm.weight, attn_hc.hc_norm.eps, attn_hc.hc_count)
        block_input, inject = attn_hc.mix_fused(rn)
        if self._is_linear:
            block_output = self.linear_attn.forward(block_input)
        else:
            block_output = self.self_attn.forward(block_input, batch)
        hidden, rn = attn_hc.combine_fused(hidden, block_output, inject, mlp_hc.hc_norm)
        block_input, inject = mlp_hc.mix_fused(rn)
        return mlp_hc.combine_fused(hidden, self.mlp.forward(block_input), inject, next_norm)


class Qwen4ExpModel(BaseOP):
    def __init__(self, config: ModelConfig, *, prefix: str = "model") -> None:
        self.hc_count = config.qwen4_args.hc_count
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        self.layers = OPList(
            [
                Qwen4ExpDecoderLayer(config, layer_id, prefix=f"{prefix}.layers.{layer_id}")
                for layer_id in range(config.num_layers)
            ]
        )
        self.hyper_connection_mixer = GatedResidual(config, use_combine=False, prefix=f"{prefix}.hyper_connection_mixer")
        # plain tuple (not an OP child), so it never shows up in the state dict
        self._ple = tuple(layer.ple for layer in self.layers.op_list if layer.ple is not None)

    @property
    def ple_layers(self) -> List[PLELayer]:
        """The PLE layers in decoder order -- the seam the loader attaches table backends to."""
        return list(self._ple)

    def forward(self, input_ids: torch.Tensor, batch: Batch) -> torch.Tensor:
        from .hc import prefill_fused_active

        hidden = embed_input_ids(self.embed_tokens, input_ids, batch)
        hidden = hidden.repeat(1, self.hc_count)
        if prefill_fused_active(hidden) and all(
            layer.attn_hyper_connection.fused_weights_bf16()
            and layer.mlp_hyper_connection.fused_weights_bf16()
            for layer in self.layers.op_list
        ):
            return self._forward_hc_fused(hidden, batch)
        meta = None
        if self._ple:
            from .ple import build_ple_metadata, commit_ngram_context

            meta = build_ple_metadata(batch, self._ple[0].args, input_ids.device)
            for ple in self._ple:  # gather the pinned-host PLE rows while the early layers run
                ple.start_prefetch(batch, meta)
        for layer in self.layers.op_list:
            hidden = layer.forward(hidden, batch)
        if meta is not None:
            # single writer: the layers only read the context, so a second PLE layer's
            # prefetch sees the un-rolled window
            commit_ngram_context(meta, getattr(batch, "fla_metadata", None))
        return self.hyper_connection_mixer.mix(hidden)[0]

    def _forward_hc_fused(self, hidden: torch.Tensor, batch: Batch) -> torch.Tensor:
        """FREETOKEN_HC_PREFILL_FUSED prefill path (gate checked by forward):
        identical math, fused HC kernels (spec in kernel/triton/hc_prefill_fused.py)."""
        meta = None
        if self._ple:
            from .ple import build_ple_metadata, commit_ngram_context

            meta = build_ple_metadata(batch, self._ple[0].args, batch.input_ids.device)
            for ple in self._ple:
                ple.start_prefetch(batch, meta)
        layers = self.layers.op_list
        rn = None
        for i, layer in enumerate(layers):
            if i + 1 < len(layers):
                nxt = layers[i + 1]
                # a PLE add between this combine and the next norm breaks the fusion
                next_norm = None if nxt.ple is not None else nxt.attn_hyper_connection.hc_norm
            else:
                next_norm = self.hyper_connection_mixer.hc_norm
            hidden, rn = layer.forward_fused(hidden, rn, batch, next_norm)
        if meta is not None:
            # single writer: the layers only read the context, so a second PLE layer's
            # prefetch sees the un-rolled window
            commit_ngram_context(meta, getattr(batch, "fla_metadata", None))
        return self.hyper_connection_mixer.mix_fused(rn)[0]


class Qwen4ExpForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig) -> None:
        self._config = config
        self.model = Qwen4ExpModel(config)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
            quant_config=config.quant,
            prefix="lm_head",
        )
        super().__init__()

    def load_host_tables(self, engine_config) -> int:
        """Attach the PLE n-gram table (pinned checkpoint bank, or zeros for dummy weights); returns the pinned host bytes the engine reserves from its pin budget."""
        ple_layers = self.model.ple_layers
        if not ple_layers:
            return 0
        from .ple import PinnedUVATable, ZeroTable, derive_ngram_hash_constants

        if getattr(engine_config, "use_dummy_weight", False):
            # Dummy fill leaves the int64 hash buffers garbage (a zero vocab size divides by
            # zero in the hash), so re-derive the real constants and read a zero table.
            for ple in ple_layers:
                args = ple.args
                mult, sizes, offsets = derive_ngram_hash_constants(
                    vocab_size=self._config.vocab_size,
                    ngram_size=args.ngram_size,
                    num_ngram_heads=args.num_ngram_heads,
                    ngram_vocab_size_base=args.ngram_vocab_size_base,
                    ple_layer_index=ple.ple_index,
                )
                emb = ple.ple_embedding
                emb.layer_multipliers.copy_(torch.tensor(mult, dtype=torch.int64))
                emb.ngram_heads_vocab_sizes.copy_(torch.tensor(sizes, dtype=torch.int64))
                emb.ngram_heads_offsets.copy_(torch.tensor(offsets, dtype=torch.int64))
                emb.attach_table(ZeroTable(offsets[-1] + sizes[-1], args.ngram_head_dim))
            return 0

        if engine_config.ple_backend == "disk":
            from freetoken.utils import download_hf_weight

            from .ple_disk import DiskRowTable, resolve_row_source

            folder = download_hf_weight(engine_config.model_path)
            # one WAIT node per captured graph: the flag protocol supports a single consume
            assert len(ple_layers) == 1, "disk PLE backend expects exactly one PLE layer"
            emb, args = ple_layers[0].ple_embedding, ple_layers[0].args
            # hash with the state-dict-loaded constants, the same source the pinned path reads
            constants = {
                "num_ngram_heads": args.num_ngram_heads,
                "layer_multipliers": emb.layer_multipliers.tolist(),
                "per_head_vocab_sizes": emb.ngram_heads_vocab_sizes.tolist(),
                "per_head_offsets": emb.ngram_heads_offsets.tolist(),
                "eos_token_id": args.ngram_boundary_token_id,
                "image_token_id": args.image_token_id,
            }
            disk_table = DiskRowTable(
                resolve_row_source(folder),
                constants,
                max_graph_rows=max(256, engine_config.cuda_graph_max_bs or 0),
                max_extend_tokens=engine_config.max_extend_tokens,
            )
            self._ple_table = disk_table
            for ple in ple_layers:
                ple.ple_embedding.attach_table(disk_table)
            # engine enters this around every dispatch; the graph itself never waits on the disk
            self.forward_host_ctx = disk_table.forward_host_ctx
            return 0

        from .weight import load_ple_table

        table = load_ple_table(engine_config.model_path, self._config.qwen4_args)
        self._ple_table = table  # owns the pinned HostBank; keep it alive
        for ple in ple_layers:
            ple.ple_embedding.attach_table(
                PinnedUVATable(table.bank.tensor, float(table.weight_scale))
            )
        return table.bank.nbytes

    def forward(self) -> torch.Tensor:
        batch = get_global_ctx().batch
        hidden = self.model.forward(batch.input_ids, batch)
        # Project only the rows the sampler reads. At this vocab (248,320) a full
        # 8192-token prefill chunk would allocate 4.07 GiB of bf16 logits and run a
        # vocab GEMM 1024x larger than needed, both on the TTFT path; lm_head is
        # row-wise, so gathering first is exact. Prefill packs the extend window, so
        # the rows to keep are each request's last token (what LMHead.forward would
        # gather via get_last_indices), NOT the leading batch.size rows; decode rows
        # are already 1:1 with requests (padded window: keep the leading batch.size).
        if batch.is_prefill:
            hidden = hidden[batch.attn_metadata.get_last_indices(batch.size)]
        else:
            hidden = hidden[: batch.size]
        return self.lm_head.forward(hidden)


class Qwen4ExpForConditionalGeneration(QwenVLVisionMixin, Qwen4ExpForCausalLM):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        if config.is_multimodal:
            assert not config.vision_config.deepstack_visual_indexes, "Qwen3.8 consumes no DeepStack features"
            self.visual = Qwen3VLVisionModel(config.vision_config, quant_config=config.quant, prefix="visual")


__all__ = [
    "Qwen4ExpDecoderLayer",
    "Qwen4ExpForCausalLM",
    "Qwen4ExpForConditionalGeneration",
    "Qwen4ExpModel",
    "build_linear_mixer",
]
