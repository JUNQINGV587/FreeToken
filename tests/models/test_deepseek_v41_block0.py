"""Layer-0 numerical comparison against the ``inference/`` reference dump (M1).

``block0_ref.pt`` is produced by ``research/v41-oracle/block0_oracle.py``: the OFFICIAL
``inference/model.py`` code, real V4.1 weights, and the three substitutions desktop SM89 forces

  * torch sparse attention -- the tilelang ``sparse_attn`` needs 138 KiB of dynamic shared memory
    per block, past Ada's 99 KiB opt-in maximum,
  * torch NVFP4 dequant -- the reference ``fp4_gemm`` assumes the older per-32 layout while the
    checkpoint is ModelOpt per-16 + fp32 global,
  * ``.to("cuda")`` on the window index list (the reference builds it on the CPU).

This test replays the same layer through the ENGINE modules (triton fp8 block-32 GEMM, the
hyper-connection sinkhorn kernels, the eager CSA2 window path) and compares every intermediate the
dump carries, so a mismatch localizes to a stage instead of only showing up at the block output.

Needs the real checkpoint and the dump, so it skips unless both are present:
    FT_V41_CHECKPOINT   default /mnt/nvme/models/DeepSeek-V4.1-Flash-NVFP4
    FT_V41_ORACLE_DUMP  default /oracle/block0_ref.pt
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest
import torch

from freetoken.layers.quantization import QuantConfig
from freetoken.models.register import get_model_spec
from freetoken.models.deepseek_v41.args import load_args
from freetoken.models.deepseek_v41.model import Block
from freetoken.models.deepseek_v41.weight import iter_weights

CKPT = os.environ.get("FT_V41_CHECKPOINT", "/mnt/nvme/models/DeepSeek-V4.1-Flash-NVFP4")
DUMP = os.environ.get("FT_V41_ORACLE_DUMP", "/oracle/block0_ref.pt")

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and os.path.isdir(CKPT) and os.path.isfile(DUMP)),
    reason="needs CUDA + the V4.1 checkpoint + the layer-0 oracle dump",
)


def _quant_config():
    with open(os.path.join(CKPT, "config.json")) as f:
        quant = json.load(f)["quantization_config"]
    spec = get_model_spec("DeepseekV41ForCausalLM")
    return QuantConfig.from_hf(
        SimpleNamespace(quantization_config=quant), unquantized=spec.unquantized_modules
    )


def _layer0_weights(block, device):
    """The layer-0 slice of the production reader, cast the way the engine loader casts it."""
    # state_dict() is rooted at the block, so re-root it to the checkpoint's own names.
    model_state = {f"model.layers.0.{k}": v for k, v in block.state_dict().items()}
    out = {}
    for name, tensor in iter_weights(CKPT, "cpu", include_moe_experts=False):
        if name.startswith("model.layers.1."):
            break  # names stream in layer order
        if name in model_state:
            out[name] = tensor.to(device=device, dtype=model_state[name].dtype)
    return out


def _load_block(record: dict | None = None):
    from freetoken.distributed import set_tp_info, try_get_tp_info

    from freetoken.utils.torch_utils import torch_dtype

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)  # single-GPU TP=1: v41's dense linears are replicated/col/row
    args = load_args(CKPT, max_batch_size=1)
    # The engine builds models under its compute dtype; RMSNorm's weight (and any unspecified
    # tensor) is allocated with the ambient default, so match it here or the norm kernel sees fp32.
    with torch_dtype(torch.bfloat16):
        block = Block(0, args, quant_config=_quant_config(), prefix="model.layers.0")
    loaded = _layer0_weights(block, "cuda")
    block.load_state_dict(loaded, prefix="model.layers.0")
    if record is not None:
        # The engine's routed path is the offload cache (M5); stand in the torch NVFP4 reference so
        # the block runs end-to-end here.
        block.ffn.experts = _StubRoutedExperts(_Nvfp4RoutedReference(CKPT), record)
    return block, args


def _errors(actual, expected):
    a, b = actual.float(), expected.float()
    return float((a - b).abs().max()), float((a - b).abs().max() / b.abs().max().clamp_min(1e-6))


# E2M1 codebook, index == 4-bit code (0 and 8 are both +0).
_E2M1 = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
)


def _dequant_nvfp4(packed, scale, scale2):
    """ModelOpt NVFP4: U8 [N, K/2] (2 codes/byte along K) x per-16 E4M3 scale [N, K/16] x fp32."""
    u = packed.view(torch.uint8).to(torch.long)
    codes = torch.stack((_E2M1[u & 0xF], _E2M1[(u >> 4) & 0xF]), dim=-1).flatten(-2)
    return (codes * scale.float().repeat_interleave(16, dim=1) * scale2.float()).to(torch.bfloat16)


class _Nvfp4RoutedReference:
    """Torch NVFP4 routed experts, mirroring the reference ``MoE.forward`` accumulation loop.

    The engine's own routed path is the offload cache (M5); M1 validates the block by feeding the
    same routes through this reference, exactly as the oracle dump did.
    """

    def __init__(self, ckpt: str, prefix: str = "layers.0.ffn.experts", swiglu_limit: float = 10.0):
        with open(os.path.join(ckpt, "model.safetensors.index.json")) as f:
            self.weight_map = json.load(f)["weight_map"]
        self.ckpt = ckpt
        self.prefix = prefix
        self.swiglu_limit = swiglu_limit
        self._cache: dict[int, dict] = {}

    def _read(self, name):
        from safetensors import safe_open

        with safe_open(os.path.join(self.ckpt, self.weight_map[name]), framework="pt") as f:
            return f.get_tensor(name)

    def _expert(self, e: int) -> dict:
        if e not in self._cache:
            mats = {}
            for proj in ("w1", "w2", "w3"):
                base = f"{self.prefix}.{e}.{proj}"
                mats[proj] = _dequant_nvfp4(
                    self._read(f"{base}.weight"),
                    self._read(f"{base}.weight_scale"),
                    self._read(f"{base}.weight_scale_2"),
                )
            self._cache[e] = mats
        return self._cache[e]

    @torch.no_grad()
    def __call__(self, x, weights, indices):
        y = torch.zeros(x.size(0), x.size(-1), dtype=torch.float32, device=x.device)
        for e in torch.unique(indices).tolist():
            mats = self._expert(int(e))
            idx, top = torch.where(indices == e)
            h = x[idx]
            gate = (h @ mats["w1"].to(h.device).T).float()
            up = (h @ mats["w3"].to(h.device).T).float()
            if self.swiglu_limit > 0:
                gate = gate.clamp(max=self.swiglu_limit)
                up = up.clamp(-self.swiglu_limit, self.swiglu_limit)
            h = (torch.nn.functional.silu(gate) * up) * weights[idx, top, None]
            y[idx] += h.bfloat16() @ mats["w2"].to(h.device).T
        return y


class _StubRoutedExperts:
    """Routed-expert stand-in for ``Block.forward`` while the cache path is M5."""

    def __init__(self, ref, record: dict):
        self.ref = ref
        self.record = record

    def routed_forward(self, x, weights, indices):
        self.record["ffn_in"] = x
        routed = self.ref(x, weights, indices)
        self.record["routed_out"] = routed
        return routed


def test_layer0_matches_the_reference_dump():
    block, args = _load_block({})
    assert args.layer_ratio(0) == 0, "layer 0 must be window-only for this comparison"
    dump = torch.load(DUMP, map_location="cuda", weights_only=False)
    x, pre_mix = dump["x"].to("cuda"), dump["pre_mix"].to("cuda")

    trace: dict = {}
    _, ffn_pre = block.forward(x, 0, pre_mix, trace=trace)

    # The window index list is pure integer bookkeeping: it must match EXACTLY.
    assert torch.equal(trace["idxs"], dump["idxs"].to(trace["idxs"].device))

    stage = {"q": 0.05, "kv": 0.05, "o": 0.05}
    report = []
    for key, tol in stage.items():
        abs_err, rel_err = _errors(trace[key], dump[key])
        report.append(f"{key}: abs={abs_err:.3e} rel={rel_err:.3e} (tol {tol})")
        assert rel_err < tol, f"{key} diverged from the reference: " + "; ".join(report)
    abs_err, rel_err = _errors(ffn_pre, dump["ffn_pre"])
    report.append(f"ffn_pre: abs={abs_err:.3e} rel={rel_err:.3e} (tol 0.05)")
    assert rel_err < 0.05, "ffn_pre diverged from the reference: " + "; ".join(report)
    print("\n".join(report))


def test_layer0_output_is_finite_and_shaped():
    """Cheap companion so the GPU can be shared with the production containers."""
    block, args = _load_block({})
    assert block.hc_attn_fn.shape == ((2 + args.hc_mult) * args.hc_mult, args.hc_mult * args.dim)
    x = torch.randn(1, 4, args.hc_mult, args.dim, dtype=torch.bfloat16, device="cuda") * 0.5
    out, ffn_pre = block.forward(x, 0)
    assert out.shape == x.shape and ffn_pre.shape == (1, 4, args.hc_mult)
    assert torch.isfinite(out).all() and torch.isfinite(ffn_pre).all()


def test_layer0_moe_matches_the_reference_dump():
    """Gate / shared expert / routed accumulation against the same layer the oracle ran."""
    record: dict = {}
    block, args = _load_block(record)
    dump = torch.load(DUMP, map_location="cuda", weights_only=False)
    x, pre_mix = dump["x"].to("cuda"), dump["pre_mix"].to("cuda")

    out, ffn_pre = block.forward(x, 0, pre_mix, trace={})
    ffn_in = record["ffn_in"]

    gate_weights, gate_indices = block.ffn.gate.forward(ffn_in)
    assert torch.equal(gate_indices.to(dump["gate_indices"].device), dump["gate_indices"]), (
        "routing differs from the reference: " + str((gate_indices != dump["gate_indices"]).sum())
    )
    shared = block.ffn.shared_experts.forward(ffn_in)
    routed = record["routed_out"]

    report = []
    checks = {
        "gate_weights": (gate_weights, dump["gate_weights"], 0.05),
        "shared_out": (shared, dump["shared_out"], 0.05),
        "routed_out": (routed, dump["routed_out"], 0.05),
        "ffn_out": (routed + shared.float(), dump["ffn_out"], 0.05),
        "out": (out, dump["out"], 0.05),
    }
    for key, (actual, expected, tol) in checks.items():
        abs_err, rel_err = _errors(actual, expected)
        report.append(f"{key}: abs={abs_err:.3e} rel={rel_err:.3e} (tol {tol})")
        assert rel_err < tol, f"{key} diverged from the reference: " + "; ".join(report)
    print("\n".join(report))
