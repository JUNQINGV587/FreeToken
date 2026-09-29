"""32x32-block fp8 linears (DeepSeek-V4.1) against the reference formula, plus method wiring.

The reference keeps one fp8 scale per 32x32 weight block and per 32 activations, both
e8m0 (``inference/model.py:27`` / ``inference/kernel.py:208-274``). The tests build a
weight whose scale differs on every 32x32 block, so a kernel that applied the scale per
128-wide tile, or read the weight codes as floats, cannot pass:

1. ``block_fp8_linear`` == ``(act_quant(x, 32) * sa) @ W_eff^T`` for prefill (M>1) and
   decode (M==1), with and without a bias.
2. The same numbers under ``FREETOKEN_FORCE_E4M3_EMU=1`` (a fresh process + triton cache:
   the flag is read once at import and is not part of triton's cache key).
3. The method/kernel table: 32-block e8m0 -> ``triton32``, 128-block e8m0 -> ``dsv4``,
   and ``create_weights`` allocates the checkpoint's own block geometry.
"""

from __future__ import annotations

import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

from freetoken.kernel.triton.dsv4.fp8_linear import act_quant_fp8  # noqa: E402
from freetoken.kernel.triton.fp8_block32 import block_fp8_linear  # noqa: E402
from freetoken.layers.quantization import Fp8BlockConfig, QuantKind  # noqa: E402
from freetoken.layers.quantization.linear import Fp8BlockLinearMethod  # noqa: E402
from freetoken.layers.quantization.scheme import fp8_block_scheme  # noqa: E402

FP8 = torch.float8_e4m3fn
E8M0 = torch.float8_e8m0fnu
BLOCK = 32
# fp32 accumulation order differs between the reference and the kernel; the output is
# bf16, so compare on the tensor's scale (the repo's kernel tolerance), not elementwise.
TOL = 5e-2


def _case(N: int, K: int, seed: int = 0, dtype=torch.bfloat16):
    """A weight whose e8m0 scale is a different power of two on every 32x32 block."""
    gen = torch.Generator().manual_seed(seed)
    w = torch.randn(N, K, generator=gen) * 0.1
    codes = torch.randint(-6, 6, (N // BLOCK, K // BLOCK), generator=gen).to(torch.int32) + 127
    scale = torch.exp2(codes.to(torch.float32) - 127.0)
    scale_full = scale.repeat_interleave(BLOCK, 0).repeat_interleave(BLOCK, 1)
    wq = (w / scale_full).clamp(-448.0, 448.0).to(FP8)
    # the fp8 weight's real value, and the codes the engine holds (E8M0 view of uint8)
    w_real = (wq.to(torch.float32).cuda() * scale_full.cuda()).to(dtype)
    codes_u8 = codes.to(torch.uint8).cuda()
    return wq.cuda(), codes_u8.view(E8M0), w_real


def _reference(x: torch.Tensor, w_real: torch.Tensor) -> torch.Tensor:
    """``(A_fp8 * scale_a) @ W_real^T`` with the engine's own activation quantizer."""
    a_fp8, sa = act_quant_fp8(x, BLOCK)
    sa_full = torch.exp2(sa.to(torch.float32) - 127.0).repeat_interleave(BLOCK, 1)
    return ((a_fp8.to(torch.float32) * sa_full) @ w_real.to(torch.float32).T).to(x.dtype)


def _check(N: int, K: int, M: int, seed: int = 0, bias: bool = False):
    wq, scale, w_real = _case(N, K, seed)
    x = (torch.randn(M, K, generator=torch.Generator().manual_seed(seed + 1)) * 0.7).to(torch.bfloat16).cuda()
    b = torch.randn(N, generator=torch.Generator().manual_seed(seed + 2)).to(torch.bfloat16).cuda() if bias else None
    got = block_fp8_linear(x, wq, scale, b)
    ref = _reference(x, w_real)
    if b is not None:
        ref = (ref.to(torch.float32) + b.to(torch.float32)).to(ref.dtype)
    assert got.shape == ref.shape
    err = (got.to(torch.float32) - ref.to(torch.float32)).abs().max().item()
    lim = TOL * ref.to(torch.float32).abs().max().item()
    assert err <= lim, f"N={N} K={K} M={M} bias={bias}: max err {err:.4g} > {lim:.4g}"
    return got, ref


@pytest.mark.parametrize("M", [1, 4, 33, 128])
@pytest.mark.parametrize("N,K", [(512, 5120), (1280, 5120), (32768, 1280), (128, 512)])
def test_matches_the_reference_32_block_formula(M, N, K):
    _check(N, K, M)


def test_bias_is_applied_after_the_scaled_gemm():
    _check(512, 5120, M=8, bias=True)


def test_uint8_codes_agree_with_the_e8m0_view():
    """The readers hand either the raw e8m0 tensor or its uint8 bit view; both must scale alike."""
    wq, scale, _ = _case(512, 5120)
    x = (torch.randn(8, 5120, generator=torch.Generator().manual_seed(7)) * 0.7).to(torch.bfloat16).cuda()
    typed = block_fp8_linear(x, wq, scale, None)
    codes = block_fp8_linear(x, wq, scale.view(torch.uint8), None)
    assert torch.equal(typed, codes)


def _report() -> dict:
    out = {}
    for (N, K, M) in ((512, 5120, 1), (512, 5120, 33), (32768, 1280, 4), (128, 512, 128), (1280, 5120, 8)):
        got, _ = _check(N, K, M, seed=N + K)
        out[f"{N}x{K}x{M}"] = got.float().cpu()
    return out


def test_force_emu_matches_native(tmp_path):
    native = _report()
    path = tmp_path / "emu.pt"
    env = dict(os.environ, FREETOKEN_FORCE_E4M3_EMU="1", TRITON_CACHE_DIR=str(tmp_path / "cache"))
    r = subprocess.run([sys.executable, __file__, str(path)], env=env, capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, f"EMU run failed:\n{r.stdout[-4000:]}\n{r.stderr[-4000:]}"
    emu = torch.load(path)
    for key, ref in native.items():
        err = (ref.float() - emu[key].float()).abs().max().item()
        lim = TOL * ref.float().abs().max().item()
        assert err <= lim, f"{key}: EMU differs from native by {err:.4g} > {lim:.4g}"


# --------------------------------------------------------------------------- method table


class _Layer:
    def __init__(self, in_features, out_features, output_sizes=()):
        self.in_features = in_features
        self.out_features = out_features
        self.output_sizes = output_sizes


def _linear_method(block, scale="e8m0", N=512, K=5120):
    from freetoken.layers.quantization.linear.base import LinearConfig

    cfg = LinearConfig(K, N, (N,), fp8_block_scheme(scale, block))
    return Fp8BlockLinearMethod(cfg), cfg


@pytest.mark.parametrize("block,kernel", [(32, "triton32"), (128, "dsv4")])
def test_kernel_table_picks_by_block_size(block, kernel):
    method, _ = _linear_method(block)
    assert method.kernel.name == kernel


def test_create_weights_follows_the_checkpoints_block():
    for block in (32, 128):
        method, _ = _linear_method(block)
        layer = _Layer(5120, 512, (512,))
        method.create_weights(layer)
        assert layer.weight.shape == (512, 5120) and layer.weight.dtype == FP8
        assert layer.weight_scale_inv.shape == (512 // block, 5120 // block)
        assert layer.weight_scale_inv.dtype == E8M0


if __name__ == "__main__":  # subprocess entry: the same numbers under the caller's environment
    torch.save(_report(), sys.argv[1])
