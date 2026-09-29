"""FP4 (e2m1) activation quant+dequant with an **E4M3** scale: CSA2's compressed KV.

The reference ``fp4_quant_kernel`` (``inference/kernel.py:128``) has two scale regimes, picked by
``scale_dtype``:

* ``torch.float8_e8m0fnu`` -- round the block scale to a power of two. The indexer's own K/Q use
  this, and ``dsv4/fp8_linear.fp4_act_quant_inplace`` already implements it.
* ``torch.float8_e4m3fn`` -- round the block scale onto the e4m3 grid instead::

      amax = max(amax, 6 * 2**-9)          # an all-zero group keeps a nonzero scale
      s    = float(e4m3(amax / 6))
      y    = bf16(fp4(clamp(x / s, -6, 6)) * s)   # in place

  which is what ``Attention._compress_kv`` applies to the compressor latent, with block 16 and the
  min-scale floor ``6 * 2**-9 = 0.01171875``.

The e4m3 rounding of the scale is the same one ``e4m3_compat.round_e4m3`` reproduces for the window
KV, whose round-trip already matches the reference bit for bit; only the power-of-two ceiling is
replaced by that e4m3 rounding.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from freetoken.kernel.triton.dsv4.fp8_linear import _round_fp4
from freetoken.kernel.triton.e4m3_compat import round_e4m3

FP4_MAX = 6.0
FP4_MAX_INV = 1.0 / FP4_MAX
# kernel.py:145 -- the smallest amax the E4M3 scale may be derived from
E4M3_MIN_AMAX = FP4_MAX * (2.0**-9)


@triton.jit
def _fp4_e4m3_act_kernel(
    x_ptr, o_ptr, M, N, stride_m, stride_n, stride_om, stride_on,
    MIN_AMAX, INV_MAX,
    BLOCK_M: tl.constexpr, BLOCK: tl.constexpr,
):
    """One scale per (row, BLOCK) group; ``o_ptr`` may alias ``x_ptr`` for a true round-trip."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_n * BLOCK + tl.arange(0, BLOCK)
    m_mask = offs_m < M
    ptrs = x_ptr + offs_m[:, None] * stride_m + offs_k[None, :] * stride_n
    x = tl.load(ptrs, mask=m_mask[:, None], other=0.0).to(tl.float32)
    amax = tl.max(tl.abs(x), axis=1)
    amax = tl.maximum(amax, MIN_AMAX)
    s = round_e4m3(tl.minimum(amax * INV_MAX, 448.0))  # e4m3-rounded scale, back in fp32
    # IEEE division: the e4m3 scale is not a power of two, so the default quotient (`div.full.f32`,
    # which `tl.fdiv(ieee_rounding=True)` also lowers to on this backend) lands a hair off an exact
    # fp4 tie and rounds the wrong way -- 2.5 comes out as 2.5000002 -> 3 instead of 2.
    q = _round_fp4(tl.clamp(tl.div_rn(x, s[:, None]), -6.0, 6.0))
    optrs = o_ptr + offs_m[:, None] * stride_om + offs_k[None, :] * stride_on
    tl.store(optrs, (q * s[:, None]).to(optrs.dtype.element_ty), mask=m_mask[:, None])


def fp4_act_quant_e4m3_inplace(x: torch.Tensor, block: int = 16) -> torch.Tensor:
    """Reference ``fp4_act_quant(x, block, inplace=True, scale_dtype=torch.float8_e4m3fn)``.

    Quantizes and dequantizes ``x`` (bf16) back into itself, one e4m3 scale per ``block`` columns.
    """
    *lead, N = x.shape
    assert N % block == 0, (N, block)
    x2d = x.reshape(-1, N)
    M = x2d.shape[0]
    BLOCK_M = 32
    grid = (triton.cdiv(M, BLOCK_M), N // block)
    _fp4_e4m3_act_kernel[grid](
        x2d, x2d, M, N, x2d.stride(0), x2d.stride(1), x2d.stride(0), x2d.stride(1),
        E4M3_MIN_AMAX, FP4_MAX_INV, BLOCK_M=BLOCK_M, BLOCK=block,
    )
    return x


__all__ = ["fp4_act_quant_e4m3_inplace", "E4M3_MIN_AMAX"]
