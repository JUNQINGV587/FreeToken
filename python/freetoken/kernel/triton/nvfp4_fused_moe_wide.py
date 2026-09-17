"""Wide-load (marlin-style) NVFP4 fused-MoE PREFILL kernel -- bit-identical prototype.

Drop-in replacement for ``_prefill_nvfp4_moe_kernel`` (see
:mod:`freetoken.kernel.triton.nvfp4_fused_moe`): identical grouped-GEMM tile
mapping, identical KB=32 lo->hi double-dot fp32 accumulation order, identical
epilogue. Only the *load / dequant* path changes:

  * packed codes are read as **int32 words** (8 codes / 4 bytes per element)
    instead of byte-at-a-time uint8 gathers;
  * the fp32 LUT gather is replaced by the ``_nvfp4_pair_f32`` ALU bit trick
    (validated bit-identical upstream, see ``nvfp4_linear.py``);
  * each fp8 block scale is loaded once per word instead of 8x per byte;
  * activations are read as one contiguous ``[BM, 2*KB]`` vector per K-iter
    and split even/odd in registers instead of two stride-2 loads.

Bit-contract invariants (spec ``202609-nvfp4-prefill-kernel-spec.md``):
  C1  ``b = bf16(RN(fp32 e2m1 x fp32 e4m3_scale))``: the bit trick yields
      fp32 ``e2m1 * 2^-14`` exactly; multiplying by ``scale * 2^14`` (also
      exact, power of two) rounds the same real product once, so the bf16
      cast is bit-identical to the LUT path.
  C2  ``BLOCK_SIZE_KB == 32``, ``acc += dot(a_lo, b_lo)`` *then*
      ``acc += dot(a_hi, b_hi)`` per K-iter, fp32 accumulator: unchanged.
  C3  epilogue ``(acc x g) [x topk_weight] -> bf16``: copied verbatim.
  C4  host-side ``moe_align_block_size`` grouping: reused unchanged.
"""

from __future__ import annotations

import triton
import triton.language as tl

from freetoken.kernel.triton.e4m3_compat import e4m3_native_cx, e4m3_u8_to_f32
from freetoken.kernel.triton.nvfp4_linear import _nvfp4_pair_f32


@triton.jit
def _prefill_nvfp4_moe_wide_kernel(
    a_ptr,             # [M, K] activations
    packed_ptr,        # [S, N, K // 8] int32 (8 fp4 codes per word, nibble j -> k=8*w+j)
    scale_ptr,         # [S, N, K // 16] fp8-e4m3
    global_ptr,        # [S, N] fp16
    c_ptr,             # [num_valid_tokens, N] output (flat over M*top_k)
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,    # cache slot per M-block
    num_tokens_post_padded_ptr,
    N,
    K,
    EM,
    num_valid_tokens,
    stride_am, stride_ak,
    stride_pe, stride_pn, stride_pkw,
    stride_se, stride_sn, stride_sblk,
    stride_ge, stride_gn,
    stride_cm, stride_cn,
    stride_tw,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_KB: tl.constexpr,  # bytes per K-iter; MUST stay 32 (bit contract C2)
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
):
    # C2 is load-bearing, not a comment: KB=32 + lo->hi double-dot order IS the
    # fp32 reduction order, i.e. the output bits.
    tl.static_assert(BLOCK_SIZE_KB == 32, "bit contract C2: BLOCK_SIZE_KB must stay 32")
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    BLOCK_KW: tl.constexpr = BLOCK_SIZE_KB // 4  # int32 words per K-iter
    offs_kw = tl.arange(0, BLOCK_KW)
    offs_kb = tl.arange(0, BLOCK_SIZE_KB)
    offs_ka = tl.arange(0, BLOCK_SIZE_KB * 2)  # activation k-offsets per K-iter
    a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am + offs_ka[None, :] * stride_ak)

    slot = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    packed_base = packed_ptr + slot * stride_pe + offs_bn[None, :] * stride_pn
    scale_base = scale_ptr + slot * stride_se + offs_bn[None, :] * stride_sn

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    K_WORDS = K // 8
    for kw in range(0, tl.cdiv(K_WORDS, BLOCK_KW)):
        widx = kw * BLOCK_KW + offs_kw
        word_mask = widx < K_WORDS

        word = tl.load(packed_base + widx[:, None] * stride_pkw, mask=word_mask[:, None], other=0)
        # 8 codes/word span bytes 4w..4w+3 -> scale block (4w)//8 == w//2.
        s_ptrs = scale_base + (widx[:, None] // 2) * stride_sblk
        if e4m3_native_cx():
            scale_w = tl.load(s_ptrs, mask=word_mask[:, None], other=0.0).to(tl.float32) * 16384.0
        else:
            scale_w = e4m3_u8_to_f32(tl.load(s_ptrs, mask=word_mask[:, None], other=0)) * 16384.0

        # Broadcast-expand the word tile [KW, BN] back to byte rows [KB, BN]
        # (row 4w+t = word w), then dequant elementwise. This shape -- NOT a
        # join/permute/reshape interleave -- is what keeps triton's layout
        # inference on the same kWidth=4 dot-operand layout as the LUT kernel
        # (kWidth=8 there changes the mma fragment mapping -> 1-ulp drift).
        wdup = tl.reshape(
            tl.broadcast_to(word[:, None, :], (BLOCK_KW, 4, BLOCK_SIZE_N)),
            (BLOCK_SIZE_KB, BLOCK_SIZE_N),
        )
        shift = ((offs_kb % 4) * 8)[:, None]  # byte t of word w -> bits [8t+3:8t]
        lo = (wdup >> shift) & 0xF
        hi = (wdup >> (shift + 4)) & 0xF
        blo32, bhi32 = _nvfp4_pair_f32(lo | (hi << 16))
        scale = tl.reshape(
            tl.broadcast_to(scale_w[:, None, :], (BLOCK_KW, 4, BLOCK_SIZE_N)),
            (BLOCK_SIZE_KB, BLOCK_SIZE_N),
        )
        # RN((e2m1 * 2^-14) * (scale * 2^14)) == RN(e2m1 * scale) -> C1.
        b_lo = blo32 * scale  # [KB, BN] fp32, row = byte index (lo nibble = even k)
        b_hi = bhi32 * scale

        a_vec = tl.load(
            a_ptrs,
            mask=token_mask[:, None] & ((kw * BLOCK_SIZE_KB * 2 + offs_ka)[None, :] < K),
            other=0.0,
        )  # [BM, 2*KB] contiguous
        a_lo, a_hi = tl.split(tl.reshape(a_vec, (BLOCK_SIZE_M, BLOCK_SIZE_KB, 2)))
        accumulator += tl.dot(a_lo, b_lo.to(a_lo.dtype))
        accumulator += tl.dot(a_hi, b_hi.to(a_hi.dtype))

        a_ptrs += BLOCK_SIZE_KB * 2 * stride_ak

    g = tl.load(global_ptr + slot * stride_ge + offs_bn * stride_gn).to(tl.float32)
    accumulator = accumulator * g[None, :]

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token * stride_tw, mask=token_mask, other=0)
        accumulator = accumulator * moe_weight[:, None]

    accumulator = accumulator.to(compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


__all__ = [
    "_prefill_nvfp4_moe_wide_kernel",
]
