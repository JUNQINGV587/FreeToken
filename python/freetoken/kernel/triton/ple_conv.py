"""Fused PLE dilated depthwise conv + SiLU + residual add + state roll.

Ported from vllm #54517 (``ops/ple.py::_ple_conv_kernel`` /
``_ple_conv_writeback_kernel``, f870b92976), reduced to the two modes FreeToken
runs: **decode** (state roll fused into the main kernel — one program per
request/channel block, no peer still reads the slice) and **prefill** (separate
writeback kernel so every token reads the old state). Spec mode, the token-index
remap, and PDL are intentionally dropped (no speculative decoding per the red
line; PDL is dead on sm89).

RED-LINE NOTE (scoping doc section 4): the kernel accumulates the K=4 taps in
fp32 tap order and applies SiLU after a bf16 rounding of the conv output,
mirroring ``F.conv1d`` + ``F.silu`` dtype boundaries, but the tap accumulation
order is not guaranteed bit-identical to cuDNN's depthwise conv — validated
against ``short_conv_reference`` within bf16 tolerance and pinned deterministic
across runs. Revert with ``FREETOKEN_PLE_FUSED_CONV=0``.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

NULL_STATE_ID = -1


def fused_conv_enabled() -> bool:
    return os.environ.get("FREETOKEN_PLE_FUSED_CONV", "1") != "0"


@triton.jit(do_not_specialize=["num_reqs", "bs_iters"])
def _ple_conv_kernel(
    x_ptr,
    state_ptr,
    w_ptr,
    residual_ptr,
    state_idx_ptr,
    qsl_ptr,
    has_init_ptr,
    num_reqs,
    bs_iters,
    state_bs,
    state_ws,
    state_cs,
    C: tl.constexpr,
    BLOCK_C: tl.constexpr,
    STATE_LEN: tl.constexpr,
    DILATION: tl.constexpr,
    KERNEL_SIZE: tl.constexpr,
    MODE: tl.constexpr,
    HAS_INIT: tl.constexpr,
    NULL_STATE_ID: tl.constexpr,
):
    t = tl.program_id(0)
    pid_c = tl.program_id(1)
    c_offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c_offs < C

    if MODE == "decode":
        r = t
        q_start = t
        j = tl.full([], 0, tl.int32)
        in_range = True
    else:
        # Locate the request containing this token.
        lo = tl.full([], 1, tl.int32)
        hi = tl.full([], num_reqs + 1, tl.int32)
        for _ in range(bs_iters):
            mid = (lo + hi) // 2
            qmid = tl.load(qsl_ptr + mid, mask=mid <= num_reqs, other=0)
            pred = qmid <= t
            lo = tl.where(pred, mid + 1, lo)
            hi = tl.where(pred, hi, mid)
        r = tl.minimum(lo - 1, num_reqs - 1)
        q_start = tl.load(qsl_ptr + r)
        j = (t - q_start).to(tl.int32)
        total_real = tl.load(qsl_ptr + num_reqs)
        in_range = t < total_real

    sid = tl.load(state_idx_ptr + r).to(tl.int64)
    state_ok = sid != NULL_STATE_ID
    sid_safe = tl.where(state_ok, sid, 0)
    if HAS_INIT:
        has_init = tl.load(has_init_ptr + r, mask=state_ok, other=0) != 0
    else:
        has_init = state_ok
    read_state = state_ok & has_init
    out_ok = in_range & state_ok

    # state[sid, c, w] may be a strided view into a shared pool.
    base_state = state_ptr + sid_safe * state_bs
    acc = tl.zeros([BLOCK_C], tl.float32)
    for k in tl.static_range(0, KERNEL_SIZE):
        h = j + DILATION * k
        from_state = h <= STATE_LEN - 1
        state_tap = tl.load(
            base_state + h * state_ws + c_offs * state_cs,
            mask=c_mask & read_state & from_state,
            other=0.0,
        )
        input_t = q_start + h - STATE_LEN
        input_tap = tl.load(
            x_ptr + input_t * C + c_offs,
            mask=c_mask & out_ok & (~from_state),
            other=0.0,
        )
        tap = tl.where(from_state, state_tap, input_tap).to(tl.float32)
        weight = tl.load(
            w_ptr + c_offs * KERNEL_SIZE + k,
            mask=c_mask,
            other=0.0,
        ).to(tl.float32)
        acc += weight * tap

    # F.conv1d materializes its output dtype before SiLU.
    conv = acc.to(residual_ptr.dtype.element_ty).to(tl.float32)
    y = conv * tl.sigmoid(conv)
    conv_output = tl.where(out_ok, y, 0.0).to(residual_ptr.dtype.element_ty)
    residual = tl.load(
        residual_ptr + t * C + c_offs,
        mask=c_mask,
        other=0.0,
    )
    tl.store(
        residual_ptr + t * C + c_offs,
        residual + conv_output,
        mask=c_mask,
    )

    if MODE == "decode":
        # Decode has one program per request and channel block, so no peer can
        # still be reading this state slice when it is updated.
        decode_input = tl.load(
            x_ptr + t * C + c_offs,
            mask=c_mask & state_ok,
            other=0.0,
        )
        for i in tl.static_range(0, STATE_LEN):
            if i < STATE_LEN - 1:
                next_state = tl.load(
                    base_state + (i + 1) * state_ws + c_offs * state_cs,
                    mask=c_mask & state_ok & has_init,
                    other=0.0,
                )
            else:
                next_state = decode_input
            tl.store(
                base_state + i * state_ws + c_offs * state_cs,
                next_state,
                mask=c_mask & state_ok,
            )


# Prefill writes back separately so every token reads the old state.
@triton.jit
def _ple_conv_writeback_kernel(
    x_ptr,
    state_ptr,
    state_idx_ptr,
    qsl_ptr,
    has_init_ptr,
    state_bs,
    state_ws,
    state_cs,
    C: tl.constexpr,
    BLOCK_C: tl.constexpr,
    STATE_LEN: tl.constexpr,
    HAS_INIT: tl.constexpr,
    NULL_STATE_ID: tl.constexpr,
):
    r = tl.program_id(0)
    pid_c = tl.program_id(1)
    c_offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c_offs < C

    sid = tl.load(state_idx_ptr + r).to(tl.int64)
    state_ok = sid != NULL_STATE_ID
    if not state_ok:
        return

    q_start = tl.load(qsl_ptr + r)
    q_end = tl.load(qsl_ptr + r + 1)
    qlen = (q_end - q_start).to(tl.int32)
    if qlen <= 0:
        return

    if HAS_INIT:
        src_ok = state_ok & (tl.load(has_init_ptr + r) != 0)
    else:
        src_ok = state_ok

    base_state = state_ptr + sid * state_bs
    for i in tl.static_range(0, STATE_LEN):
        m = qlen + i
        from_state = m <= STATE_LEN - 1
        state_value = tl.load(
            base_state + m * state_ws + c_offs * state_cs,
            mask=c_mask & from_state & src_ok,
            other=0.0,
        )
        input_t = q_start + m - STATE_LEN
        input_value = tl.load(
            x_ptr + input_t * C + c_offs,
            mask=c_mask & (~from_state),
            other=0.0,
        )
        value = tl.where(from_state, state_value, input_value)
        tl.store(
            base_state + i * state_ws + c_offs * state_cs,
            value,
            mask=c_mask,
        )


def ple_conv(
    inputs: torch.Tensor,
    residual: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weights: torch.Tensor,
    state_indices: torch.Tensor,
    *,
    mode: str,
    dilation: int,
    query_start_loc: torch.Tensor | None = None,
    has_initial_states: torch.Tensor | None = None,
) -> None:
    """Add the short-conv output to ``residual`` in place and roll ``conv_state``.

    ``mode`` is ``"decode"`` (one token per request, roll fused) or ``"prefill"``
    (ragged chunk, separate writeback). ``conv_state`` is ``[slots, C, window]``.
    """
    assert mode in ("decode", "prefill"), f"spec mode intentionally not ported: {mode}"
    BLOCK_C = 512
    T, C = inputs.shape
    K = conv_weights.shape[1]
    state_len = (K - 1) * dilation
    if conv_state.shape[1] != C or conv_state.shape[2] < state_len:
        raise ValueError(
            "conv_state must have shape [slots, channels, window], with "
            f"channels={C} and window >= {state_len}"
        )
    state_bs, state_cs, state_ws = conv_state.stride()

    if mode == "decode":
        num_reqs = T
        binary_search_iters = 1
        has_init_arg = has_initial_states is not None
    else:
        if query_start_loc is None:
            raise ValueError("query_start_loc is required for prefill")
        num_reqs = state_indices.numel()
        binary_search_iters = max(num_reqs, 1).bit_length()
        has_init_arg = has_initial_states is not None

    num_warps = 4 if mode == "prefill" else 8
    grid = (T, triton.cdiv(C, BLOCK_C))
    _ple_conv_kernel[grid](
        inputs,
        conv_state,
        conv_weights,
        residual,
        state_indices,
        query_start_loc if query_start_loc is not None else state_indices,
        has_initial_states if has_initial_states is not None else state_indices,
        num_reqs,
        binary_search_iters,
        state_bs,
        state_ws,
        state_cs,
        C=C,
        BLOCK_C=BLOCK_C,
        STATE_LEN=state_len,
        DILATION=dilation,
        KERNEL_SIZE=K,
        MODE=mode,
        HAS_INIT=has_init_arg,
        NULL_STATE_ID=NULL_STATE_ID,
        num_warps=num_warps,
    )
    if mode == "prefill":
        _ple_conv_writeback_kernel[(num_reqs, triton.cdiv(C, BLOCK_C))](
            inputs,
            conv_state,
            state_indices,
            query_start_loc,
            has_initial_states if has_initial_states is not None else state_indices,
            state_bs,
            state_ws,
            state_cs,
            C=C,
            BLOCK_C=BLOCK_C,
            STATE_LEN=state_len,
            HAS_INIT=has_init_arg,
            NULL_STATE_ID=NULL_STATE_ID,
            num_warps=num_warps,
        )
