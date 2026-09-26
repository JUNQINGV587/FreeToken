# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from vLLM (vllm/models/qwen4_exp/nvidia/ops/qsa.py)
"""QSA block scoring over the paged compressed-key slab."""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _qsa_mqa_paged_kernel(
    q_ptr,
    k_cache_ptr,
    page_table_ptr,
    token_to_req_ptr,
    query_positions_ptr,
    sequence_lengths_ptr,
    visible_blocks_ptr,
    logits_ptr,
    stride_q_row,
    stride_q_head,
    stride_q_dim,
    stride_cache_block,
    stride_cache_token,
    stride_cache_dim,
    stride_table_req,
    stride_table_page,
    stride_logits_row,
    num_rows,
    num_columns,
    num_pages,
    num_requests,
    score_divisor,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    TILES_PER_PROG: tl.constexpr,
    STAGES: tl.constexpr,
    MAX_N: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    dims = tl.arange(0, BLOCK_D)
    heads = tl.arange(0, MAX_N)
    request = tl.load(token_to_req_ptr + row)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    query_position = tl.load(query_positions_ptr + row)
    sequence_length = tl.load(
        sequence_lengths_ptr + safe_request,
        mask=(request >= 0) & (request < num_requests),
        other=0,
    )
    visible = tl.minimum(
        (query_position + 1) // COMPRESS_RATIO,
        sequence_length // COMPRESS_RATIO,
    )
    if tl.program_id(1) == 0:
        tl.store(visible_blocks_ptr + row, visible)
    tile_start = tl.program_id(1) * TILES_PER_PROG
    # Top-k is bounded by visible_blocks, so columns beyond it need no value.
    if tile_start * BLOCK_N >= visible:
        return
    tile_end = tl.minimum(tile_start + TILES_PER_PROG, tl.cdiv(visible, BLOCK_N))
    tile_end = tl.minimum(tile_end, tl.cdiv(num_columns, BLOCK_N))

    # Pad the small head axis to a tensor-core-compatible N dimension.
    query = tl.load(
        q_ptr
        + row * stride_q_row
        + heads[None, :] * stride_q_head
        + dims[:, None] * stride_q_dim,
        mask=(heads[None, :] < NUM_HEADS) & (dims[:, None] < HEAD_DIM),
        other=0.0,
    )
    column_offsets = tl.arange(0, BLOCK_N)
    for tile in tl.range(tile_start, tile_end, num_stages=STAGES):
        columns = tile * BLOCK_N + column_offsets
        live = columns < visible
        logical_page = tl.minimum(columns // PAGE_SIZE, PAGE_TABLE_WIDTH - 1)
        page_offset = columns % PAGE_SIZE
        physical_page = tl.load(
            page_table_ptr
            + safe_request * stride_table_req
            + logical_page * stride_table_page,
            mask=live,
            other=-1,
        )
        page_valid = live & (physical_page >= 0) & (physical_page < num_pages)
        # physical_page * block stride can overflow int32 for large caches.
        safe_physical_page = tl.maximum(physical_page, 0).to(tl.int64)
        keys = tl.load(
            k_cache_ptr
            + safe_physical_page[:, None] * stride_cache_block
            + page_offset[:, None] * stride_cache_token
            + dims[None, :] * stride_cache_dim,
            mask=page_valid[:, None] & (dims[None, :] < HEAD_DIM),
            other=0.0,
            eviction_policy="evict_first",
        )
        scores = tl.dot(keys, query, out_dtype=tl.float32)
        scores = tl.where(heads[None, :] < NUM_HEADS, tl.maximum(scores, 0.0), 0.0)
        score = tl.sum(scores, axis=1) / score_divisor
        tl.store(
            logits_ptr + row * stride_logits_row + columns,
            tl.where(page_valid, score, -float("inf")),
            mask=live & (columns < num_columns),
        )


def qsa_mqa_paged(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    compress_ratio: int,
    logits: torch.Tensor,
    visible_blocks: torch.Tensor,
    score_scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute QSA scores directly from a paged compressed-key cache."""

    if q.ndim != 3 or q.shape[1] <= 0 or q.shape[2] <= 0:
        raise ValueError("QSA query must be [rows, heads, head_dim]")
    if k_cache.ndim != 4 or k_cache.shape[2] != 1:
        raise ValueError("QSA cache must be [pages, page_size, 1, head_dim]")
    if k_cache.shape[3] != q.shape[2]:
        raise ValueError("QSA query and cache dimensions must match")
    if token_to_req.shape != (q.shape[0],) or query_positions.shape != (q.shape[0],):
        raise ValueError("QSA request mapping and positions must match query rows")
    if sequence_lengths.shape != (page_table.shape[0],):
        raise ValueError("QSA sequence lengths must match page-table requests")
    score_divisor = math.sqrt(q.shape[2]) if score_scale is None else score_scale
    columns = logits.shape[1]
    if not q.shape[0] or not columns:
        return logits, visible_blocks
    BLOCK_N = 64
    BLOCK_D = max(16, triton.next_power_of_2(q.shape[2]))
    MAX_N = max(16, triton.next_power_of_2(q.shape[1]))
    # Tuned on GB300: larger row batches provide enough parallelism to reuse Q.
    tiles_per_program = 1 if q.shape[0] <= 32 else 8
    _qsa_mqa_paged_kernel[
        (q.shape[0], triton.cdiv(columns, BLOCK_N * tiles_per_program))
    ](
        q,
        k_cache,
        page_table,
        token_to_req,
        query_positions,
        sequence_lengths,
        visible_blocks,
        logits,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(3),
        page_table.stride(0),
        page_table.stride(1),
        logits.stride(0),
        q.shape[0],
        columns,
        k_cache.shape[0],
        page_table.shape[0],
        float(score_divisor),
        PAGE_SIZE=k_cache.shape[1],
        PAGE_TABLE_WIDTH=page_table.shape[1],
        NUM_HEADS=q.shape[1],
        HEAD_DIM=q.shape[2],
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        TILES_PER_PROG=tiles_per_program,
        STAGES=2,
        MAX_N=MAX_N,
        COMPRESS_RATIO=compress_ratio,
        num_warps=2,
    )
    return logits, visible_blocks


@triton.jit
def _qsa_mqa_paged_prefill_kernel(
    q_ptr,
    k_cache_ptr,
    page_table_ptr,
    cu_seqlens_ptr,
    query_positions_ptr,
    sequence_lengths_ptr,
    visible_blocks_ptr,
    logits_ptr,
    stride_q_row,
    stride_q_head,
    stride_q_dim,
    stride_cache_block,
    stride_cache_token,
    stride_cache_dim,
    stride_table_req,
    stride_table_page,
    stride_logits_row,
    num_rows,
    query_offset,
    num_columns,
    num_pages,
    score_divisor,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    TILE_R: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    K_TILES: tl.constexpr,
    STAGES: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
) -> None:
    # Prefill/extend scoring with TILE_R rows packed into the dot's N dimension
    # (vllm #54513): the per-row kernel's [BLOCK_N, 16] mma is tensor-core-starved;
    # [BLOCK_N, TILE_R * HEADS_PAD] keeps the pipes full on long extends.
    request = tl.program_id(0)
    HEADS_PAD: tl.constexpr = triton.next_power_of_2(NUM_HEADS)
    cu_base = tl.load(cu_seqlens_ptr)
    query_end = query_offset + num_rows
    # The chunk [query_offset, query_end) may cut through a request: clamp.
    request_start = tl.maximum(tl.load(cu_seqlens_ptr + request) - cu_base, query_offset)
    request_end = tl.minimum(tl.load(cu_seqlens_ptr + request + 1) - cu_base, query_end)
    absolute_row_start = request_start + tl.program_id(1) * TILE_R
    if absolute_row_start >= request_end:
        return

    lanes = tl.arange(0, TILE_R)
    absolute_rows = absolute_row_start + lanes
    valid_rows = absolute_rows < request_end
    positions = tl.load(query_positions_ptr + absolute_rows, mask=valid_rows, other=0)
    seq_len = tl.load(sequence_lengths_ptr + request)
    visible = tl.minimum(
        (positions + 1) // COMPRESS_RATIO,
        seq_len // COMPRESS_RATIO,
    )
    rows = absolute_rows - query_offset
    if tl.program_id(2) == 0:
        tl.store(visible_blocks_ptr + rows, visible, mask=valid_rows)
    max_visible = tl.max(tl.where(valid_rows, visible, 0), axis=0)
    k_tile_start = tl.program_id(2) * K_TILES
    # Top-k is bounded by visible blocks, so columns beyond it need no value.
    if k_tile_start * BLOCK_N >= max_visible:
        return
    k_tile_end = tl.minimum(k_tile_start + K_TILES, tl.cdiv(max_visible, BLOCK_N))
    k_tile_end = tl.minimum(k_tile_end, tl.cdiv(num_columns, BLOCK_N))

    dims = tl.arange(0, BLOCK_D)
    m = tl.arange(0, TILE_R * HEADS_PAD)
    q_row_offsets = m // HEADS_PAD
    heads = m % HEADS_PAD
    q_rows = absolute_row_start + q_row_offsets
    query = tl.load(
        q_ptr
        + q_rows[None, :] * stride_q_row
        + heads[None, :] * stride_q_head
        + dims[:, None] * stride_q_dim,
        mask=(heads[None, :] < NUM_HEADS)
        & (q_rows[None, :] < request_end)
        & (dims[:, None] < HEAD_DIM),
        other=0.0,
    )
    column_offsets = tl.arange(0, BLOCK_N)
    for tile in tl.range(k_tile_start, k_tile_end, num_stages=STAGES):
        columns = tile * BLOCK_N + column_offsets
        live = columns < max_visible
        logical_page = tl.minimum(columns // PAGE_SIZE, PAGE_TABLE_WIDTH - 1)
        page_offset = columns % PAGE_SIZE
        physical_page = tl.load(
            page_table_ptr
            + request * stride_table_req
            + logical_page * stride_table_page,
            mask=live,
            other=-1,
        )
        page_valid = live & (physical_page >= 0) & (physical_page < num_pages)
        # physical_page * block stride can overflow int32 for large caches.
        safe_physical_page = tl.maximum(physical_page, 0).to(tl.int64)
        keys = tl.load(
            k_cache_ptr
            + safe_physical_page[:, None] * stride_cache_block
            + page_offset[:, None] * stride_cache_token
            + dims[None, :] * stride_cache_dim,
            mask=page_valid[:, None] & (dims[None, :] < HEAD_DIM),
            other=0.0,
            eviction_policy="evict_first",
        )
        scores = tl.dot(keys, query, out_dtype=tl.float32)
        scores = tl.reshape(scores, (BLOCK_N, TILE_R, HEADS_PAD))
        # Padded heads loaded query=0.0, so their dot is exactly 0 and relu keeps it 0 --
        # no head mask needed inside the sum.
        score = tl.sum(tl.maximum(scores, 0.0), axis=2) / score_divisor
        # live is tile-level (max_visible); each row's own bound is its visible count.
        store_mask = (
            valid_rows[None, :]
            & (columns[:, None] < visible[None, :])
            & (columns[:, None] < num_columns)
        )
        tl.store(
            logits_ptr + rows[None, :] * stride_logits_row + columns[:, None],
            tl.where(page_valid[:, None], score, -float("inf")),
            mask=store_mask,
        )


def qsa_mqa_paged_prefill(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    cu_seqlens: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    compress_ratio: int,
    logits: torch.Tensor,
    visible_blocks: torch.Tensor,
    *,
    query_offset: int,
    num_rows: int,
    max_query_len: int,
    score_scale: float | None = None,
) -> None:
    """Prefill/extend counterpart of :func:`qsa_mqa_paged` (vllm #54513 port).

    Same per-element math (relu of the q·k dot, summed over heads, divided by
    ``score_divisor``) but TILE_R rows share one tensor-core dot. ``q``,
    ``cu_seqlens``, ``query_positions`` address the WHOLE packed batch; the call
    scores the chunk ``[query_offset, query_offset + num_rows)`` into the
    chunk-sized ``logits``/``visible_blocks`` buffers. The chunk may cut through
    a request -- the kernel clamps each request's row range to the chunk.
    """

    if q.ndim != 3 or q.shape[1] <= 0 or q.shape[2] <= 0:
        raise ValueError("QSA query must be [rows, heads, head_dim]")
    if k_cache.ndim != 4 or k_cache.shape[2] != 1:
        raise ValueError("QSA cache must be [pages, page_size, 1, head_dim]")
    if k_cache.shape[3] != q.shape[2]:
        raise ValueError("QSA query and cache dimensions must match")
    if cu_seqlens.shape != (page_table.shape[0] + 1,):
        raise ValueError("QSA cu_seqlens must be [requests + 1]")
    if not 0 <= query_offset <= query_offset + num_rows <= q.shape[0]:
        raise ValueError("QSA prefill chunk outside the packed query rows")
    if logits.shape != (num_rows, logits.shape[1]) or visible_blocks.shape != (num_rows,):
        raise ValueError("QSA prefill outputs must be chunk-sized")
    if num_rows == 0 or logits.shape[1] == 0:
        return
    score_divisor = math.sqrt(q.shape[2]) if score_scale is None else score_scale
    columns = logits.shape[1]
    TILE_R = 64
    BLOCK_N = 64
    K_TILES = 16
    BLOCK_D = max(16, triton.next_power_of_2(q.shape[2]))
    grid = (
        page_table.shape[0],
        triton.cdiv(min(num_rows, max_query_len), TILE_R),
        triton.cdiv(columns, BLOCK_N * K_TILES),
    )
    _qsa_mqa_paged_prefill_kernel[grid](
        q,
        k_cache,
        page_table,
        cu_seqlens,
        query_positions,
        sequence_lengths,
        visible_blocks,
        logits,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(3),
        page_table.stride(0),
        page_table.stride(1),
        logits.stride(0),
        num_rows,
        query_offset,
        columns,
        k_cache.shape[0],
        float(score_divisor),
        PAGE_SIZE=k_cache.shape[1],
        PAGE_TABLE_WIDTH=page_table.shape[1],
        NUM_HEADS=q.shape[1],
        HEAD_DIM=q.shape[2],
        TILE_R=TILE_R,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        K_TILES=K_TILES,
        STAGES=2,
        COMPRESS_RATIO=compress_ratio,
        num_warps=4,
    )


__all__ = ["qsa_mqa_paged", "qsa_mqa_paged_prefill"]
