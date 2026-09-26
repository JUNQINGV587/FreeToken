# Adapt from https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/utils/index.py
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import torch
import triton

from freetoken.kernel.fla.utils import tensor_cache


@tensor_cache
def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]


@tensor_cache
def prepare_chunk_indices(
    cu_seqlens: torch.LongTensor, chunk_size: int
) -> torch.LongTensor:
    # ``build_fla_metadata`` attaches the pinned host cu_seqlens as ``_ft_cpu_shadow``;
    # deriving chunk counts from it avoids a ``.tolist()`` device sync every prefill step
    # (the tensor_cache keys on tensor identity and every step builds a fresh cu_seqlens).
    # Chunk indices are integer metadata: host-derived values are identical to the
    # device's, so numerics are unaffected.
    shadow = getattr(cu_seqlens, "_ft_cpu_shadow", None)
    lens = shadow[1:] - shadow[:-1] if shadow is not None else prepare_lens(cu_seqlens)
    indices = torch.cat(
        [
            torch.arange(n)
            for n in triton.cdiv(lens, chunk_size).tolist()
        ]
    )
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)


@tensor_cache
def prepare_chunk_offsets(
    cu_seqlens: torch.LongTensor, chunk_size: int
) -> torch.LongTensor:
    return torch.cat(
        [cu_seqlens.new_tensor([0]), triton.cdiv(prepare_lens(cu_seqlens), chunk_size)]
    ).cumsum(-1)
