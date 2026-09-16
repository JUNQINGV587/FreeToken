"""图安全的 MoE 对齐内核（vLLM moe_align_block_size 的 decode-regime 替代）。

为什么写：vLLM 原生 `moe_align_block_size` 的 CUDA 内核把专家数帽在 1024
（`padded_num_experts must be less than 1024`，vLLM 0.20 实测），而我们的专家空间 =
槽缓存 8800。集成 marlin 路径必须自带对齐。

Decode 域的结构性简化：每专家 token 数 ≤ T ≤ block_m=8 ⇒ **每个专家恰好占一个块**，
且 marlin 不要求专家有序（只要求同专家的 token 连续成块）⇒ 无需排序/扫描，
单程序 O(n²) 比较（n ≤ 128 填充）一次出全部三个输出。固定形状、无 host 读 ⇒
CUDA graph 可捕获。

语义（与 vLLM 对齐）：
  · sorted_token_ids[i] = 展平位置 p（token = p // top_k），填充 = topk_ids.numel()
  · expert_ids[b] = 第 b 块的专家（槽）id
  · num_tokens_post_pad[0] = 不同专家数 × block_m
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _moe_align_decode_kernel(
    ids_ptr,          # [n] int32：展平的路由 id（槽号）
    sorted_ptr,       # [n_pad * BM] int32 输出
    expert_ptr,       # [n_pad] int32 输出
    ntp_ptr,          # [1] int32 输出
    n: tl.constexpr,      # 实际 token×k 数
    n_pad: tl.constexpr,  # 2 的幂，>= n
    BM: tl.constexpr,     # marlin 的 block_m
):
    i = tl.arange(0, n_pad)
    BIG = 1 << 30
    ids = tl.load(ids_ptr + i, mask=i < n, other=BIG)
    # 同专家判等时排除填充位（填充全为 BIG，否则会互相误判同专家）
    valid = (i < n).to(tl.int32)
    eq = (ids[:, None] == ids[None, :]) & (valid[:, None] * valid[None, :] == 1)
    prev = eq & (i[None, :] < i[:, None])       # 存在 j<i 且同专家
    has_prev = tl.sum(prev.to(tl.int32), axis=1) > 0
    is_first = (has_prev == 0) & (i < n)
    # 每个专家的块号 = 其首次出现位置的"首次序数"
    first_pos = tl.min(tl.where(eq, i[None, :], n_pad), axis=1)
    first_blk = tl.cumsum(is_first.to(tl.int32), axis=0) - 1
    block_id = tl.sum(tl.where(first_pos[:, None] == i[None, :], first_blk[None, :], 0), axis=1)
    rank = tl.sum(prev.to(tl.int32), axis=1)   # 同专家内的次序
    pos = block_id * BM + rank
    tl.store(sorted_ptr + pos, i, mask=i < n)
    tl.store(expert_ptr + block_id, ids, mask=is_first)
    nb = tl.sum(is_first.to(tl.int32))
    tl.store(ntp_ptr, nb * BM)


def moe_align_decode(topk_ids: torch.Tensor, block_m: int = 8):
    """图安全对齐。topk_ids [T, k] int32（槽号）。返回 (sorted_token_ids, expert_ids, ntp)。"""
    T, k = topk_ids.shape
    n = T * k
    n_pad = max(16, triton.next_power_of_2(n))
    dev = topk_ids.device
    flat = topk_ids.reshape(-1).contiguous()
    sorted_token_ids = torch.full((n_pad * block_m,), n, dtype=torch.int32, device=dev)
    expert_ids = torch.zeros(n_pad, dtype=torch.int32, device=dev)
    ntp = torch.empty(1, dtype=torch.int32, device=dev)
    _moe_align_decode_kernel[(1,)](
        flat, sorted_token_ids, expert_ids, ntp, n=n, n_pad=n_pad, BM=block_m
    )
    return sorted_token_ids, expert_ids, ntp
