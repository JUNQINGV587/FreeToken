"""``moe_align_block_size`` must not silently drop a large id space.

``num_experts`` here is the id space the routing ids are drawn from, and it is not always a
model's expert count: V4.1's prompt path routes on *slot* ids out of its 1675-row shared cache,
so the align kernel is handed 1675. The vendored sglang kernel walks a fixed ~1024-entry space
and past it writes nothing -- ``num_tokens_post_padded`` comes back 0 (measured: correct at
``num_experts + 1 <= 1024``, 0 at 1025 and up). A zero count makes the prefill GEMM skip every
block, i.e. silently serves zeros; on some shapes the buffers instead hold garbage and the GEMM
indexes a wild bank row (the shipped V4.1 boot did exactly that: ``ntpp = -1694418143``,
``expert_ids ~ 1e9``, illegal memory access).

So the invariant this file pins is the contract itself, for both a model-sized id space (which
the sgl kernel handles) and a pool-sized one (which must take the triton align):

  for every block ``b`` in ``[0, ntpp / block_size)``: every sorted entry is either padding
  (``>= numel``, masked off by the GEMM) or a real token whose routed expert is ``expert_ids[b]``.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

BLOCK = 16


def _assert_valid_plan(topk_ids: torch.Tensor, expert_ids, sorted_ids, ntpp, num_experts: int) -> int:
    numel = topk_ids.numel()
    blocks = int(ntpp) // BLOCK
    assert int(ntpp) > 0, "the align kernel returned an empty plan: the id space was dropped"
    assert int(ntpp) % BLOCK == 0, f"ntpp={int(ntpp)} is not a multiple of the block size"
    assert blocks <= expert_ids.numel(), "the plan claims more blocks than the buffer holds"
    e = expert_ids[:blocks].long()
    assert int(e.min()) >= 0 and int(e.max()) < num_experts, (
        f"expert ids {int(e.min())}..{int(e.max())} outside the id space [0, {num_experts})"
    )
    s = sorted_ids[: int(ntpp)].long()
    real = s < numel
    got = topk_ids.reshape(-1)[s[real]]
    want = e[torch.arange(int(ntpp), device=s.device)[real] // BLOCK]
    assert int((got != want).sum()) == 0, "a sorted slot points at a token routed elsewhere"
    assert int(real.sum()) == numel, "every routed token has to appear exactly once"
    return blocks


@pytest.mark.parametrize("num_experts", [384, 1024, 1025, 1675, 2049])
def test_a_slot_pool_sized_id_space_still_gets_a_plan(num_experts: int):
    from freetoken.moe.fused import moe_align_block_size

    torch.manual_seed(0)
    ids = torch.randint(0, num_experts, (30,), dtype=torch.int32, device="cuda")
    sorted_ids, expert_ids, ntpp = moe_align_block_size(ids, BLOCK, num_experts)
    blocks = _assert_valid_plan(ids, expert_ids, sorted_ids, ntpp.item(), num_experts)
    # 30 ids over a wide space land in ~30 distinct blocks; the point is that planes exist at all
    assert blocks >= 30 // BLOCK


def test_a_wide_id_space_with_many_ids_keeps_every_token():
    """The prefill shape that crashed: 2048 ids drawn from the whole pool."""
    from freetoken.moe.fused import moe_align_block_size

    torch.manual_seed(1)
    num_experts = 1675
    ids = torch.randint(0, num_experts, (2048,), dtype=torch.int32, device="cuda")
    sorted_ids, expert_ids, ntpp = moe_align_block_size(ids, BLOCK, num_experts)
    _assert_valid_plan(ids, expert_ids, sorted_ids, ntpp.item(), num_experts)


def test_the_two_align_kernels_agree_on_a_model_sized_id_space():
    """Below the sgl bound both kernels must produce the same *plan*, so the fallback cannot
    change results for the paths that already ran on sgl."""
    from freetoken.kernel.triton.moe_align import moe_align_block_size as triton_align
    from freetoken.moe.fused import moe_align_block_size

    torch.manual_seed(2)
    num_experts = 384
    ids = torch.randint(0, num_experts, (256,), dtype=torch.int32, device="cuda")
    s1, e1, n1 = moe_align_block_size(ids, BLOCK, num_experts)
    s2, e2, n2 = triton_align(ids, BLOCK, num_experts)
    assert int(n1) == int(n2)
    b = int(n1) // BLOCK
    assert torch.equal(e1[:b].long(), e2[:b].long())
    # the same multiset of real tokens per block, whatever order the atomics landed them in
    for block in range(b):
        lo, hi = block * BLOCK, (block + 1) * BLOCK
        a = sorted(s1[lo:hi].long()[s1[lo:hi] < ids.numel()].tolist())
        c = sorted(s2[lo:hi].long()[s2[lo:hi] < ids.numel()].tolist())
        assert a == c, f"block {block} holds different tokens"
