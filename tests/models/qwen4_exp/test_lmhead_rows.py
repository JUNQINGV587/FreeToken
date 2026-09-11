"""lm_head must run on the rows the sampler reads, not the whole forward window.

engine.forward_batch keeps logits[:batch.size]. Projecting the full window costs
M x vocab bf16 and an M/batch.size times larger vocab GEMM for results nobody reads --
at vocab 248,320 an 8192-token prefill chunk is 4.07 GiB of logits to discard.

Slicing must not change the logits that ARE read: lm_head is row-wise, so projecting
a slice equals slicing the projection.
"""

import torch


class _RowWiseHead:
    """Stands in for ParallelLMHead / Fp8ParallelLMHead: any row-wise projection."""

    def __init__(self, hidden: int, vocab: int) -> None:
        torch.manual_seed(0)
        self.w = torch.randn(vocab, hidden)
        self.calls: list[tuple[int, ...]] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls.append(tuple(x.shape))
        return x @ self.w.T


def test_slicing_before_the_head_matches_slicing_after():
    head = _RowWiseHead(hidden=16, vocab=32)
    window, size = 64, 3
    hidden = torch.randn(window, 16)

    full_then_slice = head.forward(hidden)[:size]
    slice_then_project = head.forward(hidden[:size])

    torch.testing.assert_close(slice_then_project, full_then_slice)
    # and the second call really did the smaller GEMM
    assert head.calls == [(window, 16), (size, 16)]


def test_qwen4_exp_forward_slices_to_batch_size():
    # Guard the call site itself: the source must slice before the head, not after.
    import inspect

    from freetoken.models.qwen4_exp.model import Qwen4ExpForCausalLM

    src = inspect.getsource(Qwen4ExpForCausalLM.forward)
    assert "batch.size" in src, "lm_head is projecting the whole forward window"
    assert "self.lm_head.forward(hidden[: batch.size])" in src, src
