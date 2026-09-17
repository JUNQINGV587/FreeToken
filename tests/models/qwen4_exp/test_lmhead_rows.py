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
        self.last_x = x
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
    # Guard the call site itself: prefill must keep each request's LAST-token row
    # (a leading-row slice silently samples the wrong tokens in a packed window),
    # decode keeps the leading batch.size rows of the padded window.
    import inspect

    from freetoken.models.qwen4_exp.model import Qwen4ExpForCausalLM

    src = inspect.getsource(Qwen4ExpForCausalLM.forward)
    assert "batch.size" in src, "lm_head is projecting the whole forward window"
    assert "get_last_indices(batch.size)" in src, "prefill must gather last-token rows"
    assert "hidden[: batch.size]" in src, "decode must keep the leading rows"


class _StubAttn:
    def __init__(self, last: list[int]) -> None:
        self._last = torch.tensor(last, dtype=torch.long)

    def get_last_indices(self, bs: int) -> torch.Tensor:
        return self._last[:bs]


class _StubBatch:
    def __init__(self, size: int, is_prefill: bool, last: list[int]) -> None:
        self.size = size
        self.is_prefill = is_prefill
        self.input_ids = None
        self.attn_metadata = _StubAttn(last)


class _StubInnerModel:
    def __init__(self, window: int, hidden: int) -> None:
        self.window = window
        self.hidden = hidden

    def forward(self, input_ids, batch) -> torch.Tensor:
        # row i holds the constant i, so a gathered row is identifiable by its value
        return torch.arange(self.window, dtype=torch.float32).unsqueeze(1).expand(
            self.window, self.hidden
        )


class _StubCtx:
    def __init__(self, batch: _StubBatch) -> None:
        self.batch = batch


def test_prefill_gathers_last_token_rows_not_leading_rows():
    # Packed prefill of two requests (lengths 5 and 7): the sampler reads rows 4 and
    # 11, not rows 0 and 1. Regression: the leading-row slice sampled the wrong tokens.
    import freetoken.models.qwen4_exp.model as model_mod

    batch = _StubBatch(size=2, is_prefill=True, last=[4, 11])
    head = _RowWiseHead(hidden=4, vocab=8)
    self_obj = type("S", (), {})()
    self_obj.model = _StubInnerModel(window=12, hidden=4)
    self_obj.lm_head = head

    old = model_mod.get_global_ctx
    model_mod.get_global_ctx = lambda: _StubCtx(batch)
    try:
        from freetoken.models.qwen4_exp.model import Qwen4ExpForCausalLM

        Qwen4ExpForCausalLM.forward(self_obj)
    finally:
        model_mod.get_global_ctx = old

    assert head.calls == [(2, 4)], head.calls
    # the head received the last-token rows (values 4 and 11), not rows 0 and 1
    assert head.last_x[:, 0].tolist() == [4.0, 11.0], head.last_x[:, 0].tolist()


def test_lmhead_skips_the_gather_when_rows_are_already_per_request():
    # embedding.py: a prefill head input whose rows already equal bs must not be
    # gathered again (the indices address the ORIGINAL window and would go OOB).
    import inspect

    from freetoken.layers.embedding import ParallelLMHead

    src = inspect.getsource(ParallelLMHead.forward)
    assert "x.shape[0] != bs" in src, "head must not re-gather pre-gathered rows"
