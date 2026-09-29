"""The head's vocab all-gather is what makes the TP ranks agree on the next token.

``ParallelLMHead`` holds only this rank's slice of the vocabulary: the reader cuts ``head.weight``
along dim 0 (``freetoken/models/deepseek_v41/weight.py``, ``_TP_VOCAB_KEYS``). Sampling from a slice
is not a smaller version of sampling over the vocabulary -- the softmax denominator and the argmax
are both different, so the ranks pick different tokens and even disagree about whether the sequence
ended at all (V4.1 stops on eos=1, which lives in the first shard only). The TP 2 run showed exactly
that shape of failure: rank 0 answered and went idle, rank 1 took one more decode step and then
blocked forever on the next collective, so the first request after boot succeeded and every later
one hung with the tokenizer queue backed up behind it.

That made the bug a *head* bug, not a scheduler bug: ``Transformer.logits`` applied the quantized
linear directly and skipped the gather that ``ParallelLMHead.forward`` performs for the engine's
batch path. ``gather_logits`` is now the single place that turns sharded logits into the full
vocabulary, and both callers go through it. The layout is the subtle part and is what these
assertions cover: the gathered tensor stacks the ranks along dim 0 (see
``PyNCCLDistributedImpl.all_gather`` / ``TorchDistributedImpl.all_gather``, which multiply
``shape[0]`` by the world size), and the last shard arrives padded to ``num_embeddings_tp`` because
the vocabulary is not required to divide evenly by the TP degree.

The TP degree is process-global and can only be set once (``freetoken.distributed.set_tp_info``), so
each rank runs in its own child process -- the same subprocess re-entry convention as
``tests/models/test_deepseek_v41_tp2_attention.py``. The collective itself is a stub: this test is
about the layout the gather is reassembled with, and a real NCCL group would need GPUs and an
initialized process group to say nothing more.
"""

from __future__ import annotations

import os
import subprocess
import sys

import torch

VOCAB = 5  # odd on purpose: at TP 2 the second shard is ragged (3 + 2, padded back to 3)
HIDDEN = 4
ROWS = 2
PAD = -999.0  # what the loader leaves in the padding columns of a ragged shard


def _shard(full: torch.Tensor, start: int, width: int, num_embeddings_tp: int) -> torch.Tensor:
    """This rank's slice of ``full``, padded to the shard width the way the reader produces it."""
    shard = torch.full((full.shape[0], num_embeddings_tp), PAD)
    shard[:, :width] = full[:, start : start + width]
    return shard


def _report() -> str:
    from freetoken.distributed import set_tp_info
    from freetoken.layers.embedding import ParallelLMHead, VocabParallelEmbedding

    mode = os.environ["FT_GATHER_TEST_MODE"]
    rank = int(mode) if mode != "tp1" else 0
    set_tp_info(rank, 2 if mode != "tp1" else 1)

    embed = VocabParallelEmbedding(VOCAB, HIDDEN)
    head = ParallelLMHead(VOCAB, HIDDEN, tie_word_embeddings=True, tied_embedding=embed)
    full = torch.arange(ROWS * VOCAB, dtype=torch.float32).reshape(ROWS, VOCAB)

    if mode == "tp1":
        # At TP 1 the shard *is* the vocabulary: the head must not enter a collective at all.
        assert head.tp_size == 1

        class _Explode:
            def all_gather(self, x: torch.Tensor) -> torch.Tensor:
                raise AssertionError("TP 1 must not gather logits")

        head._comm = _Explode()
        torch.testing.assert_close(head.gather_logits(full), full)
        return "tp1(identity=(2, 5))"

    assert head.tp_size == 2 and head.num_embeddings_tp == 3, head.num_embeddings_tp
    start, width = head.vocab_range
    local = _shard(full, start, width, head.num_embeddings_tp)
    shards = {
        r: _shard(full, head.num_embeddings_tp * r,
                  min(head.num_embeddings_tp, VOCAB - head.num_embeddings_tp * r),
                  head.num_embeddings_tp)
        for r in range(2)
    }

    class _Comm:
        def all_gather(self, x: torch.Tensor) -> torch.Tensor:
            rows = x.shape[0]
            here = shards[rank][:rows]
            assert x.shape == here.shape, (tuple(x.shape), tuple(here.shape))
            assert torch.equal(x, here), "gathered something that is not its own shard"
            return torch.cat([shards[0][:rows], shards[1][:rows]], dim=0)

    head._comm = _Comm()

    gathered = head.gather_logits(local)
    torch.testing.assert_close(gathered, full)
    # The single-row branch (one decode step) must land on the same vocabulary order.
    one_row = head.gather_logits(local[:1])
    torch.testing.assert_close(one_row, full[:1])
    return (
        f"rank{rank}(range={head.vocab_range} gathered={tuple(gathered.shape)} "
        f"one_row={tuple(one_row.shape)})"
    )


def _child(mode: str) -> str:
    r = subprocess.run(
        [sys.executable, __file__],
        env={**os.environ, "FT_GATHER_TEST_MODE": mode},
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert r.returncode == 0, f"{mode} child failed:\n{r.stdout[-4000:]}\n{r.stderr[-4000:]}"
    return r.stdout.strip()


def test_each_rank_reassembles_the_whole_vocabulary():
    assert _child("0") == "rank0(range=(0, 3) gathered=(2, 5) one_row=(1, 5))"
    assert _child("1") == "rank1(range=(3, 2) gathered=(2, 5) one_row=(1, 5))"


def test_the_gather_is_skipped_at_one_rank():
    assert _child("tp1") == "tp1(identity=(2, 5))"


if __name__ == "__main__":  # subprocess entry: the TP degree is one-shot per process
    print(_report() if "FT_GATHER_TEST_MODE" in os.environ else "")
