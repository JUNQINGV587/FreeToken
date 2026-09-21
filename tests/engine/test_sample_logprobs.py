from types import SimpleNamespace

import pytest
import torch

from freetoken.engine.sample import BatchSamplingArgs, Sampler


def test_compute_logprobs_matches_raw_log_softmax_and_sorted_top() -> None:
    sampler = Sampler(torch.device("cpu"), vocab_size=4)

    logits = torch.tensor(
        [
            [2.0, 0.0, 1.0, -1.0],
            [0.0, -1.0, 1.0, 3.0],
            [1.0, 2.0, 3.0, 4.0],
        ],
        dtype=torch.float32,
    )
    sampled_tokens = torch.tensor([2, 3, 0], dtype=torch.long)
    args = BatchSamplingArgs(
        temperatures=torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32),
        logprob_rows=torch.tensor([True, False, True], dtype=torch.bool),
        max_top_logprobs=3,
    )

    result = sampler.compute_logprobs(logits, sampled_tokens, args)

    assert result is not None
    chosen_logprobs, top_ids, top_logprobs = result
    expected = torch.log_softmax(logits.float(), dim=-1)

    assert torch.isclose(chosen_logprobs[0], expected[0, sampled_tokens[0]])
    assert torch.isnan(chosen_logprobs[1])
    assert torch.isclose(chosen_logprobs[2], expected[2, sampled_tokens[2]])

    expected_top0 = torch.topk(expected[0], k=3)
    expected_top2 = torch.topk(expected[2], k=3)
    assert torch.equal(top_ids[0], expected_top0.indices)
    assert torch.allclose(top_logprobs[0], expected_top0.values)
    assert torch.equal(top_ids[2], expected_top2.indices)
    assert torch.allclose(top_logprobs[2], expected_top2.values)

    assert torch.equal(top_ids[1], torch.full((3,), -1, dtype=torch.int32))
    assert torch.isneginf(top_logprobs[1]).all()
    assert top_ids.shape == (3, 3)
    assert top_logprobs.shape == (3, 3)


def test_compute_logprobs_ignores_temperatures() -> None:
    sampler = Sampler(torch.device("cpu"), vocab_size=4)

    logits = torch.tensor(
        [
            [0.5, 1.0, 2.0, 3.0],
            [4.0, 3.0, 2.0, 1.0],
        ],
        dtype=torch.float32,
    )
    sampled_tokens = torch.tensor([1, 2], dtype=torch.long)
    logprob_rows = torch.tensor([True, True], dtype=torch.bool)

    cold = BatchSamplingArgs(
        temperatures=torch.full((2,), 0.3, dtype=torch.float32),
        logprob_rows=logprob_rows,
        max_top_logprobs=2,
    )
    hot = BatchSamplingArgs(
        temperatures=torch.full((2,), 2.5, dtype=torch.float32),
        logprob_rows=logprob_rows,
        max_top_logprobs=2,
    )

    cold_out = sampler.compute_logprobs(logits, sampled_tokens, cold)
    hot_out = sampler.compute_logprobs(logits, sampled_tokens, hot)

    assert cold_out is not None and hot_out is not None
    cold_chosen, cold_top_ids, cold_top_logprobs = cold_out
    hot_chosen, hot_top_ids, hot_top_logprobs = hot_out

    assert torch.allclose(cold_chosen, hot_chosen)
    assert torch.equal(cold_top_ids, hot_top_ids)
    assert torch.allclose(cold_top_logprobs, hot_top_logprobs)


def test_compute_logprobs_returns_none_without_requested_rows() -> None:
    sampler = Sampler(torch.device("cpu"), vocab_size=4)
    logits = torch.zeros((2, 4), dtype=torch.float32)
    sampled_tokens = torch.tensor([0, 1], dtype=torch.long)
    args = BatchSamplingArgs(
        temperatures=torch.ones(2),
        logprob_rows=torch.zeros(2, dtype=torch.bool),
        max_top_logprobs=2,
    )

    assert sampler.compute_logprobs(logits, sampled_tokens, args) is None


def test_compute_logprobs_handles_zero_top_logprobs() -> None:
    sampler = Sampler(torch.device("cpu"), vocab_size=4)
    logits = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [4.0, 3.0, 2.0, 1.0],
            [0.1, 0.2, 0.3, 0.4],
        ],
        dtype=torch.float32,
    )
    sampled_tokens = torch.tensor([3, 0, 1], dtype=torch.long)
    args = BatchSamplingArgs(
        temperatures=torch.full((3,), 1.0),
        logprob_rows=torch.tensor([True, False, True], dtype=torch.bool),
        max_top_logprobs=0,
    )

    result = sampler.compute_logprobs(logits, sampled_tokens, args)

    assert result is not None
    chosen_logprobs, top_ids, top_logprobs = result
    expected = torch.log_softmax(logits, dim=-1)
    assert torch.isclose(chosen_logprobs[0], expected[0, 3])
    assert torch.isnan(chosen_logprobs[1])
    assert torch.isclose(chosen_logprobs[2], expected[2, 1])
    assert top_ids.shape == (3, 0)
    assert top_logprobs.shape == (3, 0)


def test_compute_logprobs_with_prepare_keeps_max_top_logprobs_clamped(monkeypatch) -> None:
    # prepare() pins host tensors for the H2D copy; pinning needs a CUDA context,
    # so stub the transfer helper to keep this test host-agnostic.
    import freetoken.engine.sample as sample_mod

    monkeypatch.setattr(
        sample_mod, "make_device_tensor",
        lambda data, dtype, device: torch.tensor(data, dtype=dtype),
    )
    sampler = Sampler(torch.device("cpu"), vocab_size=5)

    # The fork's prepare() also builds penalty rows and reads req.can_decode.
    batch = SimpleNamespace(
        reqs=[
            SimpleNamespace(
                can_decode=True,
                sampling_params=SimpleNamespace(
                    logprobs=True,
                    top_logprobs=17,
                    is_greedy=True,
                    presence_penalty=0.0,
                    frequency_penalty=0.0,
                ),
            ),
            SimpleNamespace(
                can_decode=True,
                sampling_params=SimpleNamespace(
                    logprobs=False,
                    top_logprobs=0,
                    is_greedy=True,
                    presence_penalty=0.0,
                    frequency_penalty=0.0,
                ),
            ),
        ]
    )
    args = sampler.prepare(batch)
    assert args.max_top_logprobs == sampler.vocab_size

    logits = torch.tensor(
        [
            [1.0, 0.0, -1.0, 0.5, 2.0],
            [2.0, 1.0, 0.0, -1.0, -2.0],
        ],
        dtype=torch.float32,
    )
    sampled_tokens = torch.tensor([4, 0], dtype=torch.long)

    result = sampler.compute_logprobs(logits, sampled_tokens, args)
    assert result is not None
    chosen_logprobs, top_ids, top_logprobs = result

    assert chosen_logprobs.shape == (2,)
    assert torch.isclose(chosen_logprobs[0], torch.log_softmax(logits[0], dim=-1)[4])
    assert torch.isnan(chosen_logprobs[1])

    assert top_ids.shape == (2, sampler.vocab_size)
    assert top_logprobs.shape == (2, sampler.vocab_size)
    assert torch.equal(top_ids[1], torch.full((sampler.vocab_size,), -1, dtype=torch.int32))
    assert torch.isneginf(top_logprobs[1]).all()


def test_apply_penalties_is_the_identity_without_penalty_rows() -> None:
    sampler = Sampler(torch.device("cpu"), vocab_size=3)
    logits = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.bfloat16)

    assert sampler.apply_penalties(logits, BatchSamplingArgs(temperatures=None)) is logits


def test_penalized_logprobs_come_from_the_distribution_that_was_sampled() -> None:
    # Frequency/presence penalties change the distribution the token is drawn from,
    # so the reported logprob must be computed on the penalized logits (engine.py
    # applies them once and feeds both sample() and compute_logprobs()).
    sampler = Sampler(torch.device("cpu"), vocab_size=4)
    logits = torch.tensor([[1.0, 2.0, 0.5, -1.0]], dtype=torch.float32)
    counts = torch.tensor([3, 0, 1, 2], dtype=torch.int32)
    args = BatchSamplingArgs(
        temperatures=None,
        penalties=[(0, counts, 0.5, 1.0)],
        logprob_rows=torch.tensor([True], dtype=torch.bool),
        max_top_logprobs=2,
    )

    penalized = sampler.apply_penalties(logits, args)
    assert torch.equal(penalized, logits - (1.0 * counts + 0.5 * (counts > 0)))

    chosen_logprobs, top_ids, _ = sampler.compute_logprobs(
        penalized, torch.tensor([0], dtype=torch.long), args
    )
    expected = torch.log_softmax(penalized, dim=-1)
    assert torch.isclose(chosen_logprobs[0], expected[0, 0])
    assert not torch.isclose(
        chosen_logprobs[0], torch.log_softmax(logits, dim=-1)[0, 0]
    )
    # Penalties reorder the tail: token 2 (barely penalized) overtakes token 0.
    assert [int(t) for t in top_ids[0]] == [1, 2]
    assert torch.log_softmax(logits, dim=-1).topk(2).indices.tolist() == [[1, 0]]


def test_logprob_rows_do_not_change_the_sampled_tokens() -> None:
    # The red line for the whole feature: asking for logprobs must not perturb the
    # sampling path. sample() reads logits only, so the extra rows change nothing;
    # the token sequence level version of this runs on the GPU suite.
    sampler = Sampler(torch.device("cpu"), vocab_size=4)
    logits = torch.tensor([[1.0, 2.0, 0.5, -1.0], [0.0, 1.0, 3.0, 2.0]], dtype=torch.float32)

    plain = sampler.sample(logits, BatchSamplingArgs(temperatures=None))
    with_logprobs = sampler.sample(
        logits,
        BatchSamplingArgs(
            temperatures=None,
            logprob_rows=torch.tensor([True, True], dtype=torch.bool),
            max_top_logprobs=4,
        ),
    )

    assert torch.equal(plain, with_logprobs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_cuda_logprob_pass_does_not_perturb_the_sampled_tokens() -> None:
    # The red line on the real kernels: the same seeded draw over the probability path
    # (mixed top-k/top-p rows plus a penalty row) must return identical tokens with and
    # without the logprob pass. Each arm gets its own penalty counters - sample()
    # increments them for the token it drew, so sharing one tensor across arms would
    # compare two different distributions instead of the thing under test.
    vocab = 8192
    torch.manual_seed(0)
    sampler = Sampler(torch.device("cuda"), vocab_size=vocab)
    logits = torch.randn(4, vocab, device="cuda", dtype=torch.bfloat16)
    temperatures = torch.tensor([0.7, 1.0, 0.5, 1.3], device="cuda", dtype=torch.float32)
    top_k = torch.tensor([vocab, 50, vocab, 200], device="cuda", dtype=torch.int32)
    top_p = torch.tensor([1.0, 1.0, 0.9, 0.95], device="cuda", dtype=torch.float32)

    def fresh_args(row_temperatures=temperatures, **extra) -> BatchSamplingArgs:
        counts = torch.zeros(vocab, device="cuda", dtype=torch.int32)
        counts[[10, 11, 12]] = 3
        return BatchSamplingArgs(
            row_temperatures, top_k=top_k, top_p=top_p, penalties=[(0, counts, 0.5, 1.0)], **extra
        )

    def draw(args: BatchSamplingArgs) -> torch.Tensor:
        torch.manual_seed(1234)
        return sampler.sample(sampler.apply_penalties(logits, args), args)

    plain = draw(fresh_args())
    with_logprobs = draw(
        fresh_args(
            logprob_rows=torch.tensor([True, False, True, False], device="cuda"),
            max_top_logprobs=20,
        )
    )
    assert torch.equal(plain, with_logprobs)

    greedy_args = fresh_args(row_temperatures=None)
    greedy = sampler.sample(sampler.apply_penalties(logits, greedy_args), greedy_args)
    greedy_lp_args = fresh_args(row_temperatures=None)
    greedy_lp_args.logprob_rows = torch.tensor([True, True, True, True], device="cuda")
    greedy_lp_args.max_top_logprobs = 20
    greedy_lp = sampler.sample(sampler.apply_penalties(logits, greedy_lp_args), greedy_lp_args)
    assert torch.equal(greedy, greedy_lp)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_cuda_computed_logprobs_match_the_penalized_distribution() -> None:
    vocab = 4096
    torch.manual_seed(7)
    sampler = Sampler(torch.device("cuda"), vocab_size=vocab)
    logits = torch.randn(3, vocab, device="cuda", dtype=torch.bfloat16)
    counts = torch.zeros(vocab, device="cuda", dtype=torch.int32)
    counts[[1, 2, 3, 4]] = 2
    args = BatchSamplingArgs(
        temperatures=None,
        penalties=[(1, counts, 1.0, 1.0)],
        logprob_rows=torch.tensor([True, True, False], device="cuda"),
        max_top_logprobs=5,
    )

    penalized = sampler.apply_penalties(logits, args)
    tokens = torch.tensor([5, 6, 7], device="cuda", dtype=torch.int32)
    chosen, top_ids, top_logprobs = sampler.compute_logprobs(penalized, tokens, args)
    torch.cuda.synchronize()

    reference = torch.log_softmax(penalized.float(), dim=-1)
    for row in (0, 1):
        assert torch.isclose(
            chosen[row].cpu(), reference[row, tokens[row]].cpu(), atol=1e-6
        )
        expected = reference[row].topk(5)
        assert torch.equal(top_ids[row].cpu(), expected.indices.cpu().to(torch.int32))
        assert torch.allclose(top_logprobs[row].cpu(), expected.values.cpu(), atol=1e-6)
    assert torch.isnan(chosen[2])
    assert torch.equal(top_ids[2].cpu(), torch.full((5,), -1, dtype=torch.int32))


def test_compute_logprobs_uses_the_distribution_before_sampling_updates_counts() -> None:
    # sample() increments the penalty counters for the token it just drew, so the
    # logprob pass must read the logits the sampler saw, never the counters again -
    # otherwise a repeated token would report a logprob from a stricter distribution
    # than the one it came from.
    sampler = Sampler(torch.device("cpu"), vocab_size=4)
    logits = torch.tensor([[1.0, 2.0, 0.5, -1.0]], dtype=torch.float32)
    counts = torch.zeros(4, dtype=torch.int32)
    args = BatchSamplingArgs(
        temperatures=None,
        penalties=[(0, counts, 0.0, 1.0)],
        logprob_rows=torch.tensor([True], dtype=torch.bool),
        max_top_logprobs=2,
    )

    sampling_logits = sampler.apply_penalties(logits, args)
    tokens = sampler.sample(sampling_logits, args)
    assert int(counts[int(tokens[0])]) == 1  # sample() bumped the drawn token's counter

    chosen_logprobs, _, _ = sampler.compute_logprobs(sampling_logits, tokens, args)
    expected = torch.log_softmax(sampling_logits, dim=-1)
    assert torch.isclose(chosen_logprobs[0], expected[0, tokens[0]])
