"""Pure unit coverage for engine-generation and admitted-token accounting.

These tests construct scheduler/frontend shells only; they never initialize an Engine,
load weights, or touch a GPU.
"""

from __future__ import annotations

import asyncio
import contextlib
import uuid
from types import SimpleNamespace

import pytest
import torch

from freetoken.core import SamplingParams
from freetoken.message import (
    AbortBackendMsg,
    BatchTokenizerMsg,
    DetokenizeMsg,
    ErrorReplyMsg,
    PromptAdmittedMsg,
    TokenizeMsg,
    UserMsg,
    UserReply,
)
from freetoken.scheduler.interleave import DecodeInterleavePolicy
from freetoken.scheduler.io import SchedulerIOMixin
from freetoken.scheduler.scheduler import Scheduler
from freetoken.server.api_server import FrontendManager
from freetoken.tokenizer.server import (
    _error_reply,
    _prompt_admitted_reply,
    _send_generation_replies,
    _tokenize_requests,
)


class _Logger:
    def __init__(self) -> None:
        self.warnings: list[str] = []

    def warning(self, message: str) -> None:
        self.warnings.append(message)


def _tokenize_msg(uid: int) -> TokenizeMsg:
    return TokenizeMsg(uid=uid, text="hello", sampling_params=SamplingParams(max_tokens=2))


def test_successful_tokenization_does_not_account_prompt_before_admission():
    class Tokenizer:
        def tokenize(self, messages):
            (msg,) = messages
            return [UserMsg(uid=msg.uid, input_ids=torch.tensor([10, 11, 12], dtype=torch.int32), sampling_params=msg.sampling_params)]

    backend, errors = _tokenize_requests(Tokenizer(), [_tokenize_msg(1)], _Logger())
    assert [msg.uid for msg in backend] == [1]
    assert backend[0].input_ids.tolist() == [10, 11, 12]
    assert errors == []  # in particular, no early prompt_tokens_delta UserReply


def test_tokenization_failure_and_empty_prompt_are_terminal_without_usage():
    class Tokenizer:
        def tokenize(self, messages):
            (msg,) = messages
            if msg.uid == 2:
                raise ValueError("bad template")
            return [UserMsg(uid=msg.uid, input_ids=torch.empty(0, dtype=torch.int32), sampling_params=msg.sampling_params)]

    logger = _Logger()
    backend, errors = _tokenize_requests(
        Tokenizer(), [_tokenize_msg(2), _tokenize_msg(3)], logger
    )
    assert backend == []
    assert [reply.uid for reply in errors] == [2, 3]
    assert all(reply.finished and reply.prompt_tokens_delta == 0 for reply in errors)
    assert "could not encode request" in errors[0].error
    assert errors[1].error == "prompt must contain at least one token"


def test_prompt_admitted_signal_uses_existing_frontend_usage_channel():
    reply = _prompt_admitted_reply(PromptAdmittedMsg(uid=7, prompt_tokens=321, cached_tokens=100))
    assert reply.uid == 7
    assert reply.prompt_tokens_delta == 321
    assert reply.cached_tokens == 100
    assert reply.completion_tokens_delta == 0
    assert reply.finished is False


def test_schedule_reports_admission_only_after_prepare_succeeds():
    batch = SimpleNamespace(is_prefill=True, prompt_admissions=[(1, 12, 4), (2, 34, 0)])
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.prefill_budget = 99
    scheduler.prefill_manager = SimpleNamespace(schedule_next_batch=lambda budget: batch)
    scheduler.decode_manager = SimpleNamespace(schedule_next_batch=lambda: None, runnable=False)
    scheduler.decode_manager = SimpleNamespace(schedule_next_batch=lambda: None)
    scheduler._prefill_start = {}
    events = []

    def prepare(value):
        events.append(("prepared", value))
        return "forward-input"

    def send(messages):
        events.append(("sent", messages))

    scheduler._prepare_batch = prepare
    scheduler.send_result = send

    assert Scheduler._schedule_next_batch(scheduler) == "forward-input"
    assert events[0] == ("prepared", batch)
    sent = events[1][1]
    assert [(m.uid, m.prompt_tokens, m.cached_tokens) for m in sent] == [(1, 12, 4), (2, 34, 0)]
    # Admission also opens each request's prefill span, closed at its first sampled token.
    assert sorted(scheduler._prefill_start) == [1, 2]


def test_prepare_failure_emits_no_prompt_admission():
    batch = SimpleNamespace(is_prefill=True, prompt_admissions=[(1, 12, 0)])
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.prefill_budget = 99
    scheduler.prefill_manager = SimpleNamespace(schedule_next_batch=lambda budget: batch)
    scheduler.decode_manager = SimpleNamespace(schedule_next_batch=lambda: None, runnable=False)
    sent = []

    def fail_prepare(_batch):
        raise RuntimeError("allocation failed")

    scheduler._prepare_batch = fail_prepare
    scheduler.send_result = sent.extend
    with pytest.raises(RuntimeError, match="allocation failed"):
        Scheduler._schedule_next_batch(scheduler)
    assert sent == []


def test_freeing_a_request_that_never_sampled_closes_its_prefill_span():
    """The first sampled token is what normally pops the span. A request aborted mid-prefill
    never reaches that point, so the free path has to pop it -- otherwise a long-lived server
    leaks one entry per aborted request."""
    scheduler = Scheduler.__new__(Scheduler)
    scheduler._prefill_start = {7: 100.0, 9: 200.0}
    scheduler.cache_manager = SimpleNamespace(cache_req=lambda req, finished: None)
    scheduler.table_manager = SimpleNamespace(free=lambda table_idx: None)

    Scheduler._free_req_resources(scheduler, SimpleNamespace(uid=7, table_idx=3))

    assert scheduler._prefill_start == {9: 200.0}  # only the freed request's span is dropped


def test_scheduler_rejection_emits_error_but_no_admission():
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.engine = SimpleNamespace(max_seq_len=4, num_pages=1)
    scheduler.config = SimpleNamespace(page_size=64)  # PR #118: admission clamps to the pool
    added = []
    scheduler.prefill_manager = SimpleNamespace(add_one_req=added.append)
    sent = []
    scheduler.send_result = sent.extend

    Scheduler._process_one_msg(
        scheduler,
        UserMsg(
            uid=8,
            input_ids=torch.arange(4, dtype=torch.int32),
            sampling_params=SamplingParams(max_tokens=1),
        ),
    )
    assert added == []
    assert len(sent) == 1 and isinstance(sent[0], ErrorReplyMsg)
    assert not any(isinstance(msg, PromptAdmittedMsg) for msg in sent)


def test_scheduler_clamps_admission_to_the_actual_kv_pool():
    """PR #118: a prompt that fits the advertised context but not the allocated pool must be
    rejected with the pool-derived limit instead of being queued forever."""
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.engine = SimpleNamespace(max_seq_len=100_000, num_pages=1)
    scheduler.config = SimpleNamespace(page_size=64)  # pool holds 64 tokens, not 100_000
    added = []
    scheduler.prefill_manager = SimpleNamespace(add_one_req=added.append)
    sent = []
    scheduler.send_result = sent.extend

    Scheduler._process_one_msg(
        scheduler,
        UserMsg(
            uid=9,
            input_ids=torch.arange(100, dtype=torch.int32),
            sampling_params=SamplingParams(max_tokens=1),
        ),
    )
    assert added == []
    assert len(sent) == 1 and isinstance(sent[0], ErrorReplyMsg)
    assert "64" in sent[0].error, sent[0].error


def test_scheduler_always_emits_terminal_abort_ack_for_unknown_uid():
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.prefill_manager = SimpleNamespace(abort_req=lambda uid: None)
    scheduler.decode_manager = SimpleNamespace(abort_req=lambda uid: None)
    scheduler._pending_abort_acks = set()
    sent = []
    scheduler.send_result = sent.extend

    Scheduler._process_one_msg(scheduler, AbortBackendMsg(uid=77))
    assert sent == []  # terminal barrier waits for prior sampled data to drain
    Scheduler._flush_abort_acks(scheduler)
    assert len(sent) == 1
    assert isinstance(sent[0], ErrorReplyMsg)
    assert sent[0].uid == 77 and sent[0].error == "request aborted"


def test_abort_before_cross_worker_user_message_cannot_resurrect_request():
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.engine = SimpleNamespace(max_seq_len=64)
    added = []
    scheduler.prefill_manager = SimpleNamespace(
        add_one_req=added.append,
        abort_req=lambda uid: None,
    )
    scheduler.decode_manager = SimpleNamespace(abort_req=lambda uid: None)
    scheduler._pending_abort_acks = set()
    scheduler._abort_tombstones = {}
    sent = []
    scheduler.send_result = sent.extend

    Scheduler._process_one_msg(scheduler, AbortBackendMsg(uid=78))
    Scheduler._flush_abort_acks(scheduler)
    Scheduler._process_one_msg(
        scheduler,
        UserMsg(
            uid=78,
            input_ids=torch.arange(8, dtype=torch.int32),
            sampling_params=SamplingParams(max_tokens=4),
        ),
    )

    assert added == []
    assert len(sent) == 1 and isinstance(sent[0], ErrorReplyMsg)
    assert not any(isinstance(msg, PromptAdmittedMsg) for msg in sent)
    assert 78 not in scheduler._abort_tombstones


def test_normal_loop_sends_prior_sample_before_abort_terminal():
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.prefill_manager = SimpleNamespace(runnable=False, abort_req=lambda uid: None)
    scheduler.decode_manager = SimpleNamespace(runnable=False, abort_req=lambda uid: None)
    scheduler._pending_abort_acks = set()
    scheduler._pending_rebuild = None
    scheduler.receive_msg = lambda blocking: [AbortBackendMsg(uid=5)]
    scheduler._schedule_next_batch = lambda: None
    sent = []
    scheduler.send_result = lambda messages: sent.append(messages)
    late_sample = DetokenizeMsg(uid=5, next_token=99, finished=False)
    scheduler._process_last_data = lambda _data: scheduler.send_result([late_sample])

    Scheduler.normal_loop(scheduler)

    assert len(sent) == 2
    assert sent[0] == [late_sample]
    assert isinstance(sent[1][0], ErrorReplyMsg)
    assert sent[1][0].uid == 5 and sent[1][0].error == "request aborted"


def test_tokenizer_drain_sends_sampled_reply_before_abort_terminal():
    class Sink:
        def __init__(self):
            self.items = []

        def put(self, item):
            self.items.append(item)

    sink = Sink()
    sampled = UserReply(uid=5, incremental_output="x", finished=False, completion_tokens_delta=1)
    terminal = _error_reply(ErrorReplyMsg(uid=5, error="request aborted"))
    _send_generation_replies(sink, [], [sampled], [terminal])

    assert sink.items == [sampled, terminal]


def test_tp_rank0_batches_admissions_and_nonprimary_is_noop():
    class Sink:
        def __init__(self):
            self.items = []

        def put(self, item):
            self.items.append(item)

    io = SchedulerIOMixin.__new__(SchedulerIOMixin)
    io._send_into_tokenizer = Sink()
    admissions = [
        PromptAdmittedMsg(uid=1, prompt_tokens=10),
        PromptAdmittedMsg(uid=2, prompt_tokens=20),
    ]
    SchedulerIOMixin._reply_tokenizer_rank0(io, admissions)
    assert len(io._send_into_tokenizer.items) == 1
    assert isinstance(io._send_into_tokenizer.items[0], BatchTokenizerMsg)
    assert io._send_into_tokenizer.items[0].data == admissions
    SchedulerIOMixin._reply_tokenizer_rank1(io, admissions)  # explicit no-op
    assert len(io._send_into_tokenizer.items) == 1


def test_offline_handler_ignores_online_prompt_accounting_signal():
    from freetoken.llm.llm import LLM

    offline = SimpleNamespace(status_map={}, eos_token_ids=set())
    LLM.offline_send_result(offline, [PromptAdmittedMsg(uid=1, prompt_tokens=10)])


def test_frontend_manager_generates_unique_uuid_instance_ids():
    config = SimpleNamespace()
    first = FrontendManager(config=config, send_tokenizer=None, recv_tokenizer=None)
    second = FrontendManager(config=config, send_tokenizer=None, recv_tokenizer=None)
    assert first.instance_id != second.instance_id
    assert str(uuid.UUID(first.instance_id)) == first.instance_id
    assert str(uuid.UUID(second.instance_id)) == second.instance_id


def test_listener_accounts_late_reply_after_ack_queue_was_removed():
    class OneThenBlock:
        def __init__(self, first):
            self.first = first

        async def get(self):
            if self.first is not None:
                result, self.first = self.first, None
                return result
            await asyncio.Future()

    async def run():
        late = UserReply(
            uid=999,
            incremental_output="",
            finished=True,
            prompt_tokens_delta=5,
            completion_tokens_delta=2,
        )
        manager = FrontendManager(
            config=SimpleNamespace(),
            send_tokenizer=None,
            recv_tokenizer=OneThenBlock(late),
        )
        task = asyncio.create_task(manager.listen())
        await asyncio.sleep(0)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        return manager

    manager = asyncio.run(run())
    assert manager.stats.prompt_tokens_total == 5
    assert manager.stats.completion_tokens_total == 2
    assert manager.ack_map == {}


def test_a_long_prefill_does_not_starve_an_active_decode():
    """Fairness: one decode step per prefill chunk, instead of draining prefill first.

    Regression for the measured tail: a 30K prompt arriving beside an active stream cost that
    stream an 18.07 s inter-token gap because every iteration served the next prefill chunk.
    """
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.prefill_budget = 99
    served: list[str] = []
    prefill_batch = SimpleNamespace(is_prefill=True, prompt_admissions=[])
    decode_batch = SimpleNamespace(is_prefill=False, prompt_admissions=[])
    scheduler.prefill_manager = SimpleNamespace(
        schedule_next_batch=lambda budget: served.append("prefill") or prefill_batch, runnable=True
    )
    scheduler.decode_manager = SimpleNamespace(
        schedule_next_batch=lambda: served.append("decode") or decode_batch, runnable=True
    )
    scheduler._interleave = DecodeInterleavePolicy(1)
    scheduler._prepare_batch = lambda batch: batch
    scheduler._report_prompt_admissions = lambda batch: None

    Scheduler._schedule_next_batch(scheduler)  # nothing served yet -> prefill first
    Scheduler._schedule_next_batch(scheduler)  # a chunk is pending -> give decode a turn
    Scheduler._schedule_next_batch(scheduler)  # then back to prefill
    assert served == ["prefill", "decode", "prefill"], served


def test_without_a_runnable_decode_the_prefill_order_is_unchanged():
    """The no-decode path must stay byte-for-byte the historical one (prefill drains first)."""
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.prefill_budget = 99
    served: list[str] = []
    prefill_batch = SimpleNamespace(is_prefill=True, prompt_admissions=[])
    scheduler.prefill_manager = SimpleNamespace(
        schedule_next_batch=lambda budget: served.append("prefill") or prefill_batch, runnable=True
    )
    # A decode that is not runnable returns None; the policy then falls through to prefill
    # instead of spending the slot idle (upstream calls schedule_next_batch, not runnable).
    scheduler.decode_manager = SimpleNamespace(schedule_next_batch=lambda: None, runnable=False)
    scheduler._interleave = DecodeInterleavePolicy(1)
    scheduler._prepare_batch = lambda batch: batch
    scheduler._report_prompt_admissions = lambda batch: None

    for _ in range(3):
        Scheduler._schedule_next_batch(scheduler)
    assert served == ["prefill", "prefill", "prefill"], served
