from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
from freetoken.message import TokenizeMsg, UserReply
from freetoken.server.anthropic_api import register_anthropic_routes
from freetoken.server.openai_api import (
    ChatCompletionRequest,
    CompletionRequest,
    chat_request_to_genspec,
    handle_chat_completion,
    handle_completion,
    register_openai_routes,
    stream_chat_completion_chunks,
    stream_completion_chunks,
)


def run(coro):
    return asyncio.run(coro)


class FakeState:
    def __init__(
        self,
        replies: list[UserReply],
        tool_call_parser: str = "llama3",
        reasoning_parser: str | None = None,
    ) -> None:
        self.config = SimpleNamespace(
            mm=SimpleNamespace(text_model_only=False, disabled_encoders=frozenset()),
            model_path="/models/unit-model",
            served_model_name="unit-model",
            tool_call_parser=tool_call_parser,
            reasoning_parser=reasoning_parser,
        )
        self.replies = replies
        self.sent: TokenizeMsg | None = None

    def new_user(self) -> int:
        return 42

    async def send_one(self, msg):
        self.sent = msg

    async def wait_for_ack(self, uid: int):
        assert uid == 42
        for reply in self.replies:
            yield reply


def tool_schema():
    return [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Return weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]


def opencode_tool_schema():
    return [
        {
            "type": "function",
            "function": {
                "name": "read",
                "parameters": {
                    "type": "object",
                    "properties": {"filePath": {"type": "string"}},
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "glob",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string"},
                        "path": {"type": "string"},
                    },
                },
            },
        },
    ]


def chat_request(**kwargs) -> ChatCompletionRequest:
    payload = {
        "model": "client-model",
        "messages": [{"role": "user", "content": "weather in Paris?"}],
        "tools": tool_schema(),
        "max_tokens": 8,
    }
    payload.update(kwargs)
    return ChatCompletionRequest(**payload)


def parse_sse(chunks: list[bytes]) -> list[dict | str]:
    events: list[dict | str] = []
    for chunk in chunks:
        for line in chunk.decode().splitlines():
            if not line.startswith("data: "):
                continue
            data = line.removeprefix("data: ")
            events.append(data if data == "[DONE]" else json.loads(data))
    return events


def test_chat_request_accepts_tool_messages_and_assistant_tool_calls():
    req = ChatCompletionRequest(
        model="client-model",
        messages=[
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
        ],
        tools=tool_schema(),
        max_completion_tokens=11,
        stream_options={"include_usage": True},
    )

    assert req.max_tokens == 11
    assert req.stream_options is not None
    assert req.stream_options.include_usage is True
    assert req.messages[0].tool_calls[0].function.arguments == '{"city":"Paris"}'


def test_chat_request_reasoning_replay_field_aliases():
    # Any replay field name in -> both template-read field names out.
    for field in ("reasoning_content", "reasoning", "thinking"):
        req = ChatCompletionRequest(
            model="client-model",
            messages=[
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "ok", field: "prior thought"},
                {"role": "user", "content": "next"},
            ],
        )
        asst = chat_request_to_genspec(req, {}).messages[1]
        assert asst["reasoning_content"] == "prior thought", field
        assert asst["thinking"] == "prior thought", field


def test_chat_reasoning_effort_enables_thinking():
    spec = chat_request_to_genspec(chat_request(reasoning_effort="high"), {})
    assert spec.chat_template_kwargs == {
        "enable_thinking": True, "thinking_mode": "enabled", "reasoning_effort": "high"
    }

    # an explicit thinking-related chat_template_kwargs key wins over the mapping
    spec = chat_request_to_genspec(
        chat_request(reasoning_effort="none", chat_template_kwargs={"enable_thinking": True}), {}
    )
    assert spec.chat_template_kwargs == {"enable_thinking": True}

    # unrelated extra kwargs ride along without discarding the effort mapping
    spec = chat_request_to_genspec(
        chat_request(reasoning_effort="none", chat_template_kwargs={"custom_var": 1}), {}
    )
    assert spec.chat_template_kwargs == {
        "enable_thinking": False, "thinking_mode": "disabled", "custom_var": 1
    }

    # absent effort -> kwargs pass through untouched
    assert chat_request_to_genspec(chat_request(), {}).chat_template_kwargs == {}


def test_chat_reasoning_effort_none_disables_thinking():
    # vLLM-compatible semantics: an explicit effort "none" DISABLES thinking.
    spec = chat_request_to_genspec(chat_request(reasoning_effort="none"), {})
    assert spec.chat_template_kwargs == {"enable_thinking": False, "thinking_mode": "disabled"}


def test_chat_reasoning_effort_broadcasts_every_toggle_spelling():
    """The toggle is broadcast in every spelling templates read (enable_thinking
    bool + M3's thinking_mode); each template picks the knob it knows and Jinja
    ignores the rest, so no per-family routing exists."""
    on = chat_request(reasoning_effort="high")
    spec = chat_request_to_genspec(on, {})
    assert spec.chat_template_kwargs == {
        "enable_thinking": True, "thinking_mode": "enabled", "reasoning_effort": "high"
    }

    off = chat_request(reasoning_effort="none")
    spec = chat_request_to_genspec(off, {})
    assert spec.chat_template_kwargs == {"enable_thinking": False, "thinking_mode": "disabled"}



def test_glm_reasoning_parser_honors_disabled_thinking_with_tools():
    # The parse side must match the encode side: thinking off + tools present
    # must not start the parser inside a think block.
    from freetoken.server.generation import _make_reasoning_parser

    state = FakeState([], reasoning_parser="glm")
    off = chat_request_to_genspec(chat_request(reasoning_effort="none"), {})
    parser = _make_reasoning_parser(off, state)
    assert parser is not None and parser.detector.force_reasoning is False

    on = chat_request_to_genspec(chat_request(), {})
    parser = _make_reasoning_parser(on, state)
    assert parser is not None and parser.detector.force_reasoning is True


def test_non_stream_chat_completion_returns_openai_tool_calls_and_sends_tools():
    output = '[TOOL_CALLS] [{"name":"get_weather","arguments":{"city":"Paris"}}]'
    state = FakeState(
        [
            UserReply(uid=42, incremental_output=output, finished=True, prompt_tokens_delta=5, completion_tokens_delta=7),
        ],
        tool_call_parser="mistral",
    )

    response = run(handle_chat_completion(chat_request(), request=None, state=state, model_sampling={}))

    assert state.sent is not None
    assert state.sent.text == [{"role": "user", "content": "weather in Paris?"}]
    assert state.sent.tools == tool_schema()
    choice = response["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] == ""
    tool_call = choice["message"]["tool_calls"][0]
    assert tool_call["type"] == "function"
    assert tool_call["function"]["name"] == "get_weather"
    assert json.loads(tool_call["function"]["arguments"]) == {"city": "Paris"}
    assert response["usage"] == {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12}


def test_non_stream_chat_completion_length_truncation_overrides_tool_calls():
    output = '[TOOL_CALLS] [{"name":"get_weather","arguments":{"city":"Paris"}}]'
    state = FakeState(
        [UserReply(uid=42, incremental_output=output, finished=True, finish_reason="length")],
        tool_call_parser="mistral",
    )
    response = run(handle_chat_completion(chat_request(), request=None, state=state, model_sampling={}))
    assert response["choices"][0]["finish_reason"] == "length"


def test_non_stream_chat_completion_parses_configured_family_tool_shape():
    output = (
        "<|channel|>analysis<|message|>Need files.<|end|><|start|>assistant"
        "<|channel|>commentary to=functions.glob <|constrain|>json<|message|>"
        '{"pattern":"**/*.py","path":"/tmp/ws"}'
    )
    state = FakeState([UserReply(uid=42, incremental_output=output, finished=True)], tool_call_parser="gpt_oss")
    req = ChatCompletionRequest(
        model="client-model",
        messages=[{"role": "user", "content": "inspect"}],
        tools=opencode_tool_schema(),
        max_tokens=8,
    )

    response = run(handle_chat_completion(req, request=None, state=state, model_sampling={}))

    choice = response["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    tool_call = choice["message"]["tool_calls"][0]
    assert tool_call["function"]["name"] == "glob"
    assert json.loads(tool_call["function"]["arguments"]) == {
        "pattern": "**/*.py",
        "path": "/tmp/ws",
    }


def test_stream_chat_completion_emits_chat_chunks_tool_delta_and_done():
    output = '[TOOL_CALLS] [{"name":"get_weather","arguments":{"city":"Paris"}}]'
    state = FakeState(
        [
            UserReply(uid=42, incremental_output=output, finished=True, prompt_tokens_delta=5, completion_tokens_delta=7),
        ],
        tool_call_parser="mistral",
    )

    chunks = run(_collect(stream_chat_completion_chunks(42, chat_request(stream=True), state)))
    events = parse_sse(chunks)

    assert events[-1] == "[DONE]"
    assert events[0]["object"] == "chat.completion.chunk"
    assert events[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
    tool_deltas = [
        tool_call
        for event in events
        if isinstance(event, dict)
        for choice in event.get("choices", [])
        for tool_call in choice.get("delta", {}).get("tool_calls", [])
    ]
    assert tool_deltas[0]["function"]["name"] == "get_weather"
    assert json.loads("".join(delta["function"].get("arguments", "") for delta in tool_deltas)) == {
        "city": "Paris"
    }
    finish_reasons = [
        choice["finish_reason"]
        for event in events
        if isinstance(event, dict)
        for choice in event.get("choices", [])
        if choice.get("finish_reason")
    ]
    assert finish_reasons == ["tool_calls"]


def test_tool_choice_none_keeps_tool_tags_as_content():
    output = '[TOOL_CALLS] [{"name":"get_weather","arguments":{"city":"Paris"}}]'
    state = FakeState([UserReply(uid=42, incremental_output=output, finished=True)])

    response = run(
        handle_chat_completion(
            chat_request(tool_choice="none"),
            request=None,
            state=state,
            model_sampling={},
        )
    )

    assert state.sent is not None
    assert state.sent.tools is None
    choice = response["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["message"] == {"role": "assistant", "content": output}


def test_completion_rejects_token_id_prompts():
    state = FakeState([])
    response = run(
        handle_completion(
            CompletionRequest(model="client-model", prompt=[1, 2, 3]),
            request=None,
            state=state,
            model_sampling={},
        )
    )

    assert response.status_code == 400
    body = json.loads(response.body)
    assert body["error"]["type"] == "invalid_request_error"
    assert "token-id prompt" in body["error"]["message"]


def test_completion_accepts_text_prompt():
    state = FakeState(
        [UserReply(uid=42, incremental_output="hello", finished=True, prompt_tokens_delta=2, completion_tokens_delta=1)]
    )

    response = run(
        handle_completion(
            CompletionRequest(model="client-model", prompt="say hi"),
            request=None,
            state=state,
            model_sampling={},
        )
    )

    assert state.sent is not None
    assert state.sent.text == "say hi"
    assert response["object"] == "text_completion"
    assert response["choices"][0]["text"] == "hello"
    assert response["usage"] == {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}


def test_completion_forwards_length_finish_reason():
    state = FakeState(
        [UserReply(uid=42, incremental_output="hello", finished=True, finish_reason="length")]
    )
    response = run(
        handle_completion(
            CompletionRequest(model="client-model", prompt="say hi"),
            request=None,
            state=state,
            model_sampling={},
        )
    )
    assert response["choices"][0]["finish_reason"] == "length"


def test_omitted_max_tokens_honors_server_default():
    from freetoken.server.generation import DEFAULT_MAX_OUTPUT_TOKENS

    chat_state = FakeState([UserReply(uid=42, incremental_output="hi", finished=True)])
    chat_state.config.max_output_tokens = 4096
    run(handle_chat_completion(
        ChatCompletionRequest(model="m", messages=[{"role": "user", "content": "hi"}]),
        request=None, state=chat_state, model_sampling={},
    ))
    assert chat_state.sent.sampling_params.max_tokens == 4096

    cmpl_state = FakeState([UserReply(uid=42, incremental_output="hi", finished=True)])
    cmpl_state.config.max_output_tokens = 4096
    run(handle_completion(
        CompletionRequest(model="m", prompt="hi"),
        request=None, state=cmpl_state, model_sampling={},
    ))
    assert cmpl_state.sent.sampling_params.max_tokens == 4096

    # explicit value wins
    exp_state = FakeState([UserReply(uid=42, incremental_output="hi", finished=True)])
    exp_state.config.max_output_tokens = 4096
    run(handle_completion(
        CompletionRequest(model="m", prompt="hi", max_tokens=50),
        request=None, state=exp_state, model_sampling={},
    ))
    assert exp_state.sent.sampling_params.max_tokens == 50

    fallback_state = FakeState([UserReply(uid=42, incremental_output="hi", finished=True)])
    run(handle_chat_completion(
        ChatCompletionRequest(model="m", messages=[{"role": "user", "content": "hi"}]),
        request=None, state=fallback_state, model_sampling={},
    ))
    assert fallback_state.sent.sampling_params.max_tokens == DEFAULT_MAX_OUTPUT_TOKENS


def test_models_route_returns_served_model_name():
    state = FakeState([])
    app = FastAPI()
    register_openai_routes(app, lambda: state, lambda: {})

    response = TestClient(app).get("/v1/models")

    assert response.status_code == 200
    card = response.json()["data"][0]
    assert card["id"] == "unit-model"
    # No max_seq_len on this config: null rather than a 500.
    assert card["max_model_len"] is None and card["context_length"] is None


def test_models_route_publishes_the_model_context_length():
    """`ft launch` reads this to size each agent's context window."""
    state = FakeState([])
    state.config.max_seq_len = 262144
    app = FastAPI()
    register_openai_routes(app, lambda: state, lambda: {})

    card = TestClient(app).get("/v1/models").json()["data"][0]

    assert card["max_model_len"] == 262144
    assert card["context_length"] == 262144
    assert card["model_max_len"] == 262144


def test_models_route_reports_the_enforced_ceiling_when_the_kv_pool_is_smaller():
    """A KV pool below the model's max_position must not be advertised as usable.

    The scheduler admits a request only while prompt_tokens < min(model max_position, KV pool
    tokens), so a card promising the model's own ceiling makes a correctly-configured client
    (one that sizes its window from this route, as `ft launch` does) send prompts the server
    then rejects with context_length_exceeded. The enforced number is what the engine publishes
    on its readiness meta; `model_max_len` keeps the raw checkpoint ceiling visible.
    """
    state = FakeState([])
    state.config.max_seq_len = 262144          # checkpoint ceiling
    state.max_seq_len = 245760                 # what the engine actually admits
    app = FastAPI()
    register_openai_routes(app, lambda: state, lambda: {})

    card = TestClient(app).get("/v1/models").json()["data"][0]

    assert card["max_model_len"] == 245760 and card["context_length"] == 245760
    assert card["model_max_len"] == 262144


async def _collect(generator):
    return [chunk async for chunk in generator]


# --------------------------------------------------------------- dsv4 reasoning
from freetoken.server.reasoning_parser import DSML_TOKEN  # noqa: E402

_TC_OPEN = f"<{DSML_TOKEN}tool_calls>"
_DSV4_TOOL_BLOCK = (
    f"{_TC_OPEN}\n"
    f'<{DSML_TOKEN}invoke name="get_weather">\n'
    f'<{DSML_TOKEN}parameter name="city" string="true">Paris</{DSML_TOKEN}parameter>\n'
    f"</{DSML_TOKEN}invoke>\n"
    f"</{DSML_TOKEN}tool_calls>"
)


def _dsv4_state(replies):
    return FakeState(replies, tool_call_parser="deepseekv32", reasoning_parser="deepseekv32")


def test_dsv4_non_stream_splits_reasoning_and_tool_call():
    # tools present -> thinking mode -> output starts inside the reasoning block.
    output = f"I should look up the weather.</think>Let me check.\n\n{_DSV4_TOOL_BLOCK}"
    state = _dsv4_state([UserReply(uid=42, incremental_output=output, finished=True)])

    response = run(handle_chat_completion(chat_request(), request=None, state=state, model_sampling={}))

    choice = response["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["reasoning_content"] == "I should look up the weather."
    assert choice["message"]["content"] == "Let me check."
    tool_call = choice["message"]["tool_calls"][0]
    assert tool_call["function"]["name"] == "get_weather"
    assert json.loads(tool_call["function"]["arguments"]) == {"city": "Paris"}


def test_dsv4_non_stream_missing_end_token_before_tool_block():
    # dsv4 sometimes skips </think> and jumps straight to the tool block.
    output = f"Looking it up now.\n\n{_DSV4_TOOL_BLOCK}"
    state = _dsv4_state([UserReply(uid=42, incremental_output=output, finished=True)])

    response = run(handle_chat_completion(chat_request(), request=None, state=state, model_sampling={}))

    choice = response["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["reasoning_content"] == "Looking it up now."
    assert choice["message"]["content"] == ""
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "get_weather"


def test_dsv4_non_stream_reasoning_without_tools():
    # No tools, but thinking explicitly requested.
    output = "Let me think about it.</think>The answer is 42."
    state = _dsv4_state([UserReply(uid=42, incremental_output=output, finished=True)])
    req = ChatCompletionRequest(
        model="client-model",
        messages=[{"role": "user", "content": "hi"}],
        chat_template_kwargs={"thinking": True},
        max_tokens=8,
    )

    response = run(handle_chat_completion(req, request=None, state=state, model_sampling={}))

    choice = response["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["message"]["reasoning_content"] == "Let me think about it."
    assert choice["message"]["content"] == "The answer is 42."
    assert "tool_calls" not in choice["message"]


def test_dsv4_non_stream_strips_leaked_special_tokens():
    bos = "<｜begin▁of▁sentence｜>"
    eos = "<｜end▁of▁sentence｜>"
    output = f"reasoning here</think>Hello{eos} world{bos}"
    state = _dsv4_state([UserReply(uid=42, incremental_output=output, finished=True)])
    req = ChatCompletionRequest(
        model="client-model",
        messages=[{"role": "user", "content": "hi"}],
        chat_template_kwargs={"thinking": True},
        max_tokens=8,
    )

    response = run(handle_chat_completion(req, request=None, state=state, model_sampling={}))

    assert response["choices"][0]["message"]["content"] == "Hello world"


def test_dsv4_stream_emits_reasoning_then_tool_calls():
    # Token-aligned deltas (the detokenizer emits one token's text per message,
    # so markers like </think> never arrive glued to preceding text).
    chunks = ["Thinking ", "hard.", "</think>", "One ", "sec.", "\n\n", _DSV4_TOOL_BLOCK]
    replies = [
        UserReply(uid=42, incremental_output=c, finished=(i == len(chunks) - 1))
        for i, c in enumerate(chunks)
    ]
    state = _dsv4_state(replies)

    events = parse_sse(run(_collect(stream_chat_completion_chunks(42, chat_request(stream=True), state))))

    reasoning = "".join(
        choice["delta"]["reasoning_content"]
        for event in events
        if isinstance(event, dict)
        for choice in event.get("choices", [])
        if "reasoning_content" in choice.get("delta", {})
    )
    assert reasoning == "Thinking hard."
    content = "".join(
        choice["delta"]["content"]
        for event in events
        if isinstance(event, dict)
        for choice in event.get("choices", [])
        if "content" in choice.get("delta", {}) and choice["delta"]["content"]
    )
    # Streaming releases the pre-tag separator whitespace as content (it is emitted
    # before the tool tag is seen); only trailing whitespace may differ from the
    # old buffer-then-strip behavior.
    assert content.rstrip() == "One sec."
    tool_names = [
        tc["function"]["name"]
        for event in events
        if isinstance(event, dict)
        for choice in event.get("choices", [])
        for tc in choice.get("delta", {}).get("tool_calls", [])
        if tc.get("function", {}).get("name")
    ]
    assert "get_weather" in tool_names
    finish_reasons = [
        choice["finish_reason"]
        for event in events
        if isinstance(event, dict)
        for choice in event.get("choices", [])
        if choice.get("finish_reason")
    ]
    assert finish_reasons == ["tool_calls"]


# --------------------------------------------------------------- gpt-oss harmony
def test_gptoss_non_stream_splits_reasoning_and_clean_content():
    output = (
        "<|channel|>analysis<|message|>The user says hi.<|end|>"
        "<|start|>assistant<|channel|>final<|message|>Hello there!"
    )
    state = FakeState(
        [UserReply(uid=42, incremental_output=output, finished=True)],
        tool_call_parser="gpt_oss",
        reasoning_parser="gpt_oss",
    )
    req = chat_request(tools=None)
    response = run(handle_chat_completion(req, request=None, state=state, model_sampling={}))
    message = response["choices"][0]["message"]
    assert message["reasoning_content"] == "The user says hi."
    assert message["content"] == "Hello there!"
    assert "<|channel|>" not in message["content"]


def test_gptoss_non_stream_still_extracts_tool_call():
    output = (
        "<|channel|>analysis<|message|>need weather<|end|>"
        "<|start|>assistant<|channel|>commentary to=functions.get_weather "
        '<|message|>{"city":"Paris"}<|call|>'
    )
    state = FakeState(
        [UserReply(uid=42, incremental_output=output, finished=True)],
        tool_call_parser="gpt_oss",
        reasoning_parser="gpt_oss",
    )
    response = run(handle_chat_completion(chat_request(), request=None, state=state, model_sampling={}))
    choice = response["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    tool_calls = choice["message"]["tool_calls"]
    assert tool_calls[0]["function"]["name"] == "get_weather"
    assert choice["message"]["reasoning_content"] == "need weather"


# ----------------------------------------------------------- cache report
def _cache_hit_replies() -> list[UserReply]:
    return [
        UserReply(uid=42, incremental_output="", finished=False, prompt_tokens_delta=5, cached_tokens=3),
        UserReply(uid=42, incremental_output="hi", finished=True, completion_tokens_delta=1),
    ]


def test_non_stream_chat_usage_reports_cached_tokens_only_with_flag():
    state = FakeState(_cache_hit_replies())
    state.config.enable_cache_report = True
    response = run(handle_chat_completion(chat_request(tools=None), request=None, state=state, model_sampling={}))
    # prompt_tokens stays inclusive of the cached prefix; the details carry the split.
    assert response["usage"]["prompt_tokens"] == 5
    assert response["usage"]["prompt_tokens_details"] == {"cached_tokens": 3}

    response = run(handle_chat_completion(chat_request(tools=None), request=None, state=FakeState(_cache_hit_replies()), model_sampling={}))
    assert "prompt_tokens_details" not in response["usage"]


def test_non_stream_chat_usage_omits_details_on_zero_hit():
    state = FakeState(
        [UserReply(uid=42, incremental_output="hi", finished=True, prompt_tokens_delta=5, completion_tokens_delta=1)]
    )
    state.config.enable_cache_report = True
    response = run(handle_chat_completion(chat_request(tools=None), request=None, state=state, model_sampling={}))
    assert "prompt_tokens_details" not in response["usage"]


def test_stream_chat_usage_chunk_carries_cached_tokens():
    state = FakeState(_cache_hit_replies())
    state.config.enable_cache_report = True
    req = chat_request(tools=None, stream_options={"include_usage": True})

    async def collect():
        return [chunk async for chunk in stream_chat_completion_chunks(42, req, state)]

    events = parse_sse(run(collect()))
    usage = next(e["usage"] for e in reversed(events) if isinstance(e, dict) and e.get("usage"))
    assert usage["prompt_tokens"] == 5
    assert usage["prompt_tokens_details"] == {"cached_tokens": 3}


# --------------------------------------------------------------- minimax think
def test_minimax_http_non_stream_forces_implicit_reasoning_without_request_knob():
    state = FakeState(
        [UserReply(uid=42, incremental_output="private thought</think>visible answer", finished=True)],
        reasoning_parser="minimax",
    )
    req = chat_request(tools=None)

    response = run(handle_chat_completion(req, request=None, state=state, model_sampling={}))

    message = response["choices"][0]["message"]
    assert message["reasoning_content"] == "private thought"
    assert message["content"] == "visible answer"


import pytest  # noqa: E402


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("temperature", -0.5),
        ("top_p", 0.0),
        ("top_k", 0),
    ],
)
def test_chat_completion_rejects_invalid_sampling(field, value):
    app = FastAPI()
    state = FakeState([])

    @app.post("/v1/chat/completions")
    async def chat_completion(req: ChatCompletionRequest):
        return await handle_chat_completion(req, request=None, state=state, model_sampling={})

    body = {
        "model": "unit-model",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
        field: value,
    }
    response = TestClient(app).post("/v1/chat/completions", json=body)

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["type"] == "invalid_request_error"
    assert field in error["message"]
    assert state.sent is None


@pytest.mark.parametrize(
    "sampling",
    [
        {"temperature": 0.0, "top_p": 1.0, "top_k": -1},
        {"temperature": 0.0, "top_p": 1.0, "top_k": 1},
    ],
)
def test_chat_completion_accepts_sampling_boundaries(sampling):
    app = FastAPI()
    state = FakeState([])

    @app.post("/v1/chat/completions")
    async def chat_completion(req: ChatCompletionRequest):
        return await handle_chat_completion(req, request=None, state=state, model_sampling={})

    response = TestClient(app).post(
        "/v1/chat/completions",
        json={
            "model": "unit-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            **sampling,
        },
    )

    assert response.status_code == 200
    assert state.sent is not None


def test_validation_error_with_non_finite_value_stays_json_safe():
    """`1e999` parses to inf; the 422 body must not carry it as a raw float.

    The default handler serializes the offending `input` verbatim, and
    json.dumps refuses inf, so a correctly rejected request still died as a
    500 with an empty body.
    """
    state = FakeState([])
    app = FastAPI()
    register_openai_routes(app, lambda: state, lambda: {})
    register_anthropic_routes(app, lambda: state, lambda: {})

    body = (
        '{"model":"unit-model","messages":[{"role":"user","content":"hi"}],'
        '"presence_penalty":1e999}'
    )
    response = TestClient(app).post(
        "/v1/chat/completions",
        content=body,
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail[0]["loc"] == ["body", "presence_penalty"]
    assert detail[0]["input"] == "inf"
    assert state.sent is None


def test_logprobs_validation_errors():
    chat_top_out_of_range = run(
        handle_chat_completion(
            chat_request(tools=None, logprobs=True, top_logprobs=25), None, FakeState([]), {}
        )
    )
    assert chat_top_out_of_range.status_code == 400

    chat_missing_flag = run(
        handle_chat_completion(chat_request(tools=None, top_logprobs=1), None, FakeState([]), {})
    )
    assert chat_missing_flag.status_code == 400

    completion_out_of_range = run(
        handle_completion(
            CompletionRequest(model="client-model", prompt="hello", logprobs=7),
            None,
            FakeState([]),
            {},
        )
    )
    assert completion_out_of_range.status_code == 400

    completion_echo = run(
        handle_completion(
            CompletionRequest(model="client-model", prompt="hello", echo=True, logprobs=1),
            None,
            FakeState([]),
            {},
        )
    )
    assert completion_echo.status_code == 400


def test_chat_logprobs_fail_closed_under_semantic_parsing():
    # A server-side reasoning parser hides reasoning tokens from message content, so
    # logprob entries cannot be aligned with it: reject up front, stream and
    # non-stream alike, before any engine work is submitted.
    with_parser = FakeState([], reasoning_parser="qwen3")
    for stream in (False, True):
        resp = run(
            handle_chat_completion(
                chat_request(tools=None, logprobs=True, stream=stream), None, with_parser, {}
            )
        )
        assert resp.status_code == 400
        assert json.loads(resp.body)["error"]["param"] == "logprobs"

    # Tool parsing consumes tokens into tool_calls -- same conflict.
    tools_resp = run(
        handle_chat_completion(chat_request(logprobs=True), None, FakeState([]), {})
    )
    assert tools_resp.status_code == 400
    assert json.loads(tools_resp.body)["error"]["param"] == "logprobs"

    # tool_choice="none" disables parsing, so logprobs stay available.
    state = FakeState([UserReply(uid=42, incremental_output="Hi", finished=True)])
    ok = run(
        handle_chat_completion(
            chat_request(logprobs=True, tool_choice="none"), None, state, {}
        )
    )
    assert state.sent is not None
    assert "logprobs" in ok["choices"][0]


def logprob_entry(token_id: int, token: str, logprob: float) -> dict:
    return {
        "token_id": token_id,
        "token": token,
        "bytes": list(token.encode("utf-8")),
        "logprob": logprob,
        "top": [
            {
                "token_id": token_id,
                "token": token,
                "bytes": list(token.encode("utf-8")),
                "logprob": logprob,
            },
            {"token_id": token_id + 1, "token": "x", "bytes": [120], "logprob": logprob - 1},
        ],
    }


def lp_reply(text: str, *, finished: bool = False, logprobs: dict | None = None) -> UserReply:
    return UserReply(
        uid=42,
        incremental_output=text,
        finished=finished,
        prompt_tokens_delta=3 if not text else 0,
        completion_tokens_delta=1 if text else 0,
        logprobs=logprobs,
    )


def test_chat_non_stream_logprobs():
    first = logprob_entry(1, "Hello", -0.1)
    second = logprob_entry(2, "!", -0.2)
    result = run(
        handle_chat_completion(
            chat_request(tools=None, logprobs=True, top_logprobs=2),
            None,
            FakeState([lp_reply("Hello", logprobs=first), lp_reply("!", finished=True, logprobs=second)]),
            {},
        )
    )

    assert result["choices"][0]["logprobs"]["content"] == [
        {
            "token": "Hello",
            "logprob": -0.1,
            "bytes": [72, 101, 108, 108, 111],
            "top_logprobs": [
                {"token": "Hello", "logprob": -0.1, "bytes": [72, 101, 108, 108, 111]},
                {"token": "x", "logprob": -1.1, "bytes": [120]},
            ],
        },
        {
            "token": "!",
            "logprob": -0.2,
            "bytes": [33],
            "top_logprobs": [
                {"token": "!", "logprob": -0.2, "bytes": [33]},
                {"token": "x", "logprob": -1.2, "bytes": [120]},
            ],
        },
    ]

    without_logprobs = run(
        handle_chat_completion(
            chat_request(tools=None), None, FakeState([lp_reply("Hello", finished=True)]), {}
        )
    )
    assert without_logprobs["choices"][0].get("logprobs") is None


def test_chat_stream_logprobs_follow_content_deltas():
    first = logprob_entry(1, "Hello", -0.1)
    state = FakeState([lp_reply("Hello", logprobs=first), lp_reply(" world", finished=True)])
    req = chat_request(tools=None, stream=True, logprobs=True, top_logprobs=2)

    events = parse_sse(run(_collect(stream_chat_completion_chunks(42, req, state))))
    content_choices = [
        event["choices"][0]
        for event in events
        if isinstance(event, dict)
        and event["choices"]
        and event["choices"][0]["delta"].get("content")
    ]

    assert content_choices[0]["logprobs"]["content"][0]["token"] == "Hello"
    assert "logprobs" not in content_choices[1]


def test_completion_logprobs_non_stream_and_stream():
    first = logprob_entry(1, "Hi", -0.1)
    second = logprob_entry(2, "!", -0.2)
    req = CompletionRequest(model="client-model", prompt="hello", logprobs=2, max_tokens=8)

    result = run(
        handle_completion(
            req,
            None,
            FakeState([lp_reply("Hi", logprobs=first), lp_reply("!", finished=True, logprobs=second)]),
            {},
        )
    )
    assert result["choices"][0]["logprobs"] == {
        "tokens": ["Hi", "!"],
        "token_logprobs": [-0.1, -0.2],
        "top_logprobs": [{"Hi": -0.1, "x": -1.1}, {"!": -0.2, "x": -1.2}],
        "text_offset": [0, 2],
    }

    events = parse_sse(
        run(
            _collect(
                stream_completion_chunks(
                    42,
                    CompletionRequest(
                        model="client-model", prompt="hello", logprobs=2, max_tokens=8, stream=True
                    ),
                    FakeState([lp_reply("Hi", finished=True, logprobs=first)]),
                )
            )
        )
    )
    chunk = next(event for event in events if isinstance(event, dict) and event["choices"][0]["text"])
    assert chunk["choices"][0]["logprobs"] == {
        "tokens": ["Hi"],
        "token_logprobs": [-0.1],
        "top_logprobs": [{"Hi": -0.1, "x": -1.1}],
        "text_offset": [0],
    }


def test_chat_logprobs_fail_closed_on_a_qwen_semantic_server():
    # The semantic special-token filter (armed server side, independent of the request)
    # can withhold or drop generated text, so entries could not be aligned with the
    # visible content: reject instead of returning a quietly mismatched list.
    state = FakeState([], tool_call_parser="qwen3_coder")
    resp = run(handle_chat_completion(chat_request(tools=None, logprobs=True), None, state, {}))

    assert resp.status_code == 400
    assert json.loads(resp.body)["error"]["param"] == "logprobs"


def test_reasoning_logprob_is_not_carried_to_content_delta():
    # The generation-layer half of the contract: even when a caller bypasses request
    # validation, a hidden reasoning token's entry is dropped, never attached to a
    # later visible delta (where its token string would leak).
    state = FakeState(
        [
            lp_reply("<think>thought</think>", logprobs=logprob_entry(1, "thought", -0.1)),
            lp_reply("answer", finished=True),
        ],
        reasoning_parser="qwen3",
    )
    req = chat_request(tools=None, stream=True, logprobs=True, top_logprobs=2)

    events = parse_sse(run(_collect(stream_chat_completion_chunks(42, req, state))))
    content_choice = next(
        event["choices"][0]
        for event in events
        if isinstance(event, dict)
        and event["choices"]
        and event["choices"][0]["delta"].get("content") == "answer"
    )

    assert "logprobs" not in content_choice


def test_semantic_filter_drops_entries_with_the_text_it_hides():
    # A Qwen semantic dialect hides transport markers from content; the entry that
    # describes such a token must not survive onto the next visible delta.
    state = FakeState(
        [
            lp_reply("<|audio_pad|>", logprobs=logprob_entry(1, "<|audio_pad|>", -0.3)),
            lp_reply("answer", finished=True),
        ],
        tool_call_parser="qwen3_coder",
    )
    req = chat_request(tools=None, stream=True, logprobs=True, top_logprobs=2)

    events = parse_sse(run(_collect(stream_chat_completion_chunks(42, req, state))))
    content_choices = [
        event["choices"][0]
        for event in events
        if isinstance(event, dict)
        and event["choices"]
        and event["choices"][0]["delta"].get("content")
    ]

    assert content_choices[-1]["delta"]["content"] == "answer"
    assert all("logprobs" not in choice for choice in content_choices)


def test_completion_logprobs_keep_one_entry_per_sampled_token_including_eos():
    # Upstream semantics, pinned on purpose: entries are one per sampled token, and a
    # sampled end-of-text token carries an entry (its logprob is the stop probability)
    # even though it contributes no text. A client zipping tokens to text must expect a
    # possibly-trailing non-text token; flip this test if the contract is ever changed.
    eos = logprob_entry(151643, "<|endoftext|>", -1.0)
    result = run(
        handle_completion(
            CompletionRequest(model="client-model", prompt="hello", logprobs=5, max_tokens=8),
            None,
            FakeState([lp_reply("hi", logprobs=logprob_entry(1, "hi", -0.1)),
                       lp_reply("", finished=True, logprobs=eos)]),
            {},
        )
    )

    choice = result["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["text"] == "hi"
    assert choice["logprobs"]["tokens"] == ["hi", "<|endoftext|>"]
    assert choice["logprobs"]["text_offset"] == [0, 2]
