"""Effort/thinking dialect handling at the OpenAI API layer.

Covers the wire-level half of the reasoning-effort pipeline: the superset
validation and DeepSeek ``thinking`` toggle in ``handle_chat_completion``, the
pre-stream render validation, and the probed vocabulary on ``/v1/models``.
The quantization itself is covered in tests/tokenizer/test_effort.py.
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.testclient import TestClient
from freetoken.message import TokenizeMsg, UserReply
from freetoken.server.model_meta import effort_toggle_kwargs
from freetoken.server.openai_api import (
    ChatCompletionRequest,
    handle_chat_completion,
    register_openai_routes,
)
from freetoken.tokenizer.effort import EffortProfile


def run(coro):
    return asyncio.run(coro)


class FakeState:
    def __init__(self, reasoning_parser: str | None = None) -> None:
        self.config = SimpleNamespace(
            mm=SimpleNamespace(text_model_only=False, disabled_encoders=frozenset()),
            model_path="/models/unit-model",
            served_model_name="unit-model",
            tool_call_parser="llama3",
            reasoning_parser=reasoning_parser,
        )
        self.sent: TokenizeMsg | None = None

    def new_user(self) -> int:
        return 42

    async def send_one(self, msg):
        self.sent = msg

    async def wait_for_ack(self, uid: int):
        yield UserReply(uid=uid, incremental_output="ok", finished=True, finish_reason="stop")


class FakeManager:
    def __init__(self, profile: EffortProfile | None = None, render_error: Exception | None = None):
        self._profile = profile
        self._render_error = render_error

    def effort_profile(self) -> EffortProfile:
        assert self._profile is not None
        return self._profile

    def render_prompt(self, msg) -> str:
        if self._render_error is not None:
            raise self._render_error
        return "rendered"


def chat_request(**overrides) -> ChatCompletionRequest:
    payload = {
        "model": "unit-model",
        "messages": [{"role": "user", "content": "hi"}],
        **overrides,
    }
    return ChatCompletionRequest(**payload)


# --------------------------------------------------------------------------- #
# effort_toggle_kwargs: the DeepSeek thinking toggle folds into template kwargs.
# --------------------------------------------------------------------------- #
OFF = {"enable_thinking": False, "thinking_mode": "disabled"}
ON = {"enable_thinking": True, "thinking_mode": "enabled"}


def test_thinking_disabled_wins_over_an_effort():
    ctk = effort_toggle_kwargs("high", {}, thinking_type="disabled")
    assert ctk == OFF


def test_thinking_enabled_forwards_the_effort():
    ctk = effort_toggle_kwargs("high", {}, thinking_type="enabled")
    assert ctk == {**ON, "reasoning_effort": "high"}


def test_thinking_enabled_alone_turns_thinking_on():
    ctk = effort_toggle_kwargs(None, {}, thinking_type="enabled")
    assert ctk == ON


def test_explicit_template_kwargs_still_win_wholesale():
    ctk = effort_toggle_kwargs("high", {"enable_thinking": False}, thinking_type="enabled")
    assert ctk == {"enable_thinking": False}


# --------------------------------------------------------------------------- #
# handle_chat_completion: superset validation and the pre-stream render check.
# --------------------------------------------------------------------------- #
def test_unknown_reasoning_effort_is_a_400():
    response = run(
        handle_chat_completion(chat_request(reasoning_effort="banana"), None, FakeState(), {})
    )
    assert isinstance(response, JSONResponse)
    assert response.status_code == 400
    assert "reasoning_effort" in json.loads(response.body)["error"]["message"]


def test_unknown_thinking_type_is_a_400():
    response = run(
        handle_chat_completion(
            chat_request(thinking={"type": "sideways"}), None, FakeState(), {}
        )
    )
    assert isinstance(response, JSONResponse)
    assert response.status_code == 400


def test_thinking_disabled_reaches_the_tokenizer_as_enable_thinking_false():
    state = FakeState(reasoning_parser="qwen3")
    response = run(
        handle_chat_completion(
            chat_request(thinking={"type": "disabled"}), None, state, {}
        )
    )
    assert not isinstance(response, JSONResponse)  # plain successful completion
    assert state.sent is not None
    assert state.sent.chat_template_kwargs == OFF


def test_off_and_mixed_case_efforts_stay_accepted():
    # effort_toggle_kwargs has always normalized case/whitespace and honored
    # "off" as a disable synonym; the superset gate must not reject them.
    for effort, expected in (
        ("off", OFF),
        ("High", {**ON, "reasoning_effort": "high"}),
        (" high ", {**ON, "reasoning_effort": "high"}),
    ):
        state = FakeState(reasoning_parser="qwen3")
        response = run(
            handle_chat_completion(chat_request(reasoning_effort=effort), None, state, {})
        )
        assert not isinstance(response, JSONResponse), effort
        assert state.sent.chat_template_kwargs == expected, effort


def test_empty_effort_is_treated_as_absent():
    state = FakeState(reasoning_parser="qwen3")
    response = run(
        handle_chat_completion(chat_request(reasoning_effort=""), None, state, {})
    )
    assert not isinstance(response, JSONResponse)
    assert state.sent.chat_template_kwargs == {}


def test_foreign_thinking_shapes_stay_ignored():
    # extra="allow" swallowed any thinking shape before the field existed;
    # a bare string, a bool, or a typeless dict must keep working unchanged.
    for shape in ("enabled", True, {}, {"budget_tokens": 1024}):
        state = FakeState(reasoning_parser="qwen3")
        response = run(
            handle_chat_completion(chat_request(thinking=shape), None, state, {})
        )
        assert not isinstance(response, JSONResponse), shape
        assert state.sent.chat_template_kwargs == {}, shape


def test_anthropic_style_thinking_dict_works():
    state = FakeState(reasoning_parser="qwen3")
    response = run(
        handle_chat_completion(
            chat_request(thinking={"type": "enabled", "budget_tokens": 1024}), None, state, {}
        )
    )
    assert not isinstance(response, JSONResponse)
    assert state.sent.chat_template_kwargs == ON


def test_stream_returns_400_when_the_template_rejects_the_render():
    state = FakeState()
    state.frontend_tokenizer = lambda: FakeManager(
        render_error=ValueError("Unexpected reasoning effort high.")
    )
    response = run(handle_chat_completion(chat_request(stream=True), None, state, {}))
    assert isinstance(response, JSONResponse)
    assert response.status_code == 400
    message = json.loads(response.body)["error"]["message"]
    assert message.startswith("could not encode request")
    assert state.sent is None  # rejected before submission


def test_stream_proceeds_without_a_frontend_tokenizer():
    # Minimal embeddings (and the unit FakeState) have no frontend tokenizer;
    # validation degrades to the old worker-side path instead of blocking.
    response = run(handle_chat_completion(chat_request(stream=True), None, FakeState(), {}))
    assert isinstance(response, StreamingResponse)


# --------------------------------------------------------------------------- #
# /v1/models: the probed vocabulary is published; absence stays None.
# --------------------------------------------------------------------------- #
def _models_payload(state) -> dict:
    app = FastAPI()
    register_openai_routes(app, lambda: state, dict)
    with TestClient(app) as client:
        response = client.get("/v1/models")
    assert response.status_code == 200
    return response.json()["data"][0]


def test_v1_models_publishes_the_probed_efforts():
    state = FakeState()
    state.frontend_tokenizer = lambda: FakeManager(
        profile=EffortProfile(
            supported=frozenset({"xhigh", "medium", "low"}),
            default="xhigh",
            consumes_effort=True,
        )
    )
    card = _models_payload(state)
    assert card["supported_reasoning_efforts"] == ["xhigh", "medium", "low"]
    assert card["default_reasoning_effort"] == "xhigh"


def test_v1_models_omits_efforts_without_a_frontend_tokenizer():
    card = _models_payload(FakeState())
    assert card["supported_reasoning_efforts"] is None
    assert card["default_reasoning_effort"] is None


def test_v1_models_omits_efforts_for_models_without_the_knob():
    state = FakeState()
    state.frontend_tokenizer = lambda: FakeManager(
        profile=EffortProfile(supported=frozenset(), default=None, consumes_effort=False)
    )
    card = _models_payload(state)
    assert card["supported_reasoning_efforts"] is None
    assert card["default_reasoning_effort"] is None


# --------------------------------------------------------------------------- #
# --default-thinking-mode: the server-wide default the request path folds in.
# --------------------------------------------------------------------------- #
def _server_kwargs(req, mode):
    """The kwargs the OpenAI handler would render: the client's thinking controls,
    then the server default. Built from the real request path so a test cannot
    agree with the injector while disagreeing with the handler."""
    from freetoken.server.openai_api import chat_request_to_genspec

    return chat_request_to_genspec(req, {}, default_thinking_mode=mode).chat_template_kwargs


#: One client channel per row: nothing, the OpenAI top-level effort, the DeepSeek
#: thinking toggle, the template kwargs themselves, and a template kwarg plus a
#: disagreeing effort.
_MATRIX_ROWS = (
    ("nothing", {}),
    ("reasoning_effort=high", {"reasoning_effort": "high"}),
    ("reasoning_effort=none", {"reasoning_effort": "none"}),
    ("reasoning_effort=low", {"reasoning_effort": "low"}),
    ("thinking.type=enabled", {"thinking": {"type": "enabled"}}),
    ("thinking.type=disabled", {"thinking": {"type": "disabled"}}),
    ("ctk={'enable_thinking': True}", {"chat_template_kwargs": {"enable_thinking": True}}),
    ("ctk={'enable_thinking': False}", {"chat_template_kwargs": {"enable_thinking": False}}),
    (
        "ctk={'thinking': True} + effort=high",
        {"chat_template_kwargs": {"thinking": True}, "reasoning_effort": "high"},
    ),
    (
        "ctk={'reasoning_effort': 'medium'}",
        {"chat_template_kwargs": {"reasoning_effort": "medium"}},
    ),
)


def _client_channels(payload):
    """True when the request expressed a thinking control through any channel."""
    return bool(
        payload.get("reasoning_effort")
        or payload.get("thinking")
        or payload.get("chat_template_kwargs")
    )


def test_default_never_overrides_an_explicit_reasoning_effort():
    # The OpenAI top-level effort and the DeepSeek toggle are client controls even
    # though they never reach chat_template_kwargs; an injected server default must
    # not swallow either one, in either direction.
    high = {**ON, "reasoning_effort": "high"}
    for label, payload, mode, expected in (
        ("effort=high, chat", {"reasoning_effort": "high"}, "chat", high),
        ("effort=high, thinking", {"reasoning_effort": "high"}, "thinking", high),
        ("effort=none, chat", {"reasoning_effort": "none"}, "chat", dict(OFF)),
        ("effort=none, thinking", {"reasoning_effort": "none"}, "thinking", dict(OFF)),
        ("thinking=enabled, chat", {"thinking": {"type": "enabled"}}, "chat", dict(ON)),
        ("thinking=disabled, thinking", {"thinking": {"type": "disabled"}}, "thinking", dict(OFF)),
    ):
        got = _server_kwargs(chat_request(**payload), mode)
        assert got == expected, (label, got)


def test_default_mode_matrix_respects_every_client_channel():
    for label, payload in _MATRIX_ROWS:
        baseline = _server_kwargs(chat_request(**payload), "auto")
        for mode in ("chat", "thinking"):
            got = _server_kwargs(chat_request(**payload), mode)
            if _client_channels(payload):
                # An explicit control, whatever channel carried it, survives the
                # server default verbatim.
                assert got == baseline, (label, mode, got)
                # ... and for template kwargs that means nothing is added.
                assert set(got) == set(payload.get("chat_template_kwargs") or baseline), (
                    label,
                    mode,
                    got,
                )
            else:
                # A request with no control of its own gets the broadcast default.
                assert got == (OFF if mode == "chat" else ON), (label, mode, got)
        # An absent/unknown default must stay a no-op whatever the request said.
        assert _server_kwargs(chat_request(**payload), None) == baseline, label
        assert _server_kwargs(chat_request(**payload), "bogus") == baseline, label


def test_default_auto_is_byte_identical_to_todays_kwargs():
    # Regression red line: with the flag left at its default the produced kwargs
    # are exactly today's -- same keys, same insertion order, same values.
    today = (
        ((), {}),
        (
            (("reasoning_effort", "high"),),
            {"enable_thinking": True, "thinking_mode": "enabled", "reasoning_effort": "high"},
        ),
        ((("reasoning_effort", "none"),), dict(OFF)),
        ((("thinking", {"type": "enabled"}),), dict(ON)),
        ((("thinking", {"type": "disabled"}),), dict(OFF)),
        ((("chat_template_kwargs", {"enable_thinking": True}),), {"enable_thinking": True}),
        (
            (("chat_template_kwargs", {"reasoning_effort": "medium"}),),
            {"reasoning_effort": "medium"},
        ),
    )
    for items, expected in today:
        got = _server_kwargs(chat_request(**dict(items)), "auto")
        assert got == expected, items
        assert list(got) == list(expected), items


def test_default_injection_broadcasts_both_spellings():
    # The fork's invariant: one broadcast, every spelling, template picks.
    assert _server_kwargs(chat_request(), "chat") == dict(OFF)
    assert _server_kwargs(chat_request(), "thinking") == dict(ON)


def test_apply_default_thinking_mode_never_mutates_the_callers_dict():
    from freetoken.server.openai_api import apply_default_thinking_mode

    original = {"other": 1}
    merged = apply_default_thinking_mode(original, "chat")
    assert merged == {**OFF, "other": 1}
    assert original == {"other": 1}


def test_anthropic_and_responses_frontends_honor_the_default():
    from freetoken.server.anthropic_api import convert_anthropic_to_genspec
    from freetoken.server.anthropic_models import AnthropicMessagesRequest
    from freetoken.server.responses_api import ResponsesRequest, convert_responses_to_genspec

    anthropic_req = AnthropicMessagesRequest.model_validate(
        {"model": "claude-x", "max_tokens": 64, "messages": [{"role": "user", "content": "hi"}]}
    )
    assert convert_anthropic_to_genspec(
        anthropic_req, {}, default_thinking_mode="chat"
    ).chat_template_kwargs == dict(OFF)
    assert convert_anthropic_to_genspec(
        anthropic_req, {}, default_thinking_mode="auto"
    ).chat_template_kwargs == {}

    responses_req = ResponsesRequest.model_validate({"model": "x", "input": "hi"})
    assert convert_responses_to_genspec(
        responses_req, {}, default_thinking_mode="thinking"
    ).chat_template_kwargs == dict(ON)
    assert convert_responses_to_genspec(
        responses_req, {}, default_thinking_mode="auto"
    ).chat_template_kwargs == {}
