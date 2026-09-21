"""derive_think_gears: the probed replacement for the per-family gear registry.

Each case fakes one model family's template behavior and asserts the derived
gears match (or improve on) what the deleted ``think_spec`` registry hardcoded.
"""
from __future__ import annotations

from freetoken.server.model_meta import derive_think_gears
from freetoken.server.openai_api import ChatCompletionRequest, chat_request_to_genspec
from freetoken.tokenizer.effort import (
    EffortProfile,
    probe_effort_profile,
    probe_thinking_profile,
)


def profile_for(render):
    return probe_thinking_profile(render, probe_effort_profile(render))


def test_qwen3_style_on_off_toggle():
    # Old registry row: ("off", "on"), default "on".
    def render(kwargs, tools):
        return f"qwen3|think={kwargs.get('enable_thinking', True)}"

    gears, default, kwargs = derive_think_gears(profile_for(render), parser_configured=True)
    assert gears == ("off", "on") and default == "on"
    assert kwargs["off"]["enable_thinking"] is False
    assert kwargs["on"]["enable_thinking"] is True


def test_qwen38_style_graded_efforts():
    # The registry had no row for Qwen3.8's gears -- it showed off/on. Derived:
    # the template's real vocabulary, ascending, with the off toggle.
    def render(kwargs, tools):
        if kwargs.get("enable_thinking") is False:
            return "qwen38|off"
        effort = kwargs.get("reasoning_effort", "xhigh")
        if effort not in ("xhigh", "medium", "low"):
            raise ValueError(f"Unexpected reasoning effort {effort}")
        return f"qwen38|{effort}"

    gears, default, kwargs = derive_think_gears(profile_for(render), parser_configured=True)
    assert gears == ("off", "low", "medium", "xhigh") and default == "xhigh"
    assert kwargs["medium"]["reasoning_effort"] == "medium"
    assert kwargs["medium"]["enable_thinking"] is True
    assert kwargs["off"]["enable_thinking"] is False


def test_gemma4_style_default_off():
    # Old registry row: ("off", "on"), default "off".
    def render(kwargs, tools):
        return f"gemma|think={bool(kwargs.get('enable_thinking'))}"

    gears, default, _ = derive_think_gears(profile_for(render), parser_configured=True)
    assert gears == ("off", "on") and default == "off"


def test_gpt_oss_style_always_on_graded():
    # Old registry row: ("low", "medium", "high"), default "medium". The
    # template grades effort but never validates it, so the derived vocabulary
    # falls back to the OpenAI triple rather than every known name.
    def render(kwargs, tools):
        return f"harmony|{kwargs.get('reasoning_effort', 'medium')}"

    gears, default, kwargs = derive_think_gears(profile_for(render), parser_configured=True)
    assert gears == ("low", "medium", "high") and default == "medium"
    assert kwargs["high"] == {"reasoning_effort": "high"}  # no toggle: always on


def test_minimax_style_always_on_no_knob():
    # Old registry row: ("on",), default "on", kwargs {}.
    def render(kwargs, tools):
        return "minimax prompt"

    gears, default, kwargs = derive_think_gears(profile_for(render), parser_configured=True)
    assert gears == ("on",) and default == "on"
    assert kwargs["on"] == {}


def test_no_reasoning_parser_offers_nothing():
    # Old registry: unknown parser -> ((), None) -> reasoning block absent.
    def render(kwargs, tools):
        return "plain prompt"

    assert derive_think_gears(profile_for(render), parser_configured=False) is None


def test_dsv4_style_toggle_plus_efforts():
    # Old registry row: ("off", "on", "max"), default "off". Derived: the
    # encoder's full vocabulary replaces the curated "on" (its low gear renders
    # exactly what "on" did), keeping default off.
    def render(kwargs, tools):
        thinking = (
            bool(tools)
            or bool(kwargs.get("enable_thinking"))
            or kwargs.get("thinking_mode") == "enabled"
        )
        if not thinking:
            return "dsv4|chat"
        effort = kwargs.get("reasoning_effort") or "low"
        assert effort in ("low", "high", "max"), f"Invalid reasoning effort: {effort}"
        return f"dsv4|think|{effort}"

    gears, default, kwargs = derive_think_gears(profile_for(render), parser_configured=True)
    assert gears == ("off", "low", "high", "max") and default == "off"
    assert kwargs["max"]["reasoning_effort"] == "max"
    assert kwargs["max"]["enable_thinking"] is True


# --------------------------------------------------------------------------- #
# The reported default follows --default-thinking-mode, so /v1/cache/status
# cannot advertise a gear the request path no longer uses.
# --------------------------------------------------------------------------- #
def _qwen38_profile():
    """The deployed Qwen3.8 template shape: an off toggle, graded efforts."""
    def render(kwargs, tools):
        if kwargs.get("enable_thinking") is False or kwargs.get("thinking_mode") == "disabled":
            return "qwen38|off"
        effort = kwargs.get("reasoning_effort") or "xhigh"
        assert effort in ("xhigh", "medium", "low"), effort
        return f"qwen38|{effort}"

    return profile_for(render)


def test_server_default_overrides_the_reported_gear():
    profile = _qwen38_profile()
    for mode, expected in (("auto", "xhigh"), ("chat", "off"), ("thinking", "xhigh")):
        gears, default, kwargs = derive_think_gears(
            profile, parser_configured=True, server_default=mode
        )
        assert default == expected, mode
        # The offered gears and their kwargs are the checkpoint's; unchanged.
        assert gears == ("off", "low", "medium", "xhigh"), mode
        assert kwargs["off"]["enable_thinking"] is False, mode


def test_server_default_is_ignored_when_the_checkpoint_offers_no_such_gear():
    # An always-thinking family with no knob to turn: "chat" has no gear to move
    # to, so the report keeps telling the truth about the checkpoint.
    def render(kwargs, tools=None):
        return "always-on"

    gears, default, _ = derive_think_gears(
        profile_for(render), parser_configured=True, server_default="chat"
    )
    assert gears == ("on",) and default == "on"


def test_server_default_alone_never_changes_auto():
    profile = _qwen38_profile()
    assert derive_think_gears(profile, parser_configured=True) == derive_think_gears(
        profile, parser_configured=True, server_default=None
    ) == derive_think_gears(profile, parser_configured=True, server_default="auto")


class _FakeManager:
    """Stands in for the frontend TokenizeManager: /v1/cache/status only peeks."""

    def __init__(self, profile):
        self._profile = profile

    def thinking_profile(self):
        return self._profile


def _geometry_for(mode):
    from types import SimpleNamespace

    from freetoken.server.api_server import _reasoning_geometry

    profile = _qwen38_profile()
    state = SimpleNamespace(
        config=SimpleNamespace(default_thinking_mode=mode),
        _frontend_tokenizer=_FakeManager(profile),
    )
    return _reasoning_geometry(state)


def test_server_default_alone_never_changes_auto():
    profile = _qwen38_profile()
    assert derive_think_gears(profile, parser_configured=True) == derive_think_gears(
        profile, parser_configured=True, server_default=None
    ) == derive_think_gears(profile, parser_configured=True, server_default="auto")


def test_reasoning_geometry_reports_the_injected_default():
    for mode, expected in (("auto", "xhigh"), ("chat", "off"), ("thinking", "xhigh")):
        assert _geometry_for(mode)["default"] == expected, mode

    from types import SimpleNamespace

    from freetoken.server.api_server import _reasoning_geometry

    # A config without the flag (any caller older than it) keeps today's answer.
    state = SimpleNamespace(
        config=SimpleNamespace(), _frontend_tokenizer=_FakeManager(_qwen38_profile())
    )
    assert _reasoning_geometry(state)["default"] == "xhigh"


def test_reported_default_gear_matches_what_the_request_path_injects():
    # The invariant the flag row exists for: /v1/cache/status names the gear an
    # uncontrolled request renders. With the flag forcing a state it names that
    # state's gear -- the checkpoint's own default effort when "on" is implicit.
    request = ChatCompletionRequest(
        model="unit-model", messages=[{"role": "user", "content": "hi"}]
    )
    _, profile_default, kwargs = derive_think_gears(_qwen38_profile(), parser_configured=True)
    for mode, expected_gear in (
        ("auto", profile_default),
        ("chat", "off"),
        ("thinking", profile_default),
    ):
        block = _geometry_for(mode)
        assert block["default"] == expected_gear, mode
        got = chat_request_to_genspec(request, {}, default_thinking_mode=mode)
        if mode == "auto":
            # Nothing is injected: the bare render IS the profile's default gear.
            assert got.chat_template_kwargs == {}, mode
        elif mode == "chat":
            assert got.chat_template_kwargs == kwargs["off"], mode
        else:
            # Broadcasting "on" and letting the template grade the effort itself:
            # this deployment's template defaults an absent effort to the same
            # gear it probed as its default, so both render the reported gear.
            assert got.chat_template_kwargs == {
                k: v for k, v in kwargs[profile_default].items() if k != "reasoning_effort"
            }, mode
            assert "reasoning_effort" not in got.chat_template_kwargs, mode
