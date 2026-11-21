import types

import anthropic
import httpx
import openai
import pytest

from copilot import llm
from copilot.llm import ToolSpec

TOOL = ToolSpec(
    name="record_findings",
    description="Record findings.",
    schema={"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]},
)


# --- client construction / base_url --------------------------------------


def test_openai_client_omits_base_url_when_unset(monkeypatch):
    monkeypatch.setattr(llm.settings, "llm_api_key", "x")
    monkeypatch.setattr(llm.settings, "llm_base_url", None)
    captured = {}
    monkeypatch.setattr(
        llm.openai, "OpenAI", lambda **kwargs: captured.update(kwargs) or object()
    )
    llm._get_openai_client()
    assert "base_url" not in captured


def test_openai_client_passes_configured_base_url(monkeypatch):
    monkeypatch.setattr(llm.settings, "llm_api_key", "x")
    monkeypatch.setattr(llm.settings, "llm_base_url", "https://api.groq.com/openai/v1")
    captured = {}
    monkeypatch.setattr(
        llm.openai, "OpenAI", lambda **kwargs: captured.update(kwargs) or object()
    )
    llm._get_openai_client()
    assert captured["base_url"] == "https://api.groq.com/openai/v1"


# --- ToolSpec translation ---------------------------------------------------


def test_to_anthropic_tool_shape():
    assert llm._to_anthropic_tool(TOOL) == {
        "name": "record_findings",
        "description": "Record findings.",
        "input_schema": TOOL.schema,
    }


def test_to_openai_tool_shape():
    assert llm._to_openai_tool(TOOL) == {
        "type": "function",
        "function": {
            "name": "record_findings",
            "description": "Record findings.",
            "parameters": TOOL.schema,
        },
    }


# --- resolve_model -----------------------------------------------------------


def test_resolve_model_explicit_model_wins(monkeypatch):
    monkeypatch.setattr(llm.settings, "llm_provider", "anthropic")
    monkeypatch.setattr(llm.settings, "llm_model", "claude-opus-5-5")
    assert llm.resolve_model("anthropic", "claude-haiku-5-5") == "claude-haiku-5-5"


def test_resolve_model_same_provider_uses_configured_model(monkeypatch):
    monkeypatch.setattr(llm.settings, "llm_provider", "anthropic")
    monkeypatch.setattr(llm.settings, "llm_model", "claude-opus-5-5")
    assert llm.resolve_model("anthropic", None) == "claude-opus-5-5"


def test_resolve_model_different_provider_without_model_raises(monkeypatch):
    monkeypatch.setattr(llm.settings, "llm_provider", "anthropic")
    monkeypatch.setattr(llm.settings, "llm_model", "claude-opus-5-5")
    with pytest.raises(ValueError, match="openai"):
        llm.resolve_model("openai", None)


# --- Anthropic adapter ---------------------------------------------------


def _fake_anthropic_response(
    *,
    stop_reason="end_turn",
    text: str | None = "hello",
    tool_input=None,
    tool_id="toolu_1",
    tool_name="record_findings",
    input_tokens=10,
    output_tokens=5,
):
    content = []
    if text is not None:
        content.append(types.SimpleNamespace(type="text", text=text))
    if tool_input is not None:
        content.append(
            types.SimpleNamespace(type="tool_use", id=tool_id, name=tool_name, input=tool_input)
        )
    return types.SimpleNamespace(
        stop_reason=stop_reason,
        content=content,
        usage=types.SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )


class _FakeAnthropicMessages:
    def __init__(self, response=None, exc=None):
        self._response = response
        self._exc = exc

    def create(self, **kwargs):
        if self._exc:
            raise self._exc
        return self._response


class _FakeAnthropicClient:
    def __init__(self, messages):
        self.messages = messages


def _install_fake_anthropic(monkeypatch, response=None, exc=None):
    monkeypatch.setattr(llm.settings, "llm_provider", "anthropic")
    monkeypatch.setattr(
        llm, "_get_anthropic_client", lambda: _FakeAnthropicClient(_FakeAnthropicMessages(response, exc))
    )


def _fake_anthropic_api_error(exc_type, status_code):
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(status_code, request=request, json={"type": "error"})
    return exc_type("boom", response=response, body=None)


def test_anthropic_happy_path_text(monkeypatch):
    _install_fake_anthropic(monkeypatch, response=_fake_anthropic_response(text="hi there"))
    result = llm.complete("sys", "user", "claude-haiku-5-5")
    assert result.text == "hi there"
    assert result.tool_input is None
    assert not result.refused
    assert not result.truncated
    assert result.input_tokens == 10
    assert result.output_tokens == 5


def test_anthropic_happy_path_tool_use(monkeypatch):
    _install_fake_anthropic(
        monkeypatch, response=_fake_anthropic_response(text=None, tool_input={"x": "y"}, stop_reason="tool_use")
    )
    result = llm.complete("sys", "user", "claude-haiku-5-5", tool=TOOL)
    assert result.tool_input == {"x": "y"}


def test_anthropic_refusal(monkeypatch):
    _install_fake_anthropic(monkeypatch, response=_fake_anthropic_response(stop_reason="refusal", text=None))
    result = llm.complete("sys", "user", "claude-haiku-5-5")
    assert result.refused


def test_anthropic_max_tokens_truncated(monkeypatch):
    _install_fake_anthropic(monkeypatch, response=_fake_anthropic_response(stop_reason="max_tokens"))
    result = llm.complete("sys", "user", "claude-haiku-5-5")
    assert result.truncated


def test_anthropic_authentication_error_wrapped(monkeypatch):
    _install_fake_anthropic(monkeypatch, exc=_fake_anthropic_api_error(anthropic.AuthenticationError, 401))
    with pytest.raises(llm.LLMAuthenticationError):
        llm.complete("sys", "user", "claude-haiku-5-5")


def test_anthropic_rate_limit_error_wrapped(monkeypatch):
    _install_fake_anthropic(monkeypatch, exc=_fake_anthropic_api_error(anthropic.RateLimitError, 429))
    with pytest.raises(llm.LLMRateLimitError):
        llm.complete("sys", "user", "claude-haiku-5-5")


def test_anthropic_connection_error_wrapped(monkeypatch):
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    _install_fake_anthropic(monkeypatch, exc=anthropic.APIConnectionError(request=request))
    with pytest.raises(llm.LLMConnectionError):
        llm.complete("sys", "user", "claude-haiku-5-5")


def test_anthropic_generic_status_error_wrapped(monkeypatch):
    _install_fake_anthropic(monkeypatch, exc=_fake_anthropic_api_error(anthropic.InternalServerError, 500))
    with pytest.raises(llm.LLMAPIStatusError) as exc_info:
        llm.complete("sys", "user", "claude-haiku-5-5")
    assert exc_info.value.status_code == 500


# --- OpenAI adapter --------------------------------------------------------


def _fake_openai_response(
    *,
    content: str | None = "hello",
    tool_arguments=None,
    tool_call_id="call_1",
    tool_name="record_findings",
    refusal=None,
    finish_reason="stop",
    prompt_tokens=10,
    completion_tokens=5,
):
    tool_calls = None
    if tool_arguments is not None:
        function = types.SimpleNamespace(name=tool_name, arguments=tool_arguments)
        tool_calls = [types.SimpleNamespace(id=tool_call_id, function=function)]
    message = types.SimpleNamespace(content=content, refusal=refusal, tool_calls=tool_calls)
    choice = types.SimpleNamespace(message=message, finish_reason=finish_reason)
    return types.SimpleNamespace(
        choices=[choice],
        usage=types.SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


class _FakeOpenAICompletions:
    def __init__(self, response=None, exc=None):
        self._response = response
        self._exc = exc

    def create(self, **kwargs):
        if self._exc:
            raise self._exc
        return self._response


class _FakeOpenAIChat:
    def __init__(self, completions):
        self.completions = completions


class _FakeOpenAIClient:
    def __init__(self, completions):
        self.chat = _FakeOpenAIChat(completions)


def _install_fake_openai(monkeypatch, response=None, exc=None):
    monkeypatch.setattr(llm.settings, "llm_provider", "openai")
    monkeypatch.setattr(
        llm,
        "_get_openai_client",
        lambda: _FakeOpenAIClient(_FakeOpenAICompletions(response, exc)),
    )


def _fake_openai_api_error(exc_type, status_code):
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx.Response(status_code, request=request, json={"type": "error"})
    return exc_type("boom", response=response, body=None)


def test_openai_happy_path_text(monkeypatch):
    _install_fake_openai(monkeypatch, response=_fake_openai_response(content="hi there"))
    result = llm.complete("sys", "user", "gpt-test")
    assert result.text == "hi there"
    assert result.input_tokens == 10
    assert result.output_tokens == 5


def test_openai_tool_call_arguments_parsed_from_json_string(monkeypatch):
    _install_fake_openai(
        monkeypatch,
        response=_fake_openai_response(content=None, tool_arguments='{"x": "y"}', finish_reason="tool_calls"),
    )
    result = llm.complete("sys", "user", "gpt-test", tool=TOOL)
    assert result.tool_input == {"x": "y"}


def test_openai_malformed_tool_call_arguments_raises(monkeypatch):
    _install_fake_openai(
        monkeypatch,
        response=_fake_openai_response(content=None, tool_arguments="not json", finish_reason="tool_calls"),
    )
    with pytest.raises(llm.LLMMalformedToolCallError):
        llm.complete("sys", "user", "gpt-test", tool=TOOL)


def test_openai_refusal_field_maps_to_refused(monkeypatch):
    _install_fake_openai(monkeypatch, response=_fake_openai_response(refusal="I can't help with that"))
    result = llm.complete("sys", "user", "gpt-test")
    assert result.refused


def test_openai_length_finish_reason_maps_to_truncated(monkeypatch):
    _install_fake_openai(monkeypatch, response=_fake_openai_response(finish_reason="length"))
    result = llm.complete("sys", "user", "gpt-test")
    assert result.truncated


def test_openai_authentication_error_wrapped(monkeypatch):
    _install_fake_openai(monkeypatch, exc=_fake_openai_api_error(openai.AuthenticationError, 401))
    with pytest.raises(llm.LLMAuthenticationError):
        llm.complete("sys", "user", "gpt-test")


def test_openai_rate_limit_error_wrapped(monkeypatch):
    _install_fake_openai(monkeypatch, exc=_fake_openai_api_error(openai.RateLimitError, 429))
    with pytest.raises(llm.LLMRateLimitError):
        llm.complete("sys", "user", "gpt-test")


def test_openai_connection_error_wrapped(monkeypatch):
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    _install_fake_openai(monkeypatch, exc=openai.APIConnectionError(request=request))
    with pytest.raises(llm.LLMConnectionError):
        llm.complete("sys", "user", "gpt-test")


# --- complete_turn() / multi-turn agent conversation -------------------------

OTHER_TOOL = ToolSpec(
    name="other_tool",
    description="Another tool.",
    schema={"type": "object", "properties": {}},
)


def test_complete_turn_anthropic_text_only(monkeypatch):
    _install_fake_anthropic(monkeypatch, response=_fake_anthropic_response(text="hi there"))
    turn = llm.complete_turn("sys", [{"role": "user", "content": "hi"}], "claude-haiku-5-5", [TOOL])
    assert turn.text == "hi there"
    assert turn.tool_call is None
    assert not turn.refused
    assert not turn.truncated


def test_complete_turn_anthropic_tool_call_parsed(monkeypatch):
    _install_fake_anthropic(
        monkeypatch,
        response=_fake_anthropic_response(
            text=None, tool_input={"x": "y"}, tool_id="toolu_42", tool_name="record_findings",
            stop_reason="tool_use",
        ),
    )
    turn = llm.complete_turn("sys", [{"role": "user", "content": "hi"}], "claude-haiku-5-5", [TOOL])
    assert turn.tool_call == llm.ToolCall(id="toolu_42", name="record_findings", input={"x": "y"})
    assert turn.provider_message["role"] == "assistant"


def test_complete_turn_anthropic_sends_auto_choice_disabled_parallel_and_all_tools(monkeypatch):
    captured = {}

    class _CapturingMessages:
        def create(self, **kwargs):
            captured.update(kwargs)
            return _fake_anthropic_response(text="ok")

    monkeypatch.setattr(llm.settings, "llm_provider", "anthropic")
    monkeypatch.setattr(
        llm, "_get_anthropic_client", lambda: _FakeAnthropicClient(_CapturingMessages())
    )

    messages = [{"role": "user", "content": "hi"}]
    llm.complete_turn("sys", messages, "claude-haiku-5-5", [TOOL, OTHER_TOOL])

    assert captured["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}
    assert captured["messages"] is messages
    assert [t["name"] for t in captured["tools"]] == ["record_findings", "other_tool"]


def test_complete_turn_anthropic_refusal(monkeypatch):
    _install_fake_anthropic(
        monkeypatch, response=_fake_anthropic_response(stop_reason="refusal", text=None)
    )
    turn = llm.complete_turn("sys", [{"role": "user", "content": "hi"}], "claude-haiku-5-5", [TOOL])
    assert turn.refused


def test_complete_turn_anthropic_truncated(monkeypatch):
    _install_fake_anthropic(monkeypatch, response=_fake_anthropic_response(stop_reason="max_tokens"))
    turn = llm.complete_turn("sys", [{"role": "user", "content": "hi"}], "claude-haiku-5-5", [TOOL])
    assert turn.truncated


def test_append_assistant_turn_anthropic_appends_provider_message():
    turn = llm.AgentTurn(
        text="hi", tool_call=None, refused=False, truncated=False,
        input_tokens=1, output_tokens=1, stop_reason="end_turn",
        provider_message={"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
    )
    messages = [{"role": "user", "content": "hello"}]
    llm.append_assistant_turn(messages, turn)
    assert messages[-1] == turn.provider_message


def test_append_tool_result_anthropic_shape(monkeypatch):
    monkeypatch.setattr(llm.settings, "llm_provider", "anthropic")
    turn = llm.AgentTurn(
        text=None, tool_call=llm.ToolCall(id="toolu_1", name="search", input={}),
        refused=False, truncated=False, input_tokens=1, output_tokens=1,
        stop_reason="tool_use", provider_message={"role": "assistant", "content": []},
    )
    messages: list[dict] = []
    llm.append_tool_result(messages, turn, {"found": 1}, is_error=False)
    assert messages == [
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_1",
                    "content": '{"found": 1}',
                    "is_error": False,
                }
            ],
        }
    ]


def test_complete_turn_openai_text_only(monkeypatch):
    _install_fake_openai(monkeypatch, response=_fake_openai_response(content="hi there"))
    turn = llm.complete_turn("sys", [{"role": "user", "content": "hi"}], "gpt-test", [TOOL])
    assert turn.text == "hi there"
    assert turn.tool_call is None


def test_complete_turn_openai_tool_call_parsed(monkeypatch):
    _install_fake_openai(
        monkeypatch,
        response=_fake_openai_response(
            content=None, tool_arguments='{"x": "y"}', tool_call_id="call_42",
            tool_name="record_findings", finish_reason="tool_calls",
        ),
    )
    turn = llm.complete_turn("sys", [{"role": "user", "content": "hi"}], "gpt-test", [TOOL])
    assert turn.tool_call == llm.ToolCall(id="call_42", name="record_findings", input={"x": "y"})
    assert turn.provider_message["tool_calls"][0]["id"] == "call_42"


def test_complete_turn_openai_malformed_tool_call_raises(monkeypatch):
    _install_fake_openai(
        monkeypatch,
        response=_fake_openai_response(content=None, tool_arguments="not json", finish_reason="tool_calls"),
    )
    with pytest.raises(llm.LLMMalformedToolCallError):
        llm.complete_turn("sys", [{"role": "user", "content": "hi"}], "gpt-test", [TOOL])


def test_complete_turn_openai_sends_auto_choice_and_disables_parallel(monkeypatch):
    captured = {}

    class _CapturingCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return _fake_openai_response(content="ok")

    monkeypatch.setattr(llm.settings, "llm_provider", "openai")
    monkeypatch.setattr(
        llm, "_get_openai_client", lambda: _FakeOpenAIClient(_CapturingCompletions())
    )

    messages = [{"role": "user", "content": "hi"}]
    llm.complete_turn("sys", messages, "gpt-test", [TOOL, OTHER_TOOL])

    assert captured["tool_choice"] == "auto"
    assert captured["parallel_tool_calls"] is False
    assert captured["messages"][0] == {"role": "system", "content": "sys"}
    assert captured["messages"][1:] == messages
    assert [t["function"]["name"] for t in captured["tools"]] == ["record_findings", "other_tool"]


def test_complete_turn_openai_refusal(monkeypatch):
    _install_fake_openai(monkeypatch, response=_fake_openai_response(refusal="no"))
    turn = llm.complete_turn("sys", [{"role": "user", "content": "hi"}], "gpt-test", [TOOL])
    assert turn.refused


def test_complete_turn_openai_truncated(monkeypatch):
    _install_fake_openai(monkeypatch, response=_fake_openai_response(finish_reason="length"))
    turn = llm.complete_turn("sys", [{"role": "user", "content": "hi"}], "gpt-test", [TOOL])
    assert turn.truncated


def test_append_tool_result_openai_shape(monkeypatch):
    monkeypatch.setattr(llm.settings, "llm_provider", "openai")
    turn = llm.AgentTurn(
        text=None, tool_call=llm.ToolCall(id="call_1", name="search", input={}),
        refused=False, truncated=False, input_tokens=1, output_tokens=1,
        stop_reason="tool_calls", provider_message={"role": "assistant", "content": None, "tool_calls": []},
    )
    messages: list[dict] = []
    llm.append_tool_result(messages, turn, {"found": 1})
    assert messages == [{"role": "tool", "tool_call_id": "call_1", "content": '{"found": 1}'}]
