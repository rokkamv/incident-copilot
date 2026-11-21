"""Provider-neutral LLM calling: normalizes Anthropic and OpenAI behind one interface.

Both stages/01_first_call.py and copilot.triage call through complete() instead of
constructing a provider SDK client directly, so neither has to know which provider
is active or how that provider's response/exception/tool shapes differ.
"""

import json
from dataclasses import dataclass

import anthropic
import openai

from copilot.config import settings

EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")
DEFAULT_EFFORT = "low"


@dataclass
class ToolSpec:
    name: str
    description: str
    schema: dict


@dataclass
class LLMResponse:
    text: str | None
    tool_input: dict | None
    refused: bool
    truncated: bool
    input_tokens: int
    output_tokens: int
    stop_reason: str


@dataclass
class ToolCall:
    """One tool call the model made this turn - id addresses its result back to it."""

    id: str
    name: str
    input: dict


@dataclass
class AgentTurn:
    """One turn of a multi-turn, multiple-tools, auto-tool-choice conversation.

    Unlike LLMResponse (complete()'s single forced-tool-call result), tool_call is
    one call or None - the model may respond with just text, e.g. to ask something
    before calling a tool. provider_message is the raw, provider-native shape for
    this turn; callers never inspect it directly, only pass it to
    append_assistant_turn().
    """

    text: str | None
    tool_call: ToolCall | None
    refused: bool
    truncated: bool
    input_tokens: int
    output_tokens: int
    stop_reason: str
    provider_message: dict


class LLMError(Exception):
    """Base for normalized LLM-provider errors. __cause__ holds the real provider exception."""


class LLMAuthenticationError(LLMError):
    pass


class LLMRateLimitError(LLMError):
    pass


class LLMConnectionError(LLMError):
    pass


class LLMAPIStatusError(LLMError):
    def __init__(self, message: str, status_code: int):
        self.status_code = status_code
        super().__init__(message)


class LLMMalformedToolCallError(LLMError):
    """The provider returned tool-call arguments that failed to parse as JSON."""


def _get_anthropic_client() -> anthropic.Anthropic:
    kwargs: dict = {"api_key": settings.llm_api_key}
    if settings.llm_base_url is not None:
        kwargs["base_url"] = settings.llm_base_url
    return anthropic.Anthropic(**kwargs)


def _get_openai_client() -> openai.OpenAI:
    kwargs: dict = {"api_key": settings.llm_api_key}
    if settings.llm_base_url is not None:
        kwargs["base_url"] = settings.llm_base_url
    return openai.OpenAI(**kwargs)


def _to_anthropic_tool(tool: ToolSpec) -> dict:
    return {"name": tool.name, "description": tool.description, "input_schema": tool.schema}


def _to_openai_tool(tool: ToolSpec) -> dict:
    return {
        "type": "function",
        "function": {"name": tool.name, "description": tool.description, "parameters": tool.schema},
    }


def resolve_model(provider: str, model: str | None) -> str:
    """Resolve which model to call for `provider`.

    An explicit `model` always wins. Otherwise falls back to settings.llm_model,
    but only when `provider` matches settings.llm_provider - settings.llm_model
    was configured alongside settings.llm_provider and can't be assumed valid
    for a different provider (e.g. an Anthropic model name passed to OpenAI).
    """
    if model is not None:
        return model
    if provider == settings.llm_provider:
        return settings.llm_model
    raise ValueError(
        f"LLM_MODEL is configured for provider {settings.llm_provider!r}; "
        f"pass --model explicitly when using provider {provider!r}."
    )


def complete(
    system: str,
    user_content: str,
    model: str,
    *,  # forces callers to pass max_tokens/tool/effort by name, never by position
    max_tokens: int = 4096,
    tool: ToolSpec | None = None,
    effort: str | None = None,
) -> LLMResponse:
    if settings.llm_provider == "anthropic":
        return _complete_anthropic(system, user_content, model, max_tokens, tool, effort)
    if settings.llm_provider == "openai":
        return _complete_openai(system, user_content, model, max_tokens, tool)
    raise LLMError(f"unknown provider {settings.llm_provider!r}")


def _call_anthropic(**kwargs):
    """Call messages.create() for the configured Anthropic client, translating the
    SDK's exceptions to LLM*Error. Shared by complete() and complete_turn() - the
    only piece genuinely identical between a single forced-tool call and a
    multi-turn auto-choice one is how a failed HTTP call gets normalized.

    Validates the required kwargs are present before calling the SDK, so a missing
    key (a bug in one of this module's own callers, not a provider issue) fails
    with a clear assertion here rather than an opaque error from deep in the SDK.
    """
    required = {"model", "max_tokens", "system", "messages"}
    missing = required - kwargs.keys()
    assert not missing, f"_call_anthropic missing required kwargs: {missing}"

    client = _get_anthropic_client()
    try:
        return client.messages.create(**kwargs)
    except anthropic.AuthenticationError as exc:
        raise LLMAuthenticationError(str(exc)) from exc
    except anthropic.RateLimitError as exc:
        raise LLMRateLimitError(str(exc)) from exc
    except anthropic.APIConnectionError as exc:
        raise LLMConnectionError(str(exc)) from exc
    except anthropic.APIStatusError as exc:
        raise LLMAPIStatusError(f"{exc.status_code}: {exc.message}", exc.status_code) from exc


def _call_openai(**kwargs):
    """Call chat.completions.create() for the configured OpenAI client, translating
    the SDK's exceptions to LLM*Error. See _call_anthropic for why kwargs are
    validated before the call.
    """
    required = {"model", "max_completion_tokens", "messages"}
    missing = required - kwargs.keys()
    assert not missing, f"_call_openai missing required kwargs: {missing}"

    client = _get_openai_client()
    try:
        return client.chat.completions.create(**kwargs)
    except openai.AuthenticationError as exc:
        raise LLMAuthenticationError(str(exc)) from exc
    except openai.RateLimitError as exc:
        raise LLMRateLimitError(str(exc)) from exc
    except openai.APIConnectionError as exc:
        raise LLMConnectionError(str(exc)) from exc
    except openai.APIStatusError as exc:
        raise LLMAPIStatusError(f"{exc.status_code}: {exc.message}", exc.status_code) from exc


def _complete_anthropic(
    system: str,
    user_content: str,
    model: str,
    max_tokens: int,
    tool: ToolSpec | None,
    effort: str | None,
) -> LLMResponse:
    kwargs: dict = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user_content}],
    }
    if effort is not None:
        kwargs["output_config"] = {"effort": effort}
    if tool is not None:
        kwargs["tools"] = [_to_anthropic_tool(tool)]
        kwargs["tool_choice"] = {"type": "tool", "name": tool.name}

    response = _call_anthropic(**kwargs)

    text = "".join(block.text for block in response.content if block.type == "text") or None
    tool_input = next((block.input for block in response.content if block.type == "tool_use"), None)

    return LLMResponse(
        text=text,
        tool_input=tool_input,
        refused=response.stop_reason == "refusal",
        truncated=response.stop_reason == "max_tokens",
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        stop_reason=response.stop_reason,
    )


def _complete_openai(
    system: str, user_content: str, model: str, max_tokens: int, tool: ToolSpec | None
) -> LLMResponse:
    kwargs: dict = {
        "model": model,
        "max_completion_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ],
    }
    if tool is not None:
        kwargs["tools"] = [_to_openai_tool(tool)]
        kwargs["tool_choice"] = {"type": "function", "function": {"name": tool.name}}

    response = _call_openai(**kwargs)

    choice = response.choices[0]
    tool_input = None
    if choice.message.tool_calls:
        raw_arguments = choice.message.tool_calls[0].function.arguments
        try:
            tool_input = json.loads(raw_arguments)
        except json.JSONDecodeError as exc:
            raise LLMMalformedToolCallError(
                f"non-JSON tool-call arguments: {raw_arguments!r}"
            ) from exc

    return LLMResponse(
        text=choice.message.content,
        tool_input=tool_input,
        refused=choice.message.refusal is not None,
        truncated=choice.finish_reason == "length",
        input_tokens=response.usage.prompt_tokens,
        output_tokens=response.usage.completion_tokens,
        stop_reason=choice.finish_reason,
    )


def complete_turn(
    system: str,
    messages: list[dict],
    model: str,
    tools: list[ToolSpec],
    *,
    max_tokens: int = 4096,
    effort: str | None = None,
) -> AgentTurn:
    """One turn of a multi-turn agent conversation: several tools offered, model
    chooses freely (or none) which to call - unlike complete()'s single forced tool.

    `messages` is provider-native conversation history (built by append_assistant_turn/
    append_tool_result) - `system` is always passed separately, every call, like
    complete() already does; `messages` itself holds no system entry.
    """
    if settings.llm_provider == "anthropic":
        return _complete_turn_anthropic(system, messages, model, tools, max_tokens, effort)
    if settings.llm_provider == "openai":
        return _complete_turn_openai(system, messages, model, tools, max_tokens)
    raise LLMError(f"unknown provider {settings.llm_provider!r}")


def _complete_turn_anthropic(
    system: str,
    messages: list[dict],
    model: str,
    tools: list[ToolSpec],
    max_tokens: int,
    effort: str | None,
) -> AgentTurn:
    kwargs: dict = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": messages,
        "tools": [_to_anthropic_tool(t) for t in tools],
        # disable_parallel_tool_use: both SDKs default to parallel tool calls ON -
        # this stage's loop is a plain state machine assuming at most one call per
        # turn, so it's turned off explicitly rather than assumed.
        "tool_choice": {"type": "auto", "disable_parallel_tool_use": True},
    }
    if effort is not None:
        kwargs["output_config"] = {"effort": effort}

    response = _call_anthropic(**kwargs)

    text = "".join(block.text for block in response.content if block.type == "text") or None
    tool_use_block = next((block for block in response.content if block.type == "tool_use"), None)
    tool_call = (
        ToolCall(id=tool_use_block.id, name=tool_use_block.name, input=tool_use_block.input)
        if tool_use_block is not None
        else None
    )

    return AgentTurn(
        text=text,
        tool_call=tool_call,
        refused=response.stop_reason == "refusal",
        truncated=response.stop_reason == "max_tokens",
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        stop_reason=response.stop_reason,
        # The assistant's turn is re-appended verbatim next time - Anthropic's own
        # docs show this exact pattern (re-send response.content as-is), rather than
        # us reconstructing it block-by-block.
        provider_message={"role": "assistant", "content": response.content},
    )


def _complete_turn_openai(
    system: str,
    messages: list[dict],
    model: str,
    tools: list[ToolSpec],
    max_tokens: int,
) -> AgentTurn:
    kwargs: dict = {
        "model": model,
        "max_completion_tokens": max_tokens,
        "messages": [{"role": "system", "content": system}, *messages],
        "tools": [_to_openai_tool(t) for t in tools],
        "tool_choice": "auto",
        "parallel_tool_calls": False,
    }

    response = _call_openai(**kwargs)

    choice = response.choices[0]
    assistant_message: dict = {"role": "assistant", "content": choice.message.content}
    tool_call = None
    if choice.message.tool_calls:
        raw_call = choice.message.tool_calls[0]
        raw_arguments = raw_call.function.arguments
        try:
            parsed_input = json.loads(raw_arguments)
        except json.JSONDecodeError as exc:
            raise LLMMalformedToolCallError(
                f"non-JSON tool-call arguments: {raw_arguments!r}"
            ) from exc
        tool_call = ToolCall(id=raw_call.id, name=raw_call.function.name, input=parsed_input)
        # OpenAI's tool_calls message field must be reconstructed by hand (unlike
        # Anthropic's response.content, there's no single "echo this back" object) -
        # the raw (unparsed) arguments string goes back, not the parsed dict.
        assistant_message["tool_calls"] = [
            {
                "id": raw_call.id,
                "type": "function",
                "function": {"name": raw_call.function.name, "arguments": raw_arguments},
            }
        ]

    return AgentTurn(
        text=choice.message.content,
        tool_call=tool_call,
        refused=choice.message.refusal is not None,
        truncated=choice.finish_reason == "length",
        input_tokens=response.usage.prompt_tokens,
        output_tokens=response.usage.completion_tokens,
        stop_reason=choice.finish_reason,
        provider_message=assistant_message,
    )


def append_assistant_turn(messages: list[dict], turn: AgentTurn) -> None:
    """Append this turn's assistant message to the conversation, provider-native."""
    messages.append(turn.provider_message)


def append_tool_result(
    messages: list[dict], turn: AgentTurn, result, *, is_error: bool = False
) -> None:
    """Append the result of executing turn.tool_call, so the next complete_turn()
    call sees it. `result` is JSON-serialized - callers pass plain dicts/lists/strings,
    not provider objects.
    """
    assert turn.tool_call is not None, "append_tool_result() needs a turn that called a tool"
    if settings.llm_provider == "anthropic":
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": turn.tool_call.id,
                        "content": json.dumps(result),
                        "is_error": is_error,
                    }
                ],
            }
        )
    elif settings.llm_provider == "openai":
        # OpenAI's tool-result message has no is_error field - an error is just
        # communicated through its content text.
        messages.append(
            {
                "role": "tool",
                "tool_call_id": turn.tool_call.id,
                "content": json.dumps(result),
            }
        )
    else:
        raise LLMError(f"unknown provider {settings.llm_provider!r}")
