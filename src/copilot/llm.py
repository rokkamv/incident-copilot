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


def _complete_anthropic(
    system: str,
    user_content: str,
    model: str,
    max_tokens: int,
    tool: ToolSpec | None,
    effort: str | None,
) -> LLMResponse:
    client = _get_anthropic_client()
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

    try:
        response = client.messages.create(**kwargs)
    except anthropic.AuthenticationError as exc:
        raise LLMAuthenticationError(str(exc)) from exc
    except anthropic.RateLimitError as exc:
        raise LLMRateLimitError(str(exc)) from exc
    except anthropic.APIConnectionError as exc:
        raise LLMConnectionError(str(exc)) from exc
    except anthropic.APIStatusError as exc:
        raise LLMAPIStatusError(f"{exc.status_code}: {exc.message}", exc.status_code) from exc

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
    client = _get_openai_client()
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

    try:
        response = client.chat.completions.create(**kwargs)
    except openai.AuthenticationError as exc:
        raise LLMAuthenticationError(str(exc)) from exc
    except openai.RateLimitError as exc:
        raise LLMRateLimitError(str(exc)) from exc
    except openai.APIConnectionError as exc:
        raise LLMConnectionError(str(exc)) from exc
    except openai.APIStatusError as exc:
        raise LLMAPIStatusError(f"{exc.status_code}: {exc.message}", exc.status_code) from exc

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
