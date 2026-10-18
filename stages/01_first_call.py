"""Stage 1: ask an LLM for likely causes of an error, and print what the call cost.

Usage:
    uv run python stages/01_first_call.py "ERROR PaymentService: duplicate transaction id TXN-123"
    uv run python stages/01_first_call.py --effort medium "..."   # try a higher effort and compare (Anthropic only)
    uv run python stages/01_first_call.py --model claude-sonnet-5-5 "..."   # try a cheaper model
    uv run python stages/01_first_call.py --provider openai --model <the model you've configured> "..."
"""

import argparse
import sys

from copilot import llm
from copilot.config import settings
from copilot.cost import assert_known_model, estimate_cost

SYSTEM = (
    "You are a senior backend engineer helping triage a production incident. "
    "Given an error message, list the three most likely root causes, most likely first. "
    "For each, give one line on how to confirm it."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("error_message", help="the error or log line to triage")
    parser.add_argument(
        "--provider",
        default=None,
        choices=["anthropic", "openai"],
        help="LLM provider to call (default: LLM_PROVIDER from .env)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="model to call (default: LLM_MODEL from .env, if --provider matches LLM_PROVIDER)",
    )
    parser.add_argument(
        "--effort",
        default=None,
        choices=llm.EFFORT_LEVELS,
        help="output_config.effort level, Anthropic only (default: LLM_EFFORT from .env, else low)",
    )
    return parser


def run(error_message: str, model: str | None, effort: str, provider: str | None = None) -> int:
    provider = provider or settings.llm_provider

    try:
        resolved_model = llm.resolve_model(provider, model)
        assert_known_model(provider, resolved_model)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if provider == "openai" and effort != "low":
        print(
            f"note: --effort has no OpenAI equivalent; ignoring (effort={effort})", file=sys.stderr
        )

    try:
        response = llm.complete(
            SYSTEM, error_message, resolved_model, max_tokens=16000, effort=effort
        )
    except llm.LLMAuthenticationError:
        print("error: the LLM provider rejected the API key. Check LLM_API_KEY.", file=sys.stderr)
        return 1
    except llm.LLMRateLimitError:
        print("error: rate limited by the LLM provider. Wait a bit and try again.", file=sys.stderr)
        return 1
    except llm.LLMAPIStatusError as exc:
        print(f"error: LLM provider API returned {exc.status_code}: {exc}", file=sys.stderr)
        return 1
    except llm.LLMConnectionError:
        print(
            "error: could not reach the LLM provider. Check your network connection.",
            file=sys.stderr,
        )
        return 1

    if response.refused:
        print("The model declined this request.", file=sys.stderr)
        return 1

    if response.truncated:
        print(
            "warning: response was cut off at the max_tokens limit; "
            "the causes below may be incomplete.",
            file=sys.stderr,
        )

    if response.text:
        print(response.text)

    cost = estimate_cost(provider, resolved_model, response.input_tokens, response.output_tokens)
    cost_str = (
        f"~${cost:.4f}" if cost is not None else "cost unknown (no pricing data for this provider)"
    )
    print(
        f"\n--- {response.input_tokens} input tokens, {response.output_tokens} output tokens, {cost_str}"
    )
    return 0


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    effort = args.effort or settings.llm_effort or llm.DEFAULT_EFFORT
    sys.exit(run(args.error_message, args.model, effort, args.provider))


if __name__ == "__main__":
    main()
