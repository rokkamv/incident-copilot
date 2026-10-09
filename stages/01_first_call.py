"""Stage 1: ask Claude for likely causes of an error, and print what the call cost.

Usage:
    uv run python stages/01_first_call.py "ERROR PaymentService: duplicate transaction id TXN-123"
    uv run python stages/01_first_call.py --effort medium "..."   # try a higher effort and compare
    uv run python stages/01_first_call.py --model claude-sonnet-5-5 "..."   # try a cheaper model
"""

import argparse
import sys

import anthropic

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
        "--model",
        default=settings.copilot_model,
        help=f"Claude model to call (default: {settings.copilot_model})",
    )
    parser.add_argument(
        "--effort",
        default="low",
        choices=["low", "medium", "high", "xhigh", "max"],
        help="output_config.effort level (default: low) - try 'medium' and compare cost/quality",
    )
    return parser


def run(error_message: str, model: str, effort: str) -> int:
    try:
        assert_known_model(model)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    try:
        response = client.messages.create(
            model=model,
            max_tokens=16000,
            output_config={"effort": effort},
            system=SYSTEM,
            messages=[{"role": "user", "content": error_message}],
        )
    except anthropic.AuthenticationError:
        print("error: Anthropic rejected the API key. Check ANTHROPIC_API_KEY.", file=sys.stderr)
        return 1
    except anthropic.RateLimitError:
        print("error: rate limited by the Anthropic API. Wait a bit and try again.", file=sys.stderr)
        return 1
    except anthropic.APIStatusError as exc:
        print(f"error: Anthropic API returned {exc.status_code}: {exc.message}", file=sys.stderr)
        return 1
    except anthropic.APIConnectionError:
        print("error: could not reach the Anthropic API. Check your network connection.", file=sys.stderr)
        return 1

    if response.stop_reason == "refusal":
        print("The model declined this request.", file=sys.stderr)
        return 1

    if response.stop_reason == "max_tokens":
        print(
            "warning: response was cut off at the max_tokens limit; "
            "the causes below may be incomplete.",
            file=sys.stderr,
        )

    for block in response.content:
        if block.type == "text":
            print(block.text)

    usage = response.usage
    cost = estimate_cost(model, usage.input_tokens, usage.output_tokens)
    print(f"\n--- {usage.input_tokens} input tokens, {usage.output_tokens} output tokens, ~${cost:.4f}")
    return 0


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    sys.exit(run(args.error_message, args.model, args.effort))


if __name__ == "__main__":
    main()
