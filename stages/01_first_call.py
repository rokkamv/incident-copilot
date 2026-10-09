"""Stage 1: ask Claude for likely causes of an error, and print what the call cost.

Usage:
    uv run python stages/01_first_call.py "ERROR PaymentService: duplicate transaction id TXN-123"
"""

import sys

import anthropic

from copilot.config import settings
from copilot.cost import estimate_cost

SYSTEM = (
    "You are a senior backend engineer helping triage a production incident. "
    "Given an error message, list the three most likely root causes, most likely first. "
    "For each, give one line on how to confirm it."
)


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    error_message = sys.argv[1]

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    response = client.messages.create(
        model=settings.copilot_model,
        max_tokens=16000,
        output_config={"effort": "low"},  # cheap and fast; try "medium" and compare
        system=SYSTEM,
        messages=[{"role": "user", "content": error_message}],
    )

    if response.stop_reason == "refusal":
        sys.exit("The model declined this request.")

    for block in response.content:
        if block.type == "text":
            print(block.text)

    usage = response.usage
    cost = estimate_cost(settings.copilot_model, usage.input_tokens, usage.output_tokens)
    print(f"\n--- {usage.input_tokens} input tokens, {usage.output_tokens} output tokens, ~${cost:.4f}")


if __name__ == "__main__":
    main()
