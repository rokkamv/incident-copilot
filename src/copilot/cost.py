"""Token cost estimates, loaded from a data file rather than hardcoded here.

Pricing is not fetched from any provider API - neither Anthropic nor OpenAI
expose a queryable pricing endpoint. See data/pricing.json (path configurable
via settings.pricing_file) for the actual numbers and where they came from.
"""

import json
import sys
from pathlib import Path

from copilot.config import settings


def load_prices(path: str | None = None) -> dict:
    """Load the provider -> model -> {input, output} pricing table.

    Prints the source file path to stderr so it's always visible where the
    numbers came from, rather than silently assumed.
    """
    resolved_path = Path(path or settings.pricing_file)
    print(f"[cost] pricing data loaded from {resolved_path}", file=sys.stderr)
    data = json.loads(resolved_path.read_text())
    return {
        provider: {model: (prices["input"], prices["output"]) for model, prices in models.items()}
        for provider, models in data.items()
    }


PRICES = load_prices()


def assert_known_model(provider: str, model: str) -> None:
    """Raise if `provider` has a pricing table but `model` isn't in it.

    A provider with no pricing table at all (e.g. "openai" until real numbers
    are added) is a no-op here - there's nothing to validate against, and that
    must not block using the provider. A provider WITH a table but an unlisted
    model (e.g. a typo'd Anthropic model name) still fails fast, before
    anything is billed.
    """
    known_models = PRICES.get(provider)
    if known_models is None:
        return
    if model not in known_models:
        known = ", ".join(sorted(known_models))
        raise ValueError(
            f"Unknown model {model!r} for cost tracking under provider {provider!r}. "
            f"Known models: {known}"
        )


def estimate_cost(provider: str, model: str, input_tokens: int, output_tokens: int) -> float | None:
    """Return the estimated cost in USD, or None if no verified pricing exists.

    Deliberately returns None instead of raising or guessing for a
    provider/model with no entry in the loaded pricing data (e.g. OpenAI,
    until real numbers are added) - see the module docstring.
    """
    known_models = PRICES.get(provider, {})
    if model not in known_models:
        return None
    input_price, output_price = known_models[model]
    return (input_tokens * input_price + output_tokens * output_price) / 1_000_000
