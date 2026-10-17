"""Token cost estimates. Check https://www.anthropic.com/pricing before relying on these."""

# USD per million tokens: (input, output)
PRICES = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}


def assert_known_model(model: str) -> None:
    """Raise if `model` has no pricing entry.

    Call this before making an API call, so an unknown/misconfigured model
    fails before anything is billed, not after.
    """
    if model not in PRICES:
        known = ", ".join(sorted(PRICES))
        raise ValueError(f"Unknown model {model!r} for cost tracking. Known models: {known}")


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    assert_known_model(model)
    input_price, output_price = PRICES[model]
    return (input_tokens * input_price + output_tokens * output_price) / 1_000_000
