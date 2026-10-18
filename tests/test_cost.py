import json

import pytest

from copilot import cost
from copilot.cost import assert_known_model, estimate_cost, load_prices

FIXTURE_PRICES = {
    "anthropic": {
        "claude-opus-5-5": (4.00, 20.00),
        "claude-sonnet-5-5": (2.00, 10.00),
        "claude-haiku-5-5": (0.10, 0.50),
    },
}


@pytest.fixture(autouse=True)
def _isolated_prices(monkeypatch):
    # cost.PRICES is loaded once from the real, user-editable data/pricing.json
    # at import time - tests must not depend on its current real-world content
    # (which legitimately changes as the user adds verified entries), so every
    # test here runs against a fixed, known fixture instead.
    monkeypatch.setattr(cost, "PRICES", FIXTURE_PRICES)


def test_estimate_cost_opus():
    # 1M input at $4 + 1M output at $20
    assert estimate_cost("anthropic", "claude-opus-5-5", 1_000_000, 1_000_000) == pytest.approx(24.0)


def test_estimate_cost_unknown_anthropic_model_returns_none():
    assert estimate_cost("anthropic", "not-a-model", 1, 1) is None


def test_estimate_cost_openai_returns_none_without_raising():
    # "openai" has no entry in FIXTURE_PRICES at all.
    assert estimate_cost("openai", "whatever-the-user-configured", 1, 1) is None


def test_assert_known_model_anthropic_known_model_is_a_noop():
    assert assert_known_model("anthropic", "claude-opus-5-5") is None


def test_assert_known_model_anthropic_unknown_model_raises():
    with pytest.raises(ValueError, match="not-a-model"):
        assert_known_model("anthropic", "not-a-model")


def test_assert_known_model_openai_with_no_pricing_table_is_a_noop():
    # A provider with no pricing table at all (by design - see cost.py
    # docstring) is a no-op: nothing to validate against, must not block usage.
    assert assert_known_model("openai", "whatever-the-user-configured") is None


def test_load_prices_prints_source_path(tmp_path, capsys):
    pricing_file = tmp_path / "pricing.json"
    pricing_file.write_text(json.dumps({"anthropic": {"test-model": {"input": 1.0, "output": 2.0}}}))

    prices = load_prices(str(pricing_file))

    assert prices == {"anthropic": {"test-model": (1.0, 2.0)}}
    assert str(pricing_file) in capsys.readouterr().err
