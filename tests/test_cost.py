import pytest

from copilot.cost import estimate_cost


def test_estimate_cost_opus():
    # 1M input at $4 + 1M output at $20
    assert estimate_cost("claude-opus-5-5", 1_000_000, 1_000_000) == pytest.approx(24.0)


def test_unknown_model_raises():
    with pytest.raises(KeyError):
        estimate_cost("not-a-model", 1, 1)
