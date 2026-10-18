import pytest
from pydantic import ValidationError

from copilot.config import Settings


def test_missing_llm_provider_raises(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("LLM_MODEL", "claude-haiku-5-5")
    monkeypatch.setenv("LLM_API_KEY", "x")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_missing_llm_model_raises(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.setenv("LLM_API_KEY", "x")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_missing_llm_api_key_raises(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_MODEL", "claude-haiku-5-5")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_fully_configured_settings_succeed(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_MODEL", "claude-haiku-5-5")
    monkeypatch.setenv("LLM_API_KEY", "x")
    settings = Settings(_env_file=None)
    assert settings.llm_provider == "anthropic"
    assert settings.llm_model == "claude-haiku-5-5"
    assert settings.pricing_file == "data/pricing.json"
