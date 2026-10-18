import pytest
from fastapi.testclient import TestClient

from copilot import api, llm, triage
from copilot.classifier import train
from copilot.triage import IncompleteResponse, ModelRefused

INCIDENTS = [
    {"id": "a1", "error_text": "connection pool exhausted on orders database", "category": "database", "team": "database-eng"},
    {"id": "a2", "error_text": "deadlock detected on inventory table rollback", "category": "database", "team": "database-eng"},
    {"id": "a3", "error_text": "replication lag on billing read replica", "category": "database", "team": "database-eng"},
    {"id": "a4", "error_text": "disk space critical on primary database writes failing", "category": "database", "team": "database-eng"},
    {"id": "a5", "error_text": "unique constraint violation duplicate key on signup", "category": "database", "team": "database-eng"},
    {"id": "b1", "error_text": "DNS resolution failed for internal host lookup timed out", "category": "network", "team": "networking"},
    {"id": "b2", "error_text": "connection refused to upstream service after retries", "category": "network", "team": "networking"},
    {"id": "b3", "error_text": "TLS handshake timeout connecting to downstream service", "category": "network", "team": "networking"},
    {"id": "b4", "error_text": "socket timeout connecting to search cluster node", "category": "network", "team": "networking"},
    {"id": "b5", "error_text": "load balancer health check failing marking endpoint unhealthy", "category": "network", "team": "networking"},
]

TOOL_INPUT = {
    "root_causes": [
        {"cause": "deadlock on database table", "confidence": "high", "how_to_confirm": "check db logs"}
    ],
    "next_steps": ["Restart the stuck transaction", "Alert database-eng"],
}


@pytest.fixture
def classifier_models():
    return train(INCIDENTS)


@pytest.fixture(autouse=True)
def _deterministic_lifespan_training(monkeypatch):
    # The real data/incidents/ dataset is a weak baseline (see Stage 2) - endpoint
    # tests need deterministic category/team predictions, so lifespan trains on
    # this fixture instead of the real 50-row dataset.
    monkeypatch.setattr(api, "load_incidents", lambda: INCIDENTS)


def _fake_response(**overrides):
    defaults = {
        "text": None,
        "tool_input": TOOL_INPUT,
        "refused": False,
        "truncated": False,
        "input_tokens": 100,
        "output_tokens": 50,
        "stop_reason": "tool_use",
    }
    return llm.LLMResponse(**{**defaults, **overrides})


def _install_fake_complete(monkeypatch, response=None, exc=None):
    def _complete(*args, **kwargs):
        if exc:
            raise exc
        return response

    monkeypatch.setattr(triage.llm, "complete", _complete)


# --- unit tests for triage.triage() ---------------------------------------


def test_triage_happy_path(monkeypatch, classifier_models):
    _install_fake_complete(monkeypatch, response=_fake_response())

    result = triage.triage("deadlock on database table, transaction rolled back", classifier_models)

    assert result["category"] == "database"
    assert result["team"] == "database-eng"
    assert result["root_causes"] == TOOL_INPUT["root_causes"]
    assert result["next_steps"] == TOOL_INPUT["next_steps"]


def test_triage_refusal_raises(monkeypatch, classifier_models):
    _install_fake_complete(monkeypatch, response=_fake_response(refused=True, tool_input=None, stop_reason="refusal"))

    with pytest.raises(ModelRefused):
        triage.triage("boom", classifier_models)


def test_triage_incomplete_response_raises(monkeypatch, classifier_models):
    _install_fake_complete(monkeypatch, response=_fake_response(tool_input=None, stop_reason="max_tokens"))

    with pytest.raises(IncompleteResponse) as exc_info:
        triage.triage("boom", classifier_models)
    assert exc_info.value.stop_reason == "max_tokens"


def test_triage_different_provider_without_model_fails_before_any_call(monkeypatch, classifier_models):
    monkeypatch.setattr(triage.settings, "llm_provider", "anthropic")
    monkeypatch.setattr(triage.settings, "llm_model", "claude-haiku-5-5")

    def _should_not_be_called(*args, **kwargs):
        raise AssertionError("llm.complete should not be called")

    monkeypatch.setattr(triage.llm, "complete", _should_not_be_called)

    with pytest.raises(ValueError):
        triage.triage("boom", classifier_models, provider="openai")


def test_triage_authentication_error_propagates(monkeypatch, classifier_models):
    _install_fake_complete(monkeypatch, exc=llm.LLMAuthenticationError("boom"))

    with pytest.raises(llm.LLMAuthenticationError):
        triage.triage("boom", classifier_models)


def test_triage_rate_limit_error_propagates(monkeypatch, classifier_models):
    _install_fake_complete(monkeypatch, exc=llm.LLMRateLimitError("boom"))

    with pytest.raises(llm.LLMRateLimitError):
        triage.triage("boom", classifier_models)


def test_triage_connection_error_propagates(monkeypatch, classifier_models):
    _install_fake_complete(monkeypatch, exc=llm.LLMConnectionError("boom"))

    with pytest.raises(llm.LLMConnectionError):
        triage.triage("boom", classifier_models)


def _install_capturing_complete(monkeypatch, response):
    captured = {}

    def _complete(*args, **kwargs):
        captured["effort"] = kwargs.get("effort")
        return response

    monkeypatch.setattr(triage.llm, "complete", _complete)
    return captured


def test_triage_explicit_effort_wins(monkeypatch, classifier_models):
    monkeypatch.setattr(triage.settings, "llm_effort", "high")
    captured = _install_capturing_complete(monkeypatch, _fake_response())

    triage.triage("boom", classifier_models, effort="medium")

    assert captured["effort"] == "medium"


def test_triage_falls_back_to_env_effort(monkeypatch, classifier_models):
    monkeypatch.setattr(triage.settings, "llm_effort", "high")
    captured = _install_capturing_complete(monkeypatch, _fake_response())

    triage.triage("boom", classifier_models)

    assert captured["effort"] == "high"


def test_triage_falls_back_to_low_when_nothing_configured(monkeypatch, classifier_models):
    monkeypatch.setattr(triage.settings, "llm_effort", None)
    captured = _install_capturing_complete(monkeypatch, _fake_response())

    triage.triage("boom", classifier_models)

    assert captured["effort"] == "low"


# --- FastAPI-level tests through /triage -----------------------------------


def test_endpoint_happy_path(monkeypatch):
    _install_fake_complete(monkeypatch, response=_fake_response())

    with TestClient(api.app) as client:
        response = client.post("/triage", json={"error_text": "deadlock on database table"})

    assert response.status_code == 200
    body = response.json()
    assert body["category"] == "database"
    assert body["team"] == "database-eng"
    assert body["root_causes"][0]["cause"] == "deadlock on database table"
    assert body["next_steps"] == TOOL_INPUT["next_steps"]


def test_endpoint_empty_error_text_returns_422():
    with TestClient(api.app) as client:
        response = client.post("/triage", json={"error_text": ""})

    assert response.status_code == 422


def test_endpoint_invalid_effort_returns_422():
    with TestClient(api.app) as client:
        response = client.post("/triage", json={"error_text": "boom", "effort": "not-a-level"})

    assert response.status_code == 422


def test_endpoint_effort_passed_through_to_triage(monkeypatch):
    captured = {}

    def _complete(*args, **kwargs):
        captured["effort"] = kwargs.get("effort")
        return _fake_response()

    monkeypatch.setattr(triage.llm, "complete", _complete)

    with TestClient(api.app) as client:
        response = client.post("/triage", json={"error_text": "boom", "effort": "medium"})

    assert response.status_code == 200
    assert captured["effort"] == "medium"


def test_endpoint_authentication_error_returns_500(monkeypatch):
    _install_fake_complete(monkeypatch, exc=llm.LLMAuthenticationError("boom"))

    with TestClient(api.app) as client:
        response = client.post("/triage", json={"error_text": "boom"})

    assert response.status_code == 500
    assert "API key" in response.json()["detail"]


def test_endpoint_rate_limit_error_returns_429(monkeypatch):
    _install_fake_complete(monkeypatch, exc=llm.LLMRateLimitError("boom"))

    with TestClient(api.app) as client:
        response = client.post("/triage", json={"error_text": "boom"})

    assert response.status_code == 429


def test_endpoint_connection_error_returns_503(monkeypatch):
    _install_fake_complete(monkeypatch, exc=llm.LLMConnectionError("boom"))

    with TestClient(api.app) as client:
        response = client.post("/triage", json={"error_text": "boom"})

    assert response.status_code == 503


def test_endpoint_generic_api_status_error_returns_502(monkeypatch):
    _install_fake_complete(monkeypatch, exc=llm.LLMAPIStatusError("boom", 500))

    with TestClient(api.app) as client:
        response = client.post("/triage", json={"error_text": "boom"})

    assert response.status_code == 502


def test_endpoint_malformed_tool_call_returns_502(monkeypatch):
    _install_fake_complete(monkeypatch, exc=llm.LLMMalformedToolCallError("boom"))

    with TestClient(api.app) as client:
        response = client.post("/triage", json={"error_text": "boom"})

    assert response.status_code == 502


def test_endpoint_refusal_returns_422(monkeypatch):
    _install_fake_complete(monkeypatch, response=_fake_response(refused=True, tool_input=None, stop_reason="refusal"))

    with TestClient(api.app) as client:
        response = client.post("/triage", json={"error_text": "boom"})

    assert response.status_code == 422


def test_endpoint_incomplete_response_returns_502(monkeypatch):
    _install_fake_complete(monkeypatch, response=_fake_response(tool_input=None, stop_reason="max_tokens"))

    with TestClient(api.app) as client:
        response = client.post("/triage", json={"error_text": "boom"})

    assert response.status_code == 502
