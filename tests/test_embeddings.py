import types

import httpx
import pytest
import voyageai.error as voyageai_error
from google.genai import errors as genai_errors

from copilot import embeddings

# --- _require_embedding_config ----------------------------------------------


def test_require_embedding_config_raises_when_provider_missing(monkeypatch):
    monkeypatch.setattr(embeddings.settings, "embedding_provider", None)
    monkeypatch.setattr(embeddings.settings, "embedding_model", "voyage-4-lite")
    monkeypatch.setattr(embeddings.settings, "embedding_api_key", "x")
    with pytest.raises(ValueError, match="EMBEDDING_PROVIDER"):
        embeddings.embed(["hello"], input_type="document")


def test_require_embedding_config_raises_when_model_missing(monkeypatch):
    monkeypatch.setattr(embeddings.settings, "embedding_provider", "voyage")
    monkeypatch.setattr(embeddings.settings, "embedding_model", None)
    monkeypatch.setattr(embeddings.settings, "embedding_api_key", "x")
    with pytest.raises(ValueError, match="EMBEDDING_MODEL"):
        embeddings.embed(["hello"], input_type="document")


def test_require_embedding_config_raises_when_api_key_missing(monkeypatch):
    monkeypatch.setattr(embeddings.settings, "embedding_provider", "voyage")
    monkeypatch.setattr(embeddings.settings, "embedding_model", "voyage-4-lite")
    monkeypatch.setattr(embeddings.settings, "embedding_api_key", None)
    with pytest.raises(ValueError, match="EMBEDDING_API_KEY"):
        embeddings.embed(["hello"], input_type="document")


# --- Voyage adapter ----------------------------------------------------------


def _fake_voyage_result(embeddings_list):
    return types.SimpleNamespace(embeddings=embeddings_list, total_tokens=10)


class _FakeVoyageClient:
    def __init__(self, result=None, exc=None):
        self._result = result
        self._exc = exc

    def embed(self, texts, **kwargs):
        if self._exc:
            raise self._exc
        return self._result


def _install_voyage(monkeypatch, result=None, exc=None):
    monkeypatch.setattr(embeddings.settings, "embedding_provider", "voyage")
    monkeypatch.setattr(embeddings.settings, "embedding_model", "voyage-4-lite")
    monkeypatch.setattr(embeddings.settings, "embedding_api_key", "x")
    monkeypatch.setattr(
        embeddings, "_get_voyage_client", lambda api_key: _FakeVoyageClient(result, exc)
    )


def test_voyage_happy_path(monkeypatch):
    _install_voyage(monkeypatch, result=_fake_voyage_result([[0.1, 0.2], [0.3, 0.4]]))
    result = embeddings.embed(["a", "b"], input_type="document")
    assert result == [[0.1, 0.2], [0.3, 0.4]]


def test_voyage_passes_input_type_and_dimension(monkeypatch):
    monkeypatch.setattr(embeddings.settings, "embedding_dimensions", 512)
    captured = {}

    class _CapturingClient:
        def embed(self, texts, **kwargs):
            captured.update(kwargs)
            return _fake_voyage_result([[0.1]])

    _install_voyage(monkeypatch)
    monkeypatch.setattr(embeddings, "_get_voyage_client", lambda api_key: _CapturingClient())

    embeddings.embed(["a"], input_type="query")

    assert captured["input_type"] == "query"
    assert captured["output_dimension"] == 512


def test_voyage_empty_embeddings_raises(monkeypatch):
    _install_voyage(monkeypatch, result=_fake_voyage_result([]))
    with pytest.raises(embeddings.EmbeddingAPIStatusError):
        embeddings.embed(["a"], input_type="document")


def test_voyage_authentication_error_wrapped(monkeypatch):
    _install_voyage(monkeypatch, exc=voyageai_error.AuthenticationError("bad key"))
    with pytest.raises(embeddings.EmbeddingAuthenticationError):
        embeddings.embed(["a"], input_type="document")


def test_voyage_rate_limit_error_wrapped(monkeypatch):
    _install_voyage(monkeypatch, exc=voyageai_error.RateLimitError("slow down"))
    with pytest.raises(embeddings.EmbeddingRateLimitError):
        embeddings.embed(["a"], input_type="document")


def test_voyage_connection_error_wrapped(monkeypatch):
    _install_voyage(monkeypatch, exc=voyageai_error.APIConnectionError("network down"))
    with pytest.raises(embeddings.EmbeddingConnectionError):
        embeddings.embed(["a"], input_type="document")


def test_voyage_generic_api_error_wrapped(monkeypatch):
    _install_voyage(
        monkeypatch, exc=voyageai_error.APIError("boom", http_status=500)
    )
    with pytest.raises(embeddings.EmbeddingAPIStatusError) as exc_info:
        embeddings.embed(["a"], input_type="document")
    assert exc_info.value.status_code == 500


# --- Google adapter -----------------------------------------------------------


def _fake_google_response(values_list):
    embeddings_list = [types.SimpleNamespace(values=v) for v in values_list]
    return types.SimpleNamespace(embeddings=embeddings_list)


class _FakeGoogleModels:
    def __init__(self, response=None, exc=None):
        self._response = response
        self._exc = exc

    def embed_content(self, **kwargs):
        if self._exc:
            raise self._exc
        return self._response


class _FakeGoogleClient:
    def __init__(self, models):
        self.models = models


def _install_google(monkeypatch, response=None, exc=None):
    monkeypatch.setattr(embeddings.settings, "embedding_provider", "google")
    monkeypatch.setattr(embeddings.settings, "embedding_model", "gemini-embedding-001")
    monkeypatch.setattr(embeddings.settings, "embedding_api_key", "x")
    monkeypatch.setattr(
        embeddings,
        "_get_google_client",
        lambda api_key: _FakeGoogleClient(_FakeGoogleModels(response, exc)),
    )


def test_google_happy_path(monkeypatch):
    _install_google(monkeypatch, response=_fake_google_response([[0.1, 0.2], [0.3, 0.4]]))
    result = embeddings.embed(["a", "b"], input_type="document")
    assert result == [[0.1, 0.2], [0.3, 0.4]]


def test_google_task_type_mapping(monkeypatch):
    captured = {}

    class _CapturingModels:
        def embed_content(self, **kwargs):
            captured.update(kwargs)
            return _fake_google_response([[0.1]])

    _install_google(monkeypatch)
    monkeypatch.setattr(
        embeddings, "_get_google_client", lambda api_key: _FakeGoogleClient(_CapturingModels())
    )

    embeddings.embed(["a"], input_type="query")

    assert captured["config"].task_type == "RETRIEVAL_QUERY"


def test_google_empty_embeddings_raises(monkeypatch):
    _install_google(monkeypatch, response=_fake_google_response([]))
    with pytest.raises(embeddings.EmbeddingAPIStatusError):
        embeddings.embed(["a"], input_type="document")


def test_google_missing_values_raises(monkeypatch):
    response = types.SimpleNamespace(embeddings=[types.SimpleNamespace(values=None)])
    _install_google(monkeypatch, response=response)
    with pytest.raises(embeddings.EmbeddingAPIStatusError):
        embeddings.embed(["a"], input_type="document")


def test_google_authentication_error_wrapped(monkeypatch):
    _install_google(monkeypatch, exc=genai_errors.ClientError(401, {"error": {}}))
    with pytest.raises(embeddings.EmbeddingAuthenticationError):
        embeddings.embed(["a"], input_type="document")


def test_google_rate_limit_error_wrapped(monkeypatch):
    _install_google(monkeypatch, exc=genai_errors.ClientError(429, {"error": {}}))
    with pytest.raises(embeddings.EmbeddingRateLimitError):
        embeddings.embed(["a"], input_type="document")


def test_google_generic_client_error_wrapped(monkeypatch):
    _install_google(monkeypatch, exc=genai_errors.ClientError(400, {"error": {}}))
    with pytest.raises(embeddings.EmbeddingAPIStatusError) as exc_info:
        embeddings.embed(["a"], input_type="document")
    assert exc_info.value.status_code == 400


def test_google_server_error_wrapped(monkeypatch):
    _install_google(monkeypatch, exc=genai_errors.ServerError(500, {"error": {}}))
    with pytest.raises(embeddings.EmbeddingAPIStatusError) as exc_info:
        embeddings.embed(["a"], input_type="document")
    assert exc_info.value.status_code == 500


def test_google_connection_error_wrapped(monkeypatch):
    request = httpx.Request("POST", "https://generativelanguage.googleapis.com/v1/models")
    _install_google(monkeypatch, exc=httpx.ConnectError("boom", request=request))
    with pytest.raises(embeddings.EmbeddingConnectionError):
        embeddings.embed(["a"], input_type="document")
