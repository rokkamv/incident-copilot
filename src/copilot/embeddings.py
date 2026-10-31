"""Provider-neutral embedding calls: normalizes Voyage AI and Google AI Studio.

Mirrors copilot.llm's shape - copilot.retrieval calls through embed() instead of
constructing a provider SDK client directly, so it doesn't have to know which
embedding provider is active or how that provider's response/exception shapes differ.
"""

from typing import cast

import httpx
import voyageai.error as voyageai_error
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types
from voyageai.client import Client as VoyageClient

from copilot.config import settings


class EmbeddingError(Exception):
    """Base for normalized embedding-provider errors. __cause__ holds the real exception."""


class EmbeddingAuthenticationError(EmbeddingError):
    pass


class EmbeddingRateLimitError(EmbeddingError):
    pass


class EmbeddingConnectionError(EmbeddingError):
    pass


class EmbeddingAPIStatusError(EmbeddingError):
    def __init__(self, message: str, status_code: int):
        self.status_code = status_code
        super().__init__(message)


def _require_embedding_config() -> tuple[str, str, str]:
    """Validate Stage 4 config is set, returning (provider, model, api_key).

    Returning the validated values (rather than just checking and raising) lets
    callers use them as plain `str`, not `str | None` - settings.embedding_* stay
    Optional at the Settings level since Stage 1-3 don't need them.
    """
    provider, model, api_key = (
        settings.embedding_provider,
        settings.embedding_model,
        settings.embedding_api_key,
    )
    missing = [
        name
        for name, value in (
            ("EMBEDDING_PROVIDER", provider),
            ("EMBEDDING_MODEL", model),
            ("EMBEDDING_API_KEY", api_key),
        )
        if value is None
    ]
    if missing:
        raise ValueError(
            f"Stage 4 embeddings are not configured: {', '.join(missing)} missing from .env."
        )
    assert provider is not None and model is not None and api_key is not None
    return provider, model, api_key


def _get_voyage_client(api_key: str) -> VoyageClient:
    return VoyageClient(api_key=api_key)


def _get_google_client(api_key: str) -> genai.Client:
    return genai.Client(api_key=api_key)


def embed(texts: list[str], *, input_type: str) -> list[list[float]]:
    """Return one embedding vector per input text, in order.

    `input_type` is "document" (text being stored/indexed) or "query" (text being
    searched with) - both providers encode these differently, so callers must say
    which this call is for.
    """
    provider, model, api_key = _require_embedding_config()
    if provider == "voyage":
        return _embed_voyage(texts, input_type, model, api_key)
    if provider == "google":
        return _embed_google(texts, input_type, model, api_key)
    raise EmbeddingError(f"unknown embedding provider {provider!r}")


def _embed_voyage(
    texts: list[str], input_type: str, model: str, api_key: str
) -> list[list[float]]:
    client = _get_voyage_client(api_key)
    try:
        result = client.embed(
            texts,
            model=model,
            input_type=input_type,
            output_dimension=settings.embedding_dimensions,
        )
    except voyageai_error.AuthenticationError as exc:
        raise EmbeddingAuthenticationError(str(exc)) from exc
    except voyageai_error.RateLimitError as exc:
        raise EmbeddingRateLimitError(str(exc)) from exc
    except voyageai_error.APIConnectionError as exc:
        raise EmbeddingConnectionError(str(exc)) from exc
    except voyageai_error.APIError as exc:
        raise EmbeddingAPIStatusError(str(exc), exc.http_status or 0) from exc

    # result is a dynamically-typed, dict-backed VoyageResponse wrapper (no real
    # type safety from the SDK itself) - validate it as defensively as the
    # strongly-typed Google response below, rather than trusting its shape.
    embeddings = result.embeddings
    if not embeddings:
        raise EmbeddingAPIStatusError("Voyage returned no embeddings", 0)
    for embedding in embeddings:
        if embedding is None:
            raise EmbeddingAPIStatusError("Voyage returned an embedding with no values", 0)
    # The SDK's own stub types this as List[List[float]] | List[List[int]] - the
    # int variant only occurs when output_dtype requests int8/uint8 quantization,
    # which we never pass, so this narrows back to the float list we actually get.
    return cast(list[list[float]], embeddings)


_GOOGLE_TASK_TYPES = {"document": "RETRIEVAL_DOCUMENT", "query": "RETRIEVAL_QUERY"}


def _embed_google(
    texts: list[str], input_type: str, model: str, api_key: str
) -> list[list[float]]:
    client = _get_google_client(api_key)
    config = genai_types.EmbedContentConfig(
        task_type=_GOOGLE_TASK_TYPES[input_type],
        output_dimensionality=settings.embedding_dimensions,
    )
    try:
        # list[str] is a valid ContentListUnion member at runtime (the official
        # Gemini embedding docs batch multiple strings this way) - the stub's
        # Union[ContentUnion, list[ContentUnion], list[Content]] rejects it only
        # because list is invariant, not because the value is actually wrong.
        response = client.models.embed_content(
            model=model,
            contents=cast(genai_types.ContentListUnion, texts),
            config=config,
        )
    except genai_errors.ClientError as exc:
        if exc.code in (401, 403):
            raise EmbeddingAuthenticationError(str(exc)) from exc
        if exc.code == 429:
            raise EmbeddingRateLimitError(str(exc)) from exc
        raise EmbeddingAPIStatusError(str(exc), exc.code) from exc
    except genai_errors.ServerError as exc:
        raise EmbeddingAPIStatusError(str(exc), exc.code) from exc
    except httpx.TransportError as exc:
        raise EmbeddingConnectionError(str(exc)) from exc

    if not response.embeddings:
        raise EmbeddingAPIStatusError("Google returned no embeddings", 0)

    result = []
    for embedding in response.embeddings:
        if embedding.values is None:
            raise EmbeddingAPIStatusError("Google returned an embedding with no values", 0)
        result.append(embedding.values)
    return result