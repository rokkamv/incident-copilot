from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from copilot import embeddings, llm
from copilot.classifier import train
from copilot.incidents import load_incidents
from copilot.triage import IncompleteResponse, ModelRefused, triage


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.classifier_models = train(load_incidents())
    yield


app = FastAPI(title="Incident Copilot", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


class TriageRequest(BaseModel):
    # max_length bounds the LLM prompt size (cost/latency) and the classifier's input;
    # 4000 chars comfortably fits a pasted error/log snippet plus a short stack trace.
    error_text: str = Field(min_length=1, max_length=4000)
    # Anthropic only; ignored by the OpenAI adapter. Falls back to LLM_EFFORT from
    # .env, then "low", when omitted - see copilot.triage.triage().
    effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None


class RootCause(BaseModel):
    cause: str
    confidence: str
    how_to_confirm: str


class SimilarIncident(BaseModel):
    id: str
    error_text: str
    category: str
    team: str
    distance: float


class TriageResponse(BaseModel):
    category: str
    team: str
    root_causes: list[RootCause]
    next_steps: list[str]
    # Empty unless EMBEDDING_PROVIDER is configured (Stage 4) - see copilot.triage.triage().
    similar_incidents: list[SimilarIncident]


@app.post("/triage", response_model=TriageResponse)
def triage_endpoint(body: TriageRequest, request: Request) -> TriageResponse:
    try:
        result = triage(body.error_text, request.app.state.classifier_models, effort=body.effort)
    except ValueError as exc:
        raise HTTPException(500, f"server misconfigured: {exc}") from exc
    except llm.LLMAuthenticationError as exc:
        raise HTTPException(
            500, "the LLM provider rejected the API key. Check LLM_API_KEY."
        ) from exc
    except llm.LLMRateLimitError as exc:
        raise HTTPException(429, "rate limited by the LLM provider. Try again shortly.") from exc
    except llm.LLMConnectionError as exc:
        raise HTTPException(503, "could not reach the LLM provider.") from exc
    except llm.LLMAPIStatusError as exc:
        raise HTTPException(502, f"LLM provider API returned {exc.status_code}: {exc}") from exc
    except llm.LLMMalformedToolCallError as exc:
        raise HTTPException(502, "the LLM provider returned a malformed tool call.") from exc
    except embeddings.EmbeddingAuthenticationError as exc:
        raise HTTPException(
            500, "the embedding provider rejected the API key. Check EMBEDDING_API_KEY."
        ) from exc
    except embeddings.EmbeddingRateLimitError as exc:
        raise HTTPException(
            429, "rate limited by the embedding provider. Try again shortly."
        ) from exc
    except embeddings.EmbeddingConnectionError as exc:
        raise HTTPException(503, "could not reach the embedding provider.") from exc
    except embeddings.EmbeddingAPIStatusError as exc:
        raise HTTPException(
            502, f"embedding provider API returned {exc.status_code}: {exc}"
        ) from exc
    except ModelRefused as exc:
        raise HTTPException(422, "the model declined to triage this input.") from exc
    except IncompleteResponse as exc:
        raise HTTPException(
            502, f"LLM response incomplete (stop_reason={exc.stop_reason})."
        ) from exc
    return TriageResponse(**result)
