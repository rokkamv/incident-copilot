"""Stage 3: triage an incident by combining the Stage 2 classifier with an LLM call.

category/team come from the classic ML classifier (Stage 2); root causes and next
steps come from an LLM via forced tool-use (Anthropic or OpenAI, see copilot.llm),
so the output is always structured.
"""

import logging

from copilot import llm, retrieval
from copilot.classifier import predict
from copilot.config import settings
from copilot.cost import assert_known_model, estimate_cost
from copilot.llm import ToolSpec

logger = logging.getLogger(__name__)

TRIAGE_TOOL = ToolSpec(
    name="record_triage_findings",
    description=(
        "Record the likely root causes and recommended next steps for this incident, "
        "to be returned as the triage result."
    ),
    schema={
        "type": "object",
        "properties": {
            "root_causes": {
                "type": "array",
                "description": "Most likely root causes, most likely first.",
                "minItems": 1,
                "maxItems": 3,
                "items": {
                    "type": "object",
                    "properties": {
                        "cause": {
                            "type": "string",
                            "description": "A concise statement of the likely root cause.",
                        },
                        "confidence": {
                            "type": "string",
                            "enum": ["low", "medium", "high"],
                            "description": "Confidence that this is the actual root cause.",
                        },
                        "how_to_confirm": {
                            "type": "string",
                            "description": "One concrete step to confirm or rule out this cause.",
                        },
                    },
                    "required": ["cause", "confidence", "how_to_confirm"],
                },
            },
            "next_steps": {
                "type": "array",
                "description": "Concrete next actions for the on-call engineer, in order.",
                "minItems": 1,
                "maxItems": 5,
                "items": {"type": "string"},
            },
        },
        "required": ["root_causes", "next_steps"],
    },
)

SYSTEM = (
    "You are a senior backend engineer helping triage a production incident. "
    "You are given an error message plus its category/team as already determined by a "
    "separate classifier - do not re-derive or contradict those. If similar past incidents "
    "are provided, use them to ground your reasoning (e.g. citing a recurring pattern), but "
    "don't assume this incident's cause is identical to theirs. Identify the most likely "
    "root causes (most likely first) and concrete next steps for the on-call engineer, and "
    "report them using the record_triage_findings tool."
)


class ModelRefused(Exception):
    """Raised when the model declines to triage this input."""


class IncompleteResponse(Exception):
    """Raised when the model didn't return the required tool call (and didn't refuse)."""

    def __init__(self, stop_reason: str):
        self.stop_reason = stop_reason
        super().__init__(f"unexpected stop_reason: {stop_reason}")


def triage(
    error_text: str,
    classifier_models: dict,
    model: str | None = None,
    provider: str | None = None,
    effort: str | None = None,
) -> dict:
    """Return {category, team, root_causes, next_steps} for `error_text`.

    `classifier_models` is the dict `classifier.train()` returns - callers (api.py)
    supply a pre-trained instance (e.g. cached on app.state) so this function never
    retrains and stays independent of any web framework. `effort` (Anthropic only;
    ignored by the OpenAI adapter) falls back to LLM_EFFORT from .env when not given
    explicitly - both are optional, unlike provider/model/api key.
    """
    provider = provider or settings.llm_provider
    resolved_model = llm.resolve_model(provider, model)
    assert_known_model(provider, resolved_model)
    resolved_effort = effort if effort is not None else (settings.llm_effort or llm.DEFAULT_EFFORT)
    category_and_team = predict(classifier_models, error_text)

    # Stage 4 retrieval is optional, not every stage needs it configured - no-op
    # when EMBEDDING_PROVIDER isn't set, so Stage 3's behavior is unchanged.
    similar_incidents = retrieval.retrieve_similar(error_text) if settings.embedding_provider else []

    user_content = (
        f"Error: {error_text}\n\n"
        f"Predicted category: {category_and_team['category']}\n"
        f"Predicted team: {category_and_team['team']}"
    )
    if similar_incidents:
        similar_lines = "\n".join(
            f"- [{s['category']}/{s['team']}] {s['error_text']}" for s in similar_incidents
        )
        user_content += f"\n\nSimilar past incidents (most similar first):\n{similar_lines}"

    response = llm.complete(
        SYSTEM,
        user_content,
        resolved_model,
        max_tokens=4096,
        tool=TRIAGE_TOOL,
        effort=resolved_effort,
    )

    if response.refused:
        raise ModelRefused()
    if response.tool_input is None:
        raise IncompleteResponse(response.stop_reason)

    cost = estimate_cost(provider, resolved_model, response.input_tokens, response.output_tokens)
    logger.info(
        "triage llm call cost=%s input_tokens=%d output_tokens=%d",
        f"${cost:.4f}" if cost is not None else "unknown",
        response.input_tokens,
        response.output_tokens,
    )

    return {
        **category_and_team,
        **response.tool_input,
        "similar_incidents": similar_incidents,
    }
