"""Stage 6: an actual agent loop, not a fixed pipeline.

Stage 3's triage() calls the classifier and (optionally) retrieval unconditionally,
in a fixed order, then makes one forced LLM tool call. Here, the model is offered
three tools - classify_incident, search_similar_incidents, record_triage_findings -
with free ("auto") choice, and decides itself which to call, in what order, how many
times, across multiple turns, until it calls record_triage_findings to finish.

This is a deliberate trade-off, not a free upgrade: category/team are no longer
guaranteed (the model might never call classify_incident), and retrieval usage is
0-to-many model-chosen calls with model-chosen queries, not one deterministic call.
Autonomy is gained; a guarantee is spent.
"""

import logging

from copilot import llm, retrieval
from copilot.classifier import predict
from copilot.config import settings
from copilot.cost import assert_known_model, estimate_cost
from copilot.llm import ToolSpec
from copilot.triage import TRIAGE_TOOL as RECORD_TRIAGE_FINDINGS_TOOL
from copilot.triage import ModelRefused

logger = logging.getLogger(__name__)

CLASSIFY_INCIDENT_TOOL = ToolSpec(
    name="classify_incident",
    description=(
        "Classify the incident you are currently triaging into its category and team, "
        "using the trained classifier. Takes no arguments - it always classifies the "
        "current incident, never a different string. Returns {category, team}. "
        "Deterministic, so calling it more than once is wasteful but harmless."
    ),
    schema={"type": "object", "properties": {}, "required": []},
)

SEARCH_SIMILAR_INCIDENTS_TOOL = ToolSpec(
    name="search_similar_incidents",
    description=(
        "Search past incidents for ones similar to a query you choose (usually the error "
        "text being triaged, or a reformulation of it), to find grounding context before "
        "diagnosing root causes. Returns up to k matches, nearest first, each "
        "{id, error_text, category, team, distance} (lower distance = more similar). "
        "Returns an empty list if retrieval isn't configured for this deployment - treat "
        "that as 'no grounding available', not an error. You may call this zero or more "
        "times, e.g. retrying with a different query."
    ),
    schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Text to search similar past incidents for.",
            },
            "k": {
                "type": "integer",
                "minimum": 1,
                "maximum": 10,
                "description": "Number of matches to return (default 3).",
            },
        },
        "required": ["query"],
    },
)

TOOLS = [CLASSIFY_INCIDENT_TOOL, SEARCH_SIMILAR_INCIDENTS_TOOL, RECORD_TRIAGE_FINDINGS_TOOL]

AGENT_SYSTEM = (
    "You are a senior backend engineer triaging a production incident, with three tools "
    "available: classify_incident (no arguments - classifies the incident you were given), "
    "search_similar_incidents (search past incidents with a query you choose, zero or more "
    "times), and record_triage_findings (root_causes and next_steps - this is the ONLY way "
    "to finish; you must call it eventually). A reasonable approach is to classify first, "
    "optionally search for grounding context, then record your findings - but the order and "
    "which tools you use (if any besides finishing) are your choice. You have a limited "
    "number of turns, so don't call tools needlessly."
)


class AgentTurnIncomplete(Exception):
    """Raised when a turn ended in neither a tool call nor usable text (e.g. truncated),
    or record_triage_findings was called without its required fields."""

    def __init__(self, stop_reason: str):
        self.stop_reason = stop_reason
        super().__init__(f"unexpected stop_reason: {stop_reason}")


class AgentMaxTurnsExceeded(Exception):
    """Raised when the agent didn't call record_triage_findings within max_turns.

    Carries the partial tool_calls trace, so a caller can see what the agent was
    doing right up to the point it ran out of budget - a loud, debuggable failure,
    not a silent degraded result.
    """

    def __init__(self, max_turns: int, tool_calls: list[dict]):
        self.max_turns = max_turns
        self.tool_calls = tool_calls
        super().__init__(f"agent did not finish within {max_turns} turns")


def triage(
    error_text: str,
    classifier_models: dict,
    model: str | None = None,
    provider: str | None = None,
    effort: str | None = None,
    max_turns: int = 8,
) -> dict:
    """Return {category, team, root_causes, next_steps, similar_incidents, tool_calls}.

    Unlike triage.triage(), category/team are None unless the model chose to call
    classify_incident, and similar_incidents is the union (deduped by id) of every
    search_similar_incidents call the model made - zero, one, or several, in
    whatever order/with whatever queries it picked. tool_calls is this module's own
    addition: the ordered trace of every tool call made, for callers to display.
    """
    provider = provider or settings.llm_provider
    resolved_model = llm.resolve_model(provider, model)
    assert_known_model(provider, resolved_model)
    resolved_effort = effort if effort is not None else (settings.llm_effort or llm.DEFAULT_EFFORT)

    messages: list[dict] = [{"role": "user", "content": f"Error: {error_text}"}]
    category_and_team: dict = {"category": None, "team": None}
    similar_incidents: list[dict] = []
    seen_incident_ids: set[str] = set()
    tool_calls: list[dict] = []
    total_input_tokens = total_output_tokens = 0

    for _ in range(max_turns):
        turn = llm.complete_turn(
            AGENT_SYSTEM, messages, resolved_model, TOOLS, max_tokens=4096, effort=resolved_effort
        )
        total_input_tokens += turn.input_tokens
        total_output_tokens += turn.output_tokens

        if turn.refused:
            raise ModelRefused()

        llm.append_assistant_turn(messages, turn)

        if turn.tool_call is None:
            if turn.truncated:
                raise AgentTurnIncomplete(turn.stop_reason)
            # A stray text-only turn (e.g. the model thinking out loud) isn't an
            # error - nudge it toward a tool and keep going, still bounded by
            # max_turns.
            messages.append(
                {
                    "role": "user",
                    "content": "Continue: call a tool, or call record_triage_findings to finish.",
                }
            )
            continue

        name, tool_input = turn.tool_call.name, turn.tool_call.input

        if name == "classify_incident":
            result = predict(classifier_models, error_text)
            category_and_team = result

        elif name == "search_similar_incidents":
            if settings.embedding_provider:
                matches = retrieval.retrieve_similar(
                    tool_input["query"], k=tool_input.get("k", 3)
                )
                for match in matches:
                    if match["id"] not in seen_incident_ids:
                        seen_incident_ids.add(match["id"])
                        similar_incidents.append(match)
                result = matches
            else:
                result = {"error": "retrieval not configured", "results": []}

        elif name == "record_triage_findings":
            if "root_causes" not in tool_input or "next_steps" not in tool_input:
                raise AgentTurnIncomplete(turn.stop_reason)
            tool_calls.append({"tool": name, "input": tool_input, "result": "FINISHED"})
            cost = estimate_cost(
                provider, resolved_model, total_input_tokens, total_output_tokens
            )
            logger.info(
                "agent triage cost=%s input_tokens=%d output_tokens=%d turns=%d",
                f"${cost:.4f}" if cost is not None else "unknown",
                total_input_tokens,
                total_output_tokens,
                len(tool_calls),
            )
            return {
                **category_and_team,
                **tool_input,
                "similar_incidents": similar_incidents,
                "tool_calls": tool_calls,
            }

        else:
            # Defensive only - the model was offered exactly these three tools.
            result = {"error": f"unknown tool {name!r}"}

        tool_calls.append({"tool": name, "input": tool_input, "result": result})
        llm.append_tool_result(messages, turn, result)

    raise AgentMaxTurnsExceeded(max_turns, tool_calls)
