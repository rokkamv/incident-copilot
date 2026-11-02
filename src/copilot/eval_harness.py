"""Stage 5: score triage() output against a hand-labeled golden set.

category/team are scored by exact match against the real Stage 2 classifier's
prediction - cheap and deterministic. root_cause quality has no such ground
truth to match against directly (it's open-ended text), so it's scored by a
second LLM call (an "LLM-as-judge") comparing the real triage() output against
a human-written reference answer, via the same forced-tool-use pattern
copilot.triage.TRIAGE_TOOL already uses.
"""

import logging

from copilot import llm
from copilot.config import settings
from copilot.cost import assert_known_model, estimate_cost
from copilot.llm import ToolSpec
from copilot.triage import triage

logger = logging.getLogger(__name__)

JUDGE_TOOL = ToolSpec(
    name="record_judge_verdict",
    description=(
        "Record whether the candidate root-cause analysis identifies the same underlying "
        "root cause as the reference answer, to be returned as the evaluation verdict."
    ),
    schema={
        "type": "object",
        "properties": {
            "verdict": {
                "type": "string",
                "enum": ["correct", "partially_correct", "incorrect"],
                "description": (
                    "Whether the candidate root causes, taken together, identify the same "
                    "underlying root cause as the reference answer."
                ),
            },
            "reasoning": {
                "type": "string",
                "description": "One or two sentences explaining the verdict.",
            },
        },
        "required": ["verdict", "reasoning"],
    },
)

JUDGE_SYSTEM = (
    "You are grading an AI incident-triage assistant's root-cause analysis against a "
    "reference answer written by a human expert. The candidate provides 1-3 root causes "
    "ranked most-likely-first - this is by design, a deliberate hedge against uncertainty, "
    "not a flaw. Judge based on the TOP-RANKED (first) candidate cause: if it identifies the "
    "same underlying root cause as the reference, the verdict is 'correct' - even if the "
    "candidate also lists additional plausible secondary causes that aren't in the reference. "
    "Listing reasonable alternatives is expected behavior, not an error, and must never be "
    "penalized. Use 'partially_correct' only when the top-ranked cause is related to the "
    "reference but meaningfully incomplete or imprecise, or when the reference's cause does "
    "appear in the candidate's list but isn't ranked first. Use 'incorrect' when the "
    "reference's cause doesn't appear anywhere in the candidate's list, or the top-ranked "
    "cause materially contradicts it. Minor wording differences never matter. Record your "
    "verdict using the record_judge_verdict tool."
)

_VERDICT_SCORES = {"correct": 1.0, "partially_correct": 0.5, "incorrect": 0.0}


class JudgeIncompleteResponse(Exception):
    """Raised when the judge didn't return a verdict (and didn't refuse)."""

    def __init__(self, stop_reason: str):
        self.stop_reason = stop_reason
        super().__init__(f"unexpected stop_reason: {stop_reason}")


def judge_root_cause(
    error_text: str,
    expected_root_cause: str,
    candidate_root_causes: list[dict],
    model: str | None = None,
    provider: str | None = None,
    effort: str | None = None,
) -> dict:
    """Return {"verdict": "correct"|"partially_correct"|"incorrect", "reasoning": str}.

    Judging with the same configured model that generated the triage output is a
    known self-evaluation bias - an accepted, documented tradeoff (see decision
    0007), not something this function works around.
    """
    provider = provider or settings.llm_provider
    resolved_model = llm.resolve_model(provider, model)
    assert_known_model(provider, resolved_model)
    resolved_effort = effort if effort is not None else (settings.llm_effort or llm.DEFAULT_EFFORT)

    candidate_lines = "\n".join(
        f"- {rc['cause']} (confidence: {rc['confidence']})" for rc in candidate_root_causes
    )
    user_content = (
        f"Error: {error_text}\n\n"
        f"Reference root cause (written by a human expert):\n{expected_root_cause}\n\n"
        f"Candidate root causes (from the assistant being graded):\n{candidate_lines}"
    )

    response = llm.complete(
        JUDGE_SYSTEM,
        user_content,
        resolved_model,
        max_tokens=1024,
        tool=JUDGE_TOOL,
        effort=resolved_effort,
    )

    if response.tool_input is None:
        raise JudgeIncompleteResponse(response.stop_reason)

    cost = estimate_cost(provider, resolved_model, response.input_tokens, response.output_tokens)
    logger.info(
        "judge llm call cost=%s input_tokens=%d output_tokens=%d",
        f"${cost:.4f}" if cost is not None else "unknown",
        response.input_tokens,
        response.output_tokens,
    )

    return response.tool_input


def run_eval(cases: list[dict], classifier_models: dict, **triage_kwargs) -> dict:
    """Score every case: category/team by exact match, root cause by LLM judge.

    `triage_kwargs` (model/provider/effort) are passed through to both triage()
    and judge_root_cause() so the whole run uses one consistent configuration.
    """
    results = []
    for case in cases:
        triage_result = triage(case["error_text"], classifier_models, **triage_kwargs)
        judgment = judge_root_cause(
            case["error_text"],
            case["expected_root_cause"],
            triage_result["root_causes"],
            model=triage_kwargs.get("model"),
            provider=triage_kwargs.get("provider"),
            effort=triage_kwargs.get("effort"),
        )
        results.append(
            {
                "id": case["id"],
                "category_correct": triage_result["category"] == case["expected_category"],
                "team_correct": triage_result["team"] == case["expected_team"],
                "root_cause_verdict": judgment["verdict"],
                "root_cause_reasoning": judgment["reasoning"],
            }
        )

    n_cases = len(results)
    return {
        "n_cases": n_cases,
        "category_accuracy": sum(r["category_correct"] for r in results) / n_cases,
        "team_accuracy": sum(r["team_correct"] for r in results) / n_cases,
        "root_cause_exact_pass_rate": (
            sum(r["root_cause_verdict"] == "correct" for r in results) / n_cases
        ),
        "root_cause_mean_score": (
            sum(_VERDICT_SCORES[r["root_cause_verdict"]] for r in results) / n_cases
        ),
        "cases": results,
    }
