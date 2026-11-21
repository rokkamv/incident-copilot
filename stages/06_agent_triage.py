"""Stage 6: triage an incident with an actual agent loop, not a fixed pipeline.

The model is offered three tools - classify_incident, search_similar_incidents,
record_triage_findings - with free choice, and decides itself which to call, in
what order, how many times, until it calls record_triage_findings to finish.
Unlike Stage 3's triage(), category/team and retrieval are no longer guaranteed -
they only happen if the model chooses to call the corresponding tool. Watching the
tool-call trace below (printed before the final answer) is the point of this stage.

Usage:
    uv run --env-file .env python stages/06_agent_triage.py "ERROR PaymentGateway: timeout calling StripeAPI charge endpoint"
    uv run --env-file .env python stages/06_agent_triage.py --max-turns 4 "..."
    uv run --env-file .env python stages/06_agent_triage.py --provider openai --model <...> "..."
"""

import argparse
import json
import sys
from pathlib import Path

from copilot import llm
from copilot.agent import AgentMaxTurnsExceeded, AgentTurnIncomplete, triage
from copilot.classifier import train
from copilot.config import settings
from copilot.cost import assert_known_model
from copilot.incidents import load_incidents
from copilot.triage import ModelRefused


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("error_message", help="the error or log line to triage")
    parser.add_argument(
        "--incidents-dir",
        type=Path,
        default=None,
        help="directory of incident JSON files to train the classifier on (default: data/incidents/)",
    )
    parser.add_argument(
        "--provider",
        default=None,
        choices=["anthropic", "openai"],
        help="LLM provider to call (default: LLM_PROVIDER from .env)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="model to call (default: LLM_MODEL from .env, if --provider matches LLM_PROVIDER)",
    )
    parser.add_argument(
        "--effort",
        default=None,
        choices=llm.EFFORT_LEVELS,
        help="output_config.effort level, Anthropic only (default: LLM_EFFORT from .env, else low)",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=8,
        help="give up if the agent hasn't finished within this many turns (default: 8)",
    )
    return parser


def _print_trace(tool_calls: list[dict]) -> None:
    print("tool calls:")
    for i, call in enumerate(tool_calls, start=1):
        result = call["result"]
        if call["tool"] == "search_similar_incidents" and isinstance(result, list):
            summary = f"{len(result)} match(es)"
        elif isinstance(result, (dict, list)):
            summary = json.dumps(result)
        else:
            summary = str(result)

        if call["tool"] == "record_triage_findings":
            # The full root_causes/next_steps payload is the final answer itself -
            # printed properly below, not worth dumping again as a wall of repr()
            # text on this one trace line (unlike classify_incident/search_similar_
            # incidents, whose args are small enough to show in full).
            n_causes = len(call["input"].get("root_causes", []))
            n_steps = len(call["input"].get("next_steps", []))
            args = f"{n_causes} root cause(s), {n_steps} next step(s)"
        else:
            args = ", ".join(f"{k}={v!r}" for k, v in call["input"].items())
        print(f"  [{i}] {call['tool']}({args}) -> {summary}")
    print()


def run(
    error_message: str,
    incidents_dir: Path | None,
    model: str | None,
    provider: str | None,
    effort: str,
    max_turns: int,
) -> int:
    provider = provider or settings.llm_provider

    try:
        resolved_model = llm.resolve_model(provider, model)
        assert_known_model(provider, resolved_model)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        incidents = load_incidents(incidents_dir)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    classifier_models = train(incidents)

    try:
        result = triage(
            error_message,
            classifier_models,
            model=model,
            provider=provider,
            effort=effort,
            max_turns=max_turns,
        )
    except llm.LLMAuthenticationError:
        print("error: the LLM provider rejected the API key. Check LLM_API_KEY.", file=sys.stderr)
        return 1
    except llm.LLMRateLimitError:
        print("error: rate limited by the LLM provider. Wait a bit and try again.", file=sys.stderr)
        return 1
    except llm.LLMAPIStatusError as exc:
        print(f"error: LLM provider API returned {exc.status_code}: {exc}", file=sys.stderr)
        return 1
    except llm.LLMConnectionError:
        print(
            "error: could not reach the LLM provider. Check your network connection.",
            file=sys.stderr,
        )
        return 1
    except ModelRefused:
        print("error: the model declined to triage this input.", file=sys.stderr)
        return 1
    except AgentTurnIncomplete as exc:
        print(f"error: agent turn incomplete (stop_reason={exc.stop_reason}).", file=sys.stderr)
        return 1
    except AgentMaxTurnsExceeded as exc:
        print(f"error: {exc}", file=sys.stderr)
        _print_trace(exc.tool_calls)
        return 1

    _print_trace(result["tool_calls"])

    print(f"category: {result['category']}")
    print(f"team: {result['team']}")
    print("root causes:")
    for cause in result["root_causes"]:
        print(f"  - [{cause['confidence']}] {cause['cause']} (confirm: {cause['how_to_confirm']})")
    print("next steps:")
    for step in result["next_steps"]:
        print(f"  - {step}")
    if result["similar_incidents"]:
        print("similar incidents:")
        for incident in result["similar_incidents"]:
            print(f"  - [{incident['distance']:.4f}] {incident['id']}: {incident['error_text']}")

    return 0


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    effort = args.effort or settings.llm_effort or llm.DEFAULT_EFFORT
    sys.exit(
        run(
            args.error_message,
            args.incidents_dir,
            args.model,
            args.provider,
            effort,
            args.max_turns,
        )
    )


if __name__ == "__main__":
    main()
