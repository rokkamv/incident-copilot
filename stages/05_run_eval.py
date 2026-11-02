"""Stage 5: score real triage() output against the golden eval set in data/evals/.

category/team are scored by exact match against the real Stage 2 classifier.
root_cause quality is scored by a second LLM call (an "LLM-as-judge") comparing
the real output against each case's human-written expected_root_cause.

Needs a real LLM_API_KEY - every case makes two real LLM calls (one for triage,
one to judge it), so this is a manual, opt-in command, not something CI runs.

Usage:
    uv run --env-file .env python stages/05_run_eval.py
    uv run --env-file .env python stages/05_run_eval.py --evals-dir data/evals_custom
    uv run --env-file .env python stages/05_run_eval.py --effort medium
    uv run --env-file .env python stages/05_run_eval.py --provider openai --model <...>
"""

import argparse
import sys
from pathlib import Path

from copilot import llm
from copilot.classifier import train
from copilot.config import settings
from copilot.cost import assert_known_model
from copilot.eval_harness import JudgeIncompleteResponse, run_eval
from copilot.evals import load_eval_cases
from copilot.incidents import load_incidents
from copilot.triage import IncompleteResponse, ModelRefused


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--evals-dir",
        type=Path,
        default=None,
        help="directory of eval case JSON files (default: data/evals/)",
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
    return parser


def run(evals_dir: Path | None, model: str | None, provider: str | None, effort: str) -> int:
    provider = provider or settings.llm_provider

    try:
        resolved_model = llm.resolve_model(provider, model)
        assert_known_model(provider, resolved_model)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        cases = load_eval_cases(evals_dir)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    classifier_models = train(load_incidents())

    try:
        report = run_eval(cases, classifier_models, model=model, provider=provider, effort=effort)
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
        print("error: the model declined to triage one of the eval cases.", file=sys.stderr)
        return 1
    except IncompleteResponse as exc:
        print(f"error: triage response incomplete (stop_reason={exc.stop_reason}).", file=sys.stderr)
        return 1
    except JudgeIncompleteResponse as exc:
        print(f"error: judge response incomplete (stop_reason={exc.stop_reason}).", file=sys.stderr)
        return 1

    print(f"n_cases: {report['n_cases']}")
    print(f"category_accuracy: {report['category_accuracy']:.2f}")
    print(f"team_accuracy: {report['team_accuracy']:.2f}")
    print(f"root_cause_exact_pass_rate: {report['root_cause_exact_pass_rate']:.2f}")
    print(f"root_cause_mean_score: {report['root_cause_mean_score']:.2f}")

    failing = [
        case
        for case in report["cases"]
        if not (case["category_correct"] and case["team_correct"])
        or case["root_cause_verdict"] != "correct"
    ]
    if failing:
        print(f"\n{len(failing)} case(s) with at least one miss:")
        for case in failing:
            print(
                f"  {case['id']}: category_correct={case['category_correct']} "
                f"team_correct={case['team_correct']} root_cause_verdict={case['root_cause_verdict']}"
            )
            print(f"    judge reasoning: {case['root_cause_reasoning']}")

    return 0


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    effort = args.effort or settings.llm_effort or llm.DEFAULT_EFFORT
    sys.exit(run(args.evals_dir, args.model, args.provider, effort))


if __name__ == "__main__":
    main()
