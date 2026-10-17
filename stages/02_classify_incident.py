"""Stage 2: predict an incident's category and team with a classic ML baseline, or evaluate it.

Usage:
    uv run python stages/02_classify_incident.py
        # evaluate mode: stratified 5-fold CV accuracy on the synthetic incidents

    uv run python stages/02_classify_incident.py "ERROR PaymentGateway: timeout calling StripeAPI"
        # predict mode: prints predicted category and team for this text

    uv run python stages/02_classify_incident.py --folds 3
        # evaluate with fewer folds
"""

import argparse
import sys
from pathlib import Path

from copilot.classifier import evaluate, predict, train
from copilot.incidents import load_incidents


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "error_message",
        nargs="?",
        default=None,
        help="an error/log line to classify; omit to run evaluation instead",
    )
    parser.add_argument(
        "--incidents-dir",
        type=Path,
        default=None,
        help="directory of incident JSON files (default: data/incidents/)",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="number of stratified CV folds for evaluation (default: 5)",
    )
    return parser


def run(error_message: str | None, incidents_dir: Path | None, folds: int) -> int:
    try:
        incidents = load_incidents(incidents_dir)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if error_message is None:
        try:
            scores = evaluate(incidents, n_splits=folds)
        except ValueError as exc:
            print(
                # stratified k-fold needs >= folds examples per class to stratify
                f"error: cross-validation failed ({exc}). "
                f"Every category/team needs at least {folds} examples for {folds}-fold CV.",
                file=sys.stderr,
            )
            return 1
        print(f"category accuracy: {scores['category_accuracy']:.2f}")
        print(f"team accuracy: {scores['team_accuracy']:.2f}")
        return 0

    models = train(incidents)
    prediction = predict(models, error_message)
    print(f"predicted category: {prediction['category']}")
    print(f"predicted team: {prediction['team']}")
    return 0


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    sys.exit(run(args.error_message, args.incidents_dir, args.folds))


if __name__ == "__main__":
    main()
