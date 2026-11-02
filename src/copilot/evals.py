"""Load and validate the synthetic golden eval cases used by Stage 5's eval harness."""

import json
from pathlib import Path

from copilot.incidents import CATEGORIES, TEAMS

EVALS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "evals"

_REQUIRED_FIELDS = ("id", "error_text", "expected_category", "expected_team", "expected_root_cause")


def load_eval_cases(directory: Path | None = None) -> list[dict]:
    """Load every eval case JSON file in `directory` (default: data/evals/).

    Validates required fields are present and that expected_category/expected_team
    are known labels, raising ValueError fail-fast on the first bad record - same
    style as copilot.incidents.load_incidents().
    """
    directory = directory or EVALS_DIR
    paths = sorted(directory.glob("*.json")) if directory.exists() else []
    if not paths:
        raise ValueError(f"No eval case files found in {directory}")

    cases = []
    for path in paths:
        record = json.loads(path.read_text())
        missing = [field for field in _REQUIRED_FIELDS if field not in record]
        if missing:
            raise ValueError(f"{path}: missing required field(s) {missing}")
        if record["expected_category"] not in CATEGORIES:
            raise ValueError(f"{path}: unknown expected_category {record['expected_category']!r}")
        if record["expected_team"] not in TEAMS:
            raise ValueError(f"{path}: unknown expected_team {record['expected_team']!r}")
        cases.append(record)

    return cases
