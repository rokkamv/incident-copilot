"""Load and validate the synthetic incident records used by Stage 2's classifier."""

import json
from pathlib import Path

CATEGORIES = (
    "database",
    "network",
    "deploy",
    "auth",
    "resource-exhaustion",
    "third-party",
    "data-quality",
)

TEAMS = (
    "database-eng",
    "networking",
    "platform-infra",
    "security",
    "payments",
    "app-backend",
)

INCIDENTS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "incidents"

_REQUIRED_FIELDS = ("id", "error_text", "category", "team")


def load_incidents(directory: Path | None = None) -> list[dict]:
    """Load every incident JSON file in `directory` (default: data/incidents/).

    Validates required fields are present and that `category`/`team` are
    known labels, raising ValueError fail-fast on the first bad record.
    """
    directory = directory or INCIDENTS_DIR
    paths = sorted(directory.glob("*.json")) if directory.exists() else []
    if not paths:
        raise ValueError(f"No incident files found in {directory}")

    incidents = []
    for path in paths:
        record = json.loads(path.read_text())
        missing = [field for field in _REQUIRED_FIELDS if field not in record]
        if missing:
            raise ValueError(f"{path}: missing required field(s) {missing}")
        if record["category"] not in CATEGORIES:
            raise ValueError(f"{path}: unknown category {record['category']!r}")
        if record["team"] not in TEAMS:
            raise ValueError(f"{path}: unknown team {record['team']!r}")
        incidents.append(record)

    return incidents