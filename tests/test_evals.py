import json

import pytest

from copilot.evals import load_eval_cases
from copilot.incidents import CATEGORIES, TEAMS


def test_load_real_eval_cases():
    cases = load_eval_cases()

    assert len(cases) == 15
    for case in cases:
        assert case["expected_category"] in CATEGORIES
        assert case["expected_team"] in TEAMS
        assert case["error_text"]
        assert case["expected_root_cause"]


def test_bad_category_raises(tmp_path):
    bad = {
        "id": "eval-999",
        "error_text": "boom",
        "expected_category": "not-a-category",
        "expected_team": "security",
        "expected_root_cause": "something",
    }
    (tmp_path / "eval-999.json").write_text(json.dumps(bad))

    with pytest.raises(ValueError, match="unknown expected_category"):
        load_eval_cases(tmp_path)


def test_missing_required_field_raises(tmp_path):
    bad = {"id": "eval-999", "expected_category": "auth", "expected_team": "security"}
    (tmp_path / "eval-999.json").write_text(json.dumps(bad))

    with pytest.raises(ValueError, match="missing required field"):
        load_eval_cases(tmp_path)


def test_empty_directory_raises(tmp_path):
    with pytest.raises(ValueError, match="No eval case files found"):
        load_eval_cases(tmp_path)


def test_missing_directory_raises(tmp_path):
    with pytest.raises(ValueError, match="No eval case files found"):
        load_eval_cases(tmp_path / "does-not-exist")
