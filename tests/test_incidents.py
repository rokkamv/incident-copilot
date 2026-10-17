import json

import pytest

from copilot.incidents import CATEGORIES, TEAMS, load_incidents


def test_load_real_incidents():
    incidents = load_incidents()

    assert len(incidents) == 50
    for incident in incidents:
        assert incident["category"] in CATEGORIES
        assert incident["team"] in TEAMS


def test_bad_category_raises(tmp_path):
    bad = {"id": "inc-999", "error_text": "boom", "category": "not-a-category", "team": "security"}
    (tmp_path / "inc-999.json").write_text(json.dumps(bad))

    with pytest.raises(ValueError, match="unknown category"):
        load_incidents(tmp_path)


def test_missing_required_field_raises(tmp_path):
    bad = {"id": "inc-999", "category": "auth", "team": "security"}
    (tmp_path / "inc-999.json").write_text(json.dumps(bad))

    with pytest.raises(ValueError, match="missing required field"):
        load_incidents(tmp_path)


def test_empty_directory_raises(tmp_path):
    with pytest.raises(ValueError, match="No incident files found"):
        load_incidents(tmp_path)


def test_missing_directory_raises(tmp_path):
    with pytest.raises(ValueError, match="No incident files found"):
        load_incidents(tmp_path / "does-not-exist")
