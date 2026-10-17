import importlib.util
import json
import pathlib

import pytest

STAGE_PATH = pathlib.Path(__file__).resolve().parent.parent / "stages" / "02_classify_incident.py"

INCIDENTS = [
    {"id": "a1", "service": "orders-service", "error_text": "connection pool exhausted on orders database", "category": "database", "team": "database-eng"},
    {"id": "a2", "service": "inventory-service", "error_text": "deadlock detected on inventory table rollback", "category": "database", "team": "database-eng"},
    {"id": "a3", "service": "billing-service", "error_text": "replication lag on billing read replica", "category": "database", "team": "database-eng"},
    {"id": "a4", "service": "user-service", "error_text": "disk space critical on primary database writes failing", "category": "database", "team": "database-eng"},
    {"id": "a5", "service": "orders-service", "error_text": "unique constraint violation duplicate key on signup", "category": "database", "team": "database-eng"},
    {"id": "b1", "service": "auth-service", "error_text": "DNS resolution failed for internal host lookup timed out", "category": "network", "team": "networking"},
    {"id": "b2", "service": "api-gateway", "error_text": "connection refused to upstream service after retries", "category": "network", "team": "networking"},
    {"id": "b3", "service": "api-gateway", "error_text": "TLS handshake timeout connecting to downstream service", "category": "network", "team": "networking"},
    {"id": "b4", "service": "search-service", "error_text": "socket timeout connecting to search cluster node", "category": "network", "team": "networking"},
    {"id": "b5", "service": "api-gateway", "error_text": "load balancer health check failing marking endpoint unhealthy", "category": "network", "team": "networking"},
]


def _load_stage():
    spec = importlib.util.spec_from_file_location("stage_02_classify_incident", STAGE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def stage():
    return _load_stage()


@pytest.fixture
def incidents_dir(tmp_path):
    for incident in INCIDENTS:
        (tmp_path / f"{incident['id']}.json").write_text(json.dumps(incident))
    return tmp_path


def test_evaluate_mode_prints_accuracy(stage, incidents_dir, capsys):
    exit_code = stage.run(None, incidents_dir, folds=5)

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "category accuracy:" in out
    assert "team accuracy:" in out


def test_predict_mode_prints_labels(stage, incidents_dir, capsys):
    exit_code = stage.run("ERROR: deadlock on database table, transaction rolled back", incidents_dir, folds=5)

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "predicted category: database" in out
    assert "predicted team: database-eng" in out


def test_empty_incidents_dir_returns_error(stage, tmp_path, capsys):
    exit_code = stage.run(None, tmp_path, folds=5)

    assert exit_code != 0
    assert "error:" in capsys.readouterr().err


def test_main_bad_folds_exits_nonzero(stage, capsys):
    with pytest.raises(SystemExit) as exc_info:
        stage.main(["--folds", "not-an-int"])

    assert exc_info.value.code != 0
    assert "usage" in capsys.readouterr().err.lower()


def test_real_dataset_evaluates_without_error(stage, capsys):
    exit_code = stage.run(None, None, folds=5)

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "category accuracy:" in out
