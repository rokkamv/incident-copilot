import pytest

from copilot import classifier, eval_harness, llm
from copilot.eval_harness import JudgeIncompleteResponse, run_eval
from copilot.triage import settings as triage_settings


@pytest.fixture(autouse=True)
def _no_retrieval_by_default(monkeypatch):
    # run_eval() calls triage() for real - without this, these tests would
    # silently start hitting the real embedding provider and Postgres the
    # moment EMBEDDING_PROVIDER is set for real Stage 4/5 testing, exactly the
    # test-isolation gap already fixed once in test_triage.py.
    monkeypatch.setattr(triage_settings, "embedding_provider", None)

CLASSIFIER_INCIDENTS = [
    {"id": "a1", "error_text": "connection pool exhausted on orders database", "category": "database", "team": "database-eng"},
    {"id": "a2", "error_text": "deadlock detected on inventory table rollback", "category": "database", "team": "database-eng"},
    {"id": "a3", "error_text": "replication lag on billing read replica", "category": "database", "team": "database-eng"},
    {"id": "a4", "error_text": "disk space critical on primary database writes failing", "category": "database", "team": "database-eng"},
    {"id": "a5", "error_text": "unique constraint violation duplicate key on signup", "category": "database", "team": "database-eng"},
    {"id": "b1", "error_text": "DNS resolution failed for internal host lookup timed out", "category": "network", "team": "networking"},
    {"id": "b2", "error_text": "connection refused to upstream service after retries", "category": "network", "team": "networking"},
    {"id": "b3", "error_text": "TLS handshake timeout connecting to downstream service", "category": "network", "team": "networking"},
    {"id": "b4", "error_text": "socket timeout connecting to search cluster node", "category": "network", "team": "networking"},
    {"id": "b5", "error_text": "load balancer health check failing marking endpoint unhealthy", "category": "network", "team": "networking"},
]

EVAL_CASES = [
    {
        "id": "eval-001",
        "error_text": "connection pool exhausted on orders database",
        "expected_category": "database",
        "expected_team": "database-eng",
        "expected_root_cause": "connection pool too small for concurrent load",
    },
    {
        "id": "eval-002",
        "error_text": "DNS resolution failed for internal host lookup timed out",
        "expected_category": "network",
        "expected_team": "networking",
        "expected_root_cause": "internal DNS resolution failure",
    },
]

TRIAGE_TOOL_INPUT = {
    "root_causes": [
        {"cause": "deadlock on database table", "confidence": "high", "how_to_confirm": "check db logs"}
    ],
    "next_steps": ["Restart the stuck transaction", "Alert database-eng"],
}


@pytest.fixture
def classifier_models():
    return classifier.train(CLASSIFIER_INCIDENTS)


def _fake_response(tool_input, stop_reason="tool_use"):
    return llm.LLMResponse(
        text=None,
        tool_input=tool_input,
        refused=False,
        truncated=False,
        input_tokens=100,
        output_tokens=50,
        stop_reason=stop_reason,
    )


def _install_fake_complete(monkeypatch, judge_verdicts):
    judge_calls = iter(judge_verdicts)

    def _complete(system, user_content, model, *, max_tokens=4096, tool=None, effort=None):
        # Both triage() and judge_root_cause() always pass a tool - this fake
        # stands in for both call sites, dispatched by which tool was forced.
        assert tool is not None
        if tool.name == "record_triage_findings":
            return _fake_response(TRIAGE_TOOL_INPUT)
        if tool.name == "record_judge_verdict":
            return _fake_response(next(judge_calls))
        raise AssertionError(f"unexpected tool {tool.name}")

    monkeypatch.setattr(llm, "complete", _complete)


# --- run_eval() --------------------------------------------------------------


def test_run_eval_aggregates_category_team_and_judge_scores(monkeypatch, classifier_models):
    _install_fake_complete(
        monkeypatch,
        [
            {"verdict": "correct", "reasoning": "matches"},
            {"verdict": "correct", "reasoning": "matches"},
        ],
    )

    report = run_eval(EVAL_CASES, classifier_models)

    assert report["n_cases"] == 2
    assert report["category_accuracy"] == 1.0
    assert report["team_accuracy"] == 1.0
    assert report["root_cause_exact_pass_rate"] == 1.0
    assert report["root_cause_mean_score"] == 1.0
    assert len(report["cases"]) == 2
    assert report["cases"][0]["id"] == "eval-001"


def test_run_eval_handles_mixed_verdicts(monkeypatch, classifier_models):
    _install_fake_complete(
        monkeypatch,
        [
            {"verdict": "correct", "reasoning": "matches"},
            {"verdict": "partially_correct", "reasoning": "close but missing detail"},
        ],
    )

    report = run_eval(EVAL_CASES, classifier_models)

    assert report["root_cause_exact_pass_rate"] == 0.5
    assert report["root_cause_mean_score"] == 0.75


def test_run_eval_incorrect_verdict_scores_zero(monkeypatch, classifier_models):
    _install_fake_complete(
        monkeypatch,
        [
            {"verdict": "incorrect", "reasoning": "wrong diagnosis"},
            {"verdict": "incorrect", "reasoning": "wrong diagnosis"},
        ],
    )

    report = run_eval(EVAL_CASES, classifier_models)

    assert report["root_cause_exact_pass_rate"] == 0.0
    assert report["root_cause_mean_score"] == 0.0


def test_run_eval_detects_category_mismatch(monkeypatch, classifier_models):
    _install_fake_complete(monkeypatch, [{"verdict": "correct", "reasoning": "matches"}])
    mismatched_case = [{**EVAL_CASES[0], "expected_category": "auth"}]

    report = run_eval(mismatched_case, classifier_models)

    assert report["category_accuracy"] == 0.0
    assert report["cases"][0]["category_correct"] is False


# --- judge_root_cause() -------------------------------------------------------


def test_judge_root_cause_returns_parsed_verdict(monkeypatch):
    _install_fake_complete(monkeypatch, [{"verdict": "incorrect", "reasoning": "wrong diagnosis"}])

    result = eval_harness.judge_root_cause(
        "boom",
        "expected cause",
        [{"cause": "wrong cause", "confidence": "low", "how_to_confirm": "check logs"}],
    )

    assert result == {"verdict": "incorrect", "reasoning": "wrong diagnosis"}


def test_judge_root_cause_incomplete_response_raises(monkeypatch):
    def _complete(system, user_content, model, *, max_tokens=4096, tool=None, effort=None):
        return _fake_response(None, stop_reason="max_tokens")

    monkeypatch.setattr(llm, "complete", _complete)

    with pytest.raises(JudgeIncompleteResponse) as exc_info:
        eval_harness.judge_root_cause(
            "boom", "expected cause", [{"cause": "x", "confidence": "low", "how_to_confirm": "y"}]
        )

    assert exc_info.value.stop_reason == "max_tokens"
