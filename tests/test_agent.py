import pytest

from copilot import agent, classifier, llm

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

FINDINGS = {
    "root_causes": [
        {"cause": "deadlock on database table", "confidence": "high", "how_to_confirm": "check db logs"}
    ],
    "next_steps": ["Restart the stuck transaction", "Alert database-eng"],
}


@pytest.fixture
def classifier_models():
    return classifier.train(CLASSIFIER_INCIDENTS)


def _turn(tool_name=None, tool_input=None, *, tool_id="t1", text=None, refused=False, truncated=False):
    tool_call = llm.ToolCall(id=tool_id, name=tool_name, input=tool_input or {}) if tool_name else None
    stop_reason = "tool_use" if tool_call else ("refusal" if refused else "max_tokens" if truncated else "end_turn")
    return llm.AgentTurn(
        text=text,
        tool_call=tool_call,
        refused=refused,
        truncated=truncated,
        input_tokens=10,
        output_tokens=5,
        stop_reason=stop_reason,
        provider_message={"role": "assistant", "content": text or ""},
    )


def _install_fake_complete_turn(monkeypatch, turns):
    """Scripts a sequence of AgentTurns, one per call. Records the `messages` seen
    on each call (a snapshot, since append_* mutate the list in place afterward)."""
    turns_iter = iter(turns)
    captured_messages = []

    def _complete_turn(system, messages, model, tools, *, max_tokens=4096, effort=None):
        captured_messages.append(list(messages))
        return next(turns_iter)

    monkeypatch.setattr(agent.llm, "complete_turn", _complete_turn)
    return captured_messages


# --- the happy path: classify, search, finish --------------------------------


def test_classify_then_search_then_record(monkeypatch, classifier_models):
    monkeypatch.setattr(agent.settings, "embedding_provider", "voyage")
    monkeypatch.setattr(
        agent.retrieval,
        "retrieve_similar",
        lambda query, k=3: [
            {"id": "sim-1", "error_text": "similar db issue", "category": "database", "team": "database-eng", "distance": 0.1}
        ],
    )
    _install_fake_complete_turn(
        monkeypatch,
        [
            _turn("classify_incident", {}),
            _turn("search_similar_incidents", {"query": "db timeout", "k": 3}),
            _turn("record_triage_findings", FINDINGS),
        ],
    )

    result = agent.triage("connection pool exhausted on orders database", classifier_models)

    assert result["category"] == "database"
    assert result["team"] == "database-eng"
    assert result["root_causes"] == FINDINGS["root_causes"]
    assert result["next_steps"] == FINDINGS["next_steps"]
    assert result["similar_incidents"] == [
        {"id": "sim-1", "error_text": "similar db issue", "category": "database", "team": "database-eng", "distance": 0.1}
    ]
    assert [c["tool"] for c in result["tool_calls"]] == [
        "classify_incident",
        "search_similar_incidents",
        "record_triage_findings",
    ]


def test_classify_never_called_leaves_category_team_none(monkeypatch, classifier_models):
    monkeypatch.setattr(agent.settings, "embedding_provider", None)
    _install_fake_complete_turn(monkeypatch, [_turn("record_triage_findings", FINDINGS)])

    result = agent.triage("connection pool exhausted on orders database", classifier_models)

    assert result["category"] is None
    assert result["team"] is None
    assert result["similar_incidents"] == []


def test_search_called_twice_dedups_by_id(monkeypatch, classifier_models):
    monkeypatch.setattr(agent.settings, "embedding_provider", "voyage")

    def _fake_retrieve(query, k=3):
        if query == "q1":
            return [{"id": "x", "error_text": "x", "category": "database", "team": "database-eng", "distance": 0.1},
                    {"id": "y", "error_text": "y", "category": "database", "team": "database-eng", "distance": 0.2}]
        return [{"id": "y", "error_text": "y", "category": "database", "team": "database-eng", "distance": 0.15},
                {"id": "z", "error_text": "z", "category": "database", "team": "database-eng", "distance": 0.3}]

    monkeypatch.setattr(agent.retrieval, "retrieve_similar", _fake_retrieve)
    _install_fake_complete_turn(
        monkeypatch,
        [
            _turn("search_similar_incidents", {"query": "q1"}),
            _turn("search_similar_incidents", {"query": "q2"}),
            _turn("record_triage_findings", FINDINGS),
        ],
    )

    result = agent.triage("boom", classifier_models)

    assert [m["id"] for m in result["similar_incidents"]] == ["x", "y", "z"]


# --- refusal, stray text, max turns -------------------------------------------


def test_refusal_raises(monkeypatch, classifier_models):
    _install_fake_complete_turn(monkeypatch, [_turn(refused=True)])

    with pytest.raises(agent.ModelRefused):
        agent.triage("boom", classifier_models)


def test_stray_text_turn_is_nudged_then_continues(monkeypatch, classifier_models):
    captured_messages = _install_fake_complete_turn(
        monkeypatch,
        [
            _turn(text="let me think about this"),
            _turn("record_triage_findings", FINDINGS),
        ],
    )

    result = agent.triage("boom", classifier_models)

    assert result["root_causes"] == FINDINGS["root_causes"]
    # by the second call, the nudge message should have been appended
    assert captured_messages[1][-1]["content"] == (
        "Continue: call a tool, or call record_triage_findings to finish."
    )


def test_max_turns_exceeded_raises_with_partial_trace(monkeypatch, classifier_models):
    _install_fake_complete_turn(
        monkeypatch,
        [_turn(text="stalling") for _ in range(3)],
    )

    with pytest.raises(agent.AgentMaxTurnsExceeded) as exc_info:
        agent.triage("boom", classifier_models, max_turns=3)

    assert exc_info.value.max_turns == 3
    assert exc_info.value.tool_calls == []


# --- retrieval not configured, malformed finish -------------------------------


def test_search_without_embedding_provider_returns_not_configured_marker(monkeypatch, classifier_models):
    monkeypatch.setattr(agent.settings, "embedding_provider", None)
    _install_fake_complete_turn(
        monkeypatch,
        [
            _turn("search_similar_incidents", {"query": "q1"}),
            _turn("record_triage_findings", FINDINGS),
        ],
    )

    result = agent.triage("boom", classifier_models)

    assert result["similar_incidents"] == []
    assert result["tool_calls"][0]["result"] == {"error": "retrieval not configured", "results": []}


def test_malformed_record_triage_findings_raises_incomplete(monkeypatch, classifier_models):
    _install_fake_complete_turn(
        monkeypatch, [_turn("record_triage_findings", {"root_causes": FINDINGS["root_causes"]})]
    )

    with pytest.raises(agent.AgentTurnIncomplete):
        agent.triage("boom", classifier_models)
