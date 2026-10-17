from copilot.classifier import build_pipeline, evaluate, predict, train

INCIDENTS = [
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


def test_train_and_predict_obvious_case():
    models = train(INCIDENTS)

    result = predict(models, "ERROR: deadlock on database table, transaction rolled back")

    assert result["category"] == "database"
    assert result["team"] == "database-eng"


def test_evaluate_returns_accuracy_metrics():
    scores = evaluate(INCIDENTS, n_splits=5)

    assert set(scores) == {"category_accuracy", "team_accuracy"}
    for value in scores.values():
        assert 0.0 <= value <= 1.0


def test_build_pipeline_returns_independent_instances():
    first = build_pipeline()
    second = build_pipeline()

    assert first is not second
    assert first.named_steps["tfidf"] is not second.named_steps["tfidf"]
