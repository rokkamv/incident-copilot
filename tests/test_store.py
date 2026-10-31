"""Tests run against a real Postgres (see docker-compose.yml / CI's postgres service) -
pgvector's <=> operator isn't worth mocking when the real thing is one `docker compose
up -d` away.
"""

import pytest

from copilot import store
from copilot.config import settings


@pytest.fixture
def conn(monkeypatch):
    monkeypatch.setattr(settings, "embedding_dimensions", 3)
    connection = store.get_connection()
    connection.execute("DROP TABLE IF EXISTS incident_embeddings")
    connection.commit()
    store.ensure_schema(connection)
    yield connection
    connection.close()


def test_ensure_schema_is_idempotent(conn):
    store.ensure_schema(conn)  # re-running must not raise
    assert conn.execute("SELECT count(*) FROM incident_embeddings").fetchone()[0] == 0


def test_upsert_and_search_similar_returns_nearest_first(conn):
    store.upsert_incident(conn, "a", "db error", "database", "database-eng", [1.0, 0.0, 0.0])
    store.upsert_incident(conn, "b", "network error", "network", "networking", [0.0, 1.0, 0.0])
    store.upsert_incident(conn, "c", "auth error", "auth", "security", [0.0, 0.0, 1.0])

    results = store.search_similar(conn, [0.9, 0.1, 0.0], k=2)

    assert [r["id"] for r in results] == ["a", "b"]
    assert results[0]["distance"] < results[1]["distance"]


def test_search_similar_respects_k(conn):
    for i in range(5):
        vector = [0.0, 0.0, 0.0]
        vector[i % 3] = 1.0
        store.upsert_incident(conn, f"inc-{i}", f"error {i}", "database", "database-eng", vector)

    results = store.search_similar(conn, [1.0, 0.0, 0.0], k=2)

    assert len(results) == 2


def test_upsert_is_idempotent_on_id(conn):
    store.upsert_incident(conn, "a", "original text", "database", "database-eng", [1.0, 0.0, 0.0])
    store.upsert_incident(conn, "a", "updated text", "network", "networking", [0.0, 1.0, 0.0])

    rows = conn.execute("SELECT error_text, category, team FROM incident_embeddings").fetchall()

    assert len(rows) == 1
    assert rows[0] == ("updated text", "network", "networking")
