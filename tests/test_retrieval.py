"""retrieval.py against real Postgres (see test_store.py), with embeddings.embed faked -
only the external embedding API call is mocked, matching how test_triage.py patches
triage.llm.complete and lets everything else (the classifier) run for real.
"""

import pytest

from copilot import retrieval, store
from copilot.config import settings


@pytest.fixture(autouse=True)
def _fixed_dimension(monkeypatch):
    monkeypatch.setattr(settings, "embedding_dimensions", 3)


@pytest.fixture
def clean_table():
    conn = store.get_connection()
    conn.execute("DROP TABLE IF EXISTS incident_embeddings")
    conn.commit()
    store.ensure_schema(conn)
    yield
    conn.close()


def _install_fake_embed(monkeypatch, vectors_by_text: dict[str, list[float]]):
    def _embed(texts, *, input_type):
        return [vectors_by_text[text] for text in texts]

    monkeypatch.setattr(retrieval.embeddings, "embed", _embed)


def test_embed_and_store_backfills_all_incidents(monkeypatch, clean_table):
    incidents = [
        {"id": "a", "error_text": "db error", "category": "database", "team": "database-eng"},
        {"id": "b", "error_text": "net error", "category": "network", "team": "networking"},
    ]
    _install_fake_embed(
        monkeypatch, {"db error": [1.0, 0.0, 0.0], "net error": [0.0, 1.0, 0.0]}
    )

    retrieval.embed_and_store(incidents)

    conn = store.get_connection()
    rows = conn.execute("SELECT id FROM incident_embeddings ORDER BY id").fetchall()
    conn.close()
    assert [r[0] for r in rows] == ["a", "b"]


def test_embed_and_store_is_a_noop_for_empty_list(monkeypatch, clean_table):
    calls = []
    monkeypatch.setattr(
        retrieval.embeddings, "embed", lambda texts, **kw: calls.append(texts) or []
    )

    retrieval.embed_and_store([])

    assert calls == []


def test_retrieve_similar_returns_nearest_first(monkeypatch, clean_table):
    incidents = [
        {"id": "a", "error_text": "db error", "category": "database", "team": "database-eng"},
        {"id": "b", "error_text": "net error", "category": "network", "team": "networking"},
    ]
    _install_fake_embed(
        monkeypatch,
        {
            "db error": [1.0, 0.0, 0.0],
            "net error": [0.0, 1.0, 0.0],
            "query text": [0.9, 0.1, 0.0],
        },
    )
    retrieval.embed_and_store(incidents)

    results = retrieval.retrieve_similar("query text", k=2)

    assert [r["id"] for r in results] == ["a", "b"]
