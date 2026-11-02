"""Point Postgres-backed tests at a dedicated database, never the real one.

tests/test_store.py and tests/test_retrieval.py drop and recreate
incident_embeddings per test. Running that against settings.database_url
directly already corrupted a real local backfill once (a test-dimension table
got left behind, then blocked - and would eventually have been silently
overwritten by - a real `stages/04_embed_and_retrieve.py` run). A separate
database removes the collision entirely, rather than just tidying up after it.
"""

import psycopg
from psycopg import sql

from copilot.config import settings

_TEST_DB_NAME = "incident_copilot_test"


def _with_db_name(url: str, db_name: str) -> str:
    base, _, _ = url.rpartition("/")
    return f"{base}/{db_name}"


def _ensure_test_database_exists(real_url: str) -> None:
    admin_url = _with_db_name(real_url, "postgres")
    with psycopg.connect(admin_url, autocommit=True) as conn:
        exists = conn.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s", (_TEST_DB_NAME,)
        ).fetchone()
        if not exists:
            conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(_TEST_DB_NAME)))


_ensure_test_database_exists(settings.database_url)
settings.database_url = _with_db_name(settings.database_url, _TEST_DB_NAME)