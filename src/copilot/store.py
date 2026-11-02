"""pgvector persistence for Stage 4's incident embeddings.

Plain psycopg, no ORM - matches the rest of the project's lightweight style
(copilot.incidents reads JSON directly, no models layer either).
"""

import psycopg
from pgvector.psycopg import register_vector
from psycopg import sql

from copilot.config import settings


def get_connection() -> psycopg.Connection:
    conn = psycopg.connect(settings.database_url)
    # The vector extension must exist before register_vector can look up its type
    # OID - on a fresh database this is the first thing to touch it, before
    # ensure_schema's CREATE TABLE ever runs.
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.commit()
    register_vector(conn)
    return conn


def ensure_schema(conn: psycopg.Connection) -> None:
    # CREATE TABLE IF NOT EXISTS silently no-ops against a table that already
    # exists with a different VECTOR(n) - e.g. EMBEDDING_DIMENSIONS changed, or
    # (as happened once) a test run left the table at a different dimension
    # than .env's real config. Left unchecked, that surfaces much later as a
    # cryptic "expected N dimensions" error deep inside upsert_incident, so fail
    # fast here instead, with a message that says what to do about it.
    existing = conn.execute(
        """
        SELECT atttypmod FROM pg_attribute
        WHERE attrelid = to_regclass('incident_embeddings') AND attname = 'embedding'
        """
    ).fetchone()
    if existing is not None and existing[0] != settings.embedding_dimensions:
        raise ValueError(
            f"incident_embeddings already exists with {existing[0]} dimensions, but "
            f"EMBEDDING_DIMENSIONS is {settings.embedding_dimensions}. Drop the table "
            f"(DROP TABLE incident_embeddings;) and re-run to re-embed at the new dimension."
        )

    # VECTOR(n)'s dimension is a type modifier, not a value - it can't go through
    # a %s parameter placeholder, so it's composed with sql.Literal instead of an
    # f-string (which psycopg's own query type rejects: an f-string is a plain
    # `str`, not the `LiteralString` its Query type requires, precisely to stop
    # unparameterized data from being interpolated into SQL).
    conn.execute(
        sql.SQL(
            """
            CREATE TABLE IF NOT EXISTS incident_embeddings (
                id TEXT PRIMARY KEY,
                error_text TEXT NOT NULL,
                category TEXT NOT NULL,
                team TEXT NOT NULL,
                embedding VECTOR({dim}) NOT NULL
            )
            """
        ).format(dim=sql.Literal(settings.embedding_dimensions))
    )
    conn.commit()


def upsert_incident(
    conn: psycopg.Connection,
    incident_id: str,
    error_text: str,
    category: str,
    team: str,
    embedding: list[float],
) -> None:
    conn.execute(
        """
        INSERT INTO incident_embeddings (id, error_text, category, team, embedding)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (id) DO UPDATE SET
            error_text = EXCLUDED.error_text,
            category = EXCLUDED.category,
            team = EXCLUDED.team,
            embedding = EXCLUDED.embedding
        """,
        (incident_id, error_text, category, team, embedding),
    )
    conn.commit()


def search_similar(conn: psycopg.Connection, embedding: list[float], k: int = 3) -> list[dict]:
    # The target column on an INSERT tells Postgres a parameter's type, so
    # upsert_incident's plain %s works - but <=> has no such target to infer
    # from, so without the explicit ::vector cast psycopg sends a bare Python
    # list as a float8[] array, which <=> isn't defined for.
    rows = conn.execute(
        """
        SELECT id, error_text, category, team, embedding <=> %s::vector AS distance
        FROM incident_embeddings
        ORDER BY distance
        LIMIT %s
        """,
        (embedding, k),
    ).fetchall()
    return [
        {"id": r[0], "error_text": r[1], "category": r[2], "team": r[3], "distance": r[4]}
        for r in rows
    ]
