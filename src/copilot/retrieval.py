"""Stage 4: embed incidents, store them in pgvector, retrieve similar ones.

Combines copilot.embeddings (provider-neutral embedding calls) with copilot.store
(pgvector persistence), the same way copilot.triage combines copilot.classifier
with copilot.llm.
"""

from copilot import embeddings, store


def embed_and_store(incidents: list[dict]) -> None:
    """Embed and upsert every incident. Idempotent - safe to re-run.

    One function for both the initial 50-incident backfill and any incident
    added later (e.g. Stage 7's feedback loop), so neither needs its own path.
    """
    if not incidents:
        return
    vectors = embeddings.embed(
        [incident["error_text"] for incident in incidents], input_type="document"
    )
    with store.get_connection() as conn:
        store.ensure_schema(conn)
        for incident, vector in zip(incidents, vectors, strict=True):
            store.upsert_incident(
                conn,
                incident["id"],
                incident["error_text"],
                incident["category"],
                incident["team"],
                vector,
            )


def retrieve_similar(error_text: str, k: int = 3) -> list[dict]:
    """Return the k most similar past incidents to `error_text`, nearest first."""
    vector = embeddings.embed([error_text], input_type="query")[0]
    with store.get_connection() as conn:
        return store.search_similar(conn, vector, k=k)
