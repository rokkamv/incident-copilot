"""Stage 4: embed incidents into pgvector, or retrieve similar past incidents.

Needs Postgres running (`docker compose up -d`) and EMBEDDING_PROVIDER/EMBEDDING_MODEL/
EMBEDDING_API_KEY set in .env - see decision 0006 for the provider choice (Voyage AI,
with Google AI Studio as a second provider).

Usage:
    uv run --env-file .env python stages/04_embed_and_retrieve.py
        # backfill mode: embeds and stores every incident in data/incidents/

    uv run --env-file .env python stages/04_embed_and_retrieve.py "ERROR PaymentGateway: timeout"
        # retrieve mode: prints the k most similar stored incidents

    uv run --env-file .env python stages/04_embed_and_retrieve.py --k 5 "ERROR ..."
        # retrieve more matches
"""

import argparse
import sys
from pathlib import Path

from copilot import embeddings, retrieval
from copilot.incidents import load_incidents


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "error_message",
        nargs="?",
        default=None,
        help="an error/log line to find similar past incidents for; omit to backfill instead",
    )
    parser.add_argument(
        "--incidents-dir",
        type=Path,
        default=None,
        help="directory of incident JSON files to backfill (default: data/incidents/)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="number of similar incidents to retrieve (default: 3)",
    )
    return parser


def run(error_message: str | None, incidents_dir: Path | None, k: int) -> int:
    try:
        if error_message is None:
            incidents = load_incidents(incidents_dir)
            retrieval.embed_and_store(incidents)
            print(f"embedded and stored {len(incidents)} incidents")
            return 0

        matches = retrieval.retrieve_similar(error_message, k=k)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except embeddings.EmbeddingAuthenticationError as exc:
        print(f"error: embedding provider rejected the API key: {exc}", file=sys.stderr)
        return 1
    except embeddings.EmbeddingRateLimitError as exc:
        print(f"error: rate limited by the embedding provider: {exc}", file=sys.stderr)
        return 1
    except embeddings.EmbeddingConnectionError as exc:
        print(f"error: could not reach the embedding provider (network issue): {exc}", file=sys.stderr)
        return 1
    except embeddings.EmbeddingAPIStatusError as exc:
        print(f"error: embedding provider returned {exc.status_code}: {exc}", file=sys.stderr)
        return 1

    if not matches:
        print("no similar incidents found - has data/incidents/ been embedded yet?")
        return 0

    for match in matches:
        print(f"[{match['distance']:.4f}] ({match['category']}/{match['team']}) {match['id']}")
        print(f"    {match['error_text']}")
    return 0


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    sys.exit(run(args.error_message, args.incidents_dir, args.k))


if __name__ == "__main__":
    main()
