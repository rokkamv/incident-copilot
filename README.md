# Incident Copilot

An AI assistant that helps on-call engineers triage production incidents. Paste an error or log snippet, and it finds similar past incidents and runbooks, suggests a likely root cause with citations, and drafts next steps.

Built stage by stage as a hands-on way to learn AI engineering. All data is synthetic.

## Stages

| Stage | Topic | Status |
| --- | --- | --- |
| 1 | How LLMs work, first API call | started |
| 2 | Classic ML baseline classifier | |
| 3 | Prompting and structured output (`POST /triage`) | |
| 4 | Embeddings and retrieval (RAG) with pgvector | |
| 5 | Evaluation, run in CI | |
| 6 | Tools and agents | |
| 7 | Production readiness: redaction, guardrails, cost | |

Decisions are recorded in [docs/decisions](docs/decisions).

## Setup

```bash
uv sync                      # install dependencies
cp .env.example .env         # then add your ANTHROPIC_API_KEY
uv run pytest                # run tests
uv run uvicorn copilot.api:app --reload   # start the API at http://localhost:8000
```

Later stages need Postgres and Redis: `docker compose up -d`.

## Stage 1

```bash
uv run --env-file .env python stages/01_first_call.py "ERROR PaymentService: duplicate transaction id TXN-123"
```
