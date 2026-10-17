# Incident Copilot

An AI assistant that helps on-call engineers triage production incidents. Paste an error or log snippet, and it finds similar past incidents and runbooks, suggests a likely root cause with citations, and drafts next steps.

Built stage by stage as a hands-on way to learn AI engineering. All data is synthetic.

## Stages

| Stage | Topic | Status |
| --- | --- | --- |
| 1 | How LLMs work, first API call | started |
| 2 | Classic ML baseline classifier | started |
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

# compare a higher effort level or a cheaper model, without editing code
uv run --env-file .env python stages/01_first_call.py --effort medium "ERROR PaymentService: duplicate transaction id TXN-123"
uv run --env-file .env python stages/01_first_call.py --model claude-sonnet-5-5 "ERROR PaymentService: duplicate transaction id TXN-123"
```

## Stage 2

```bash
# evaluate: 5-fold cross-validated accuracy on the 50 synthetic incidents
uv run python stages/02_classify_incident.py

# predict category + team for a new error message
uv run python stages/02_classify_incident.py "ERROR PaymentGateway: timeout calling StripeAPI charge endpoint"
```
