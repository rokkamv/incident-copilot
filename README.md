# Incident Copilot

An AI assistant that helps on-call engineers triage production incidents. Paste an error or log snippet, and it finds similar past incidents and runbooks, suggests a likely root cause with citations, and drafts next steps.

Built stage by stage as a hands-on way to learn AI engineering. All data is synthetic.

## Stages

| Stage | Topic | Status |
| --- | --- | --- |
| 1 | How LLMs work, first API call | started |
| 2 | Classic ML baseline classifier | started |
| 3 | Prompting and structured output (`POST /triage`) | started |
| 4 | Embeddings and retrieval (RAG) with pgvector | |
| 5 | Evaluation, run in CI | |
| 6 | Tools and agents | |
| 7 | Production readiness: redaction, guardrails, cost | |

Decisions are recorded in [docs/decisions](docs/decisions).

## Setup

```bash
uv sync                      # install dependencies
cp .env.example .env         # then set LLM_PROVIDER, LLM_MODEL and LLM_API_KEY
uv run pytest                # run tests
uv run uvicorn copilot.api:app --reload   # start the API at http://localhost:8000
```

`LLM_PROVIDER`, `LLM_MODEL` and `LLM_API_KEY` are all required - the app fails at startup
if any is missing. There is no default provider or model for either Anthropic or OpenAI;
pick one yourself and verify the model name is current against the provider's own docs.

Later stages need Postgres and Redis: `docker compose up -d`.

## Stage 1

```bash
uv run --env-file .env python stages/01_first_call.py "ERROR PaymentService: duplicate transaction id TXN-123"

# compare a higher effort level or a cheaper model, without editing code
uv run --env-file .env python stages/01_first_call.py --effort medium "ERROR PaymentService: duplicate transaction id TXN-123"
uv run --env-file .env python stages/01_first_call.py --model claude-sonnet-5-5 "ERROR PaymentService: duplicate transaction id TXN-123"

# use OpenAI instead of Anthropic for this one call (no default model - use one you've configured)
uv run --env-file .env python stages/01_first_call.py --provider openai --model <the model you've configured> "ERROR PaymentService: duplicate transaction id TXN-123"
```

## Stage 2

```bash
# evaluate: 5-fold cross-validated accuracy on the 50 synthetic incidents
uv run python stages/02_classify_incident.py

# predict category + team for a new error message
uv run python stages/02_classify_incident.py "ERROR PaymentGateway: timeout calling StripeAPI charge endpoint"
```

## Stage 3

Needs a real `LLM_API_KEY` in `.env` (unlike Stage 2, which needs no API key). `/triage`
honors the same `LLM_PROVIDER`/`LLM_MODEL` config as Stage 1 - switching providers means
editing `.env` and restarting the API (provider/model are validated once, at startup).
The request body also accepts an optional `"effort"` field (Anthropic only), falling
back to `LLM_EFFORT` from `.env`, then `"low"`.

```bash
uv run uvicorn copilot.api:app --reload   # starts the API, training the classifier once at startup

curl -X POST http://localhost:8000/triage \
  -H "Content-Type: application/json" \
  -d '{"error_text": "ERROR PaymentGateway: timeout calling StripeAPI charge endpoint"}'
```
