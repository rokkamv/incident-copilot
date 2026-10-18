# 0004: Stage 3 triage endpoint architecture

**Status:** accepted

## Decision

`POST /triage` (`src/copilot/api.py`) combines Stage 2's classic ML classifier (predicting
`category`/`team`) with a separate Anthropic tool-use call (predicting `root_causes`/
`next_steps`), training the classifier once at FastAPI startup via a `lifespan` context
manager rather than per request. The orchestration lives in a plain, FastAPI-independent
function, `copilot.triage.triage()`.

## Why

- **Hybrid classifier + LLM, not LLM-only:** category/team is a fixed, small label space -
  the Stage 2 classifier already handles it for free and improves with more labeled data.
  Root cause/next-steps is open-ended reasoning where the LLM earns its cost. The LLM is
  given the classifier's prediction as context rather than asked to re-derive it, so the two
  halves stay coherent instead of potentially disagreeing with each other.
- **Tool-use with a forced schema, not prompt-and-hope-for-JSON:** `tool_choice` forces the
  `record_triage_findings` tool, so the model's structured input comes back as a parsed dict
  straight from the SDK - no manual JSON parsing or retry-on-malformed-JSON logic needed. (No
  `strict` flag is set; the schema is enforced by forcing the tool call itself, not a separate
  strict-mode toggle.)
- **Startup-time training via `lifespan`, cached on `app.state`:** the synthetic dataset
  doesn't change between requests, so retraining on every call is pure wasted latency.
  Caching the fitted pipelines on `app.state` keeps `/triage` requests fast.
- **`triage()` as a plain function, not reaching into `app.state`:** keeps the orchestration
  logic unit-testable without spinning up FastAPI/`TestClient`, and keeps `api.py` as a thin
  HTTP-to-exception-code translation layer. Anthropic's typed exceptions propagate out of
  `triage()` uncaught; only `api.py` knows how to turn them into HTTP status codes.
- **`error_text` bounded with `max_length=4000`:** an unbounded request body would let a
  caller send an arbitrarily large payload straight into a paid LLM call and the classifier's
  vectorizer - bounding it rejects oversized input at validation time (422), before any cost
  is incurred.

## Rejected

- LLM-only prediction of category/team/root-causes/next-steps in one call: duplicates the
  classifier's job with a slower, costlier, less consistent approach for a label space it
  already solves well.
- Per-request classifier retraining: the training data doesn't change between requests at
  this stage, so retraining on every call is wasted latency with no benefit.
- A feedback/correction endpoint that grows `data/incidents/` from real triage outcomes:
  explicitly deferred to a later stage (likely Stage 7, production readiness). `load_incidents()`/
  `train()` remain the only data-loading/training entry points so adding this later doesn't
  require reworking Stage 3's code.
- Prompt-based JSON output without tool-use: no schema guarantee, requires defensive parsing
  and retry logic the SDK's forced tool-use already avoids.
- Unbounded `error_text`: a 255-character cap (the TODO's original suggestion) is too tight
  for a real stack trace or multi-line log excerpt; 4000 characters was chosen as generous
  enough for realistic input while still bounding LLM cost/latency.

---

**Extended by [0005](0005-multi-provider-llm-abstraction.md):** `triage()` now calls through
`copilot.llm.complete()` instead of `anthropic.Anthropic()` directly, and `/triage`'s
exception-to-HTTP mapping catches normalized `llm.LLM*Error` types instead of `anthropic.*`
exceptions (same status codes).
