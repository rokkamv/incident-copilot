# 0002: Stage 1 defaults and error handling

**Status:** accepted

## Decision

Stage 1 (`stages/01_first_call.py`) defaults to `claude-opus-5-5` at `effort="low"`, exposes
`--model`/`--effort` flags to vary both without editing code, validates the model against the
known cost table before making any API call, and turns the Anthropic SDK's typed exceptions
into one-line, actionable messages instead of raw tracebacks.

## Why

- **Default model (`claude-opus-5-5`), overridable via `--model`/`COPILOT_MODEL`:** the stage
  defaults to the most capable current model so a learner's first impression of LLM output
  quality isn't hobbled by a cheap model. The `--model` flag exists specifically so the same
  prompt can be re-run against Sonnet or Haiku to compare cost and quality directly, which is the
  point of the stage.
- **Default effort (`low`), with a `--effort` flag:** cheap and fast by default, since this is a
  script a reader re-runs many times while learning. The inline code comment already invited the
  reader to "try effort=medium and compare"; `--effort` makes that a flag instead of a code edit.
- **Cost tracking via a hard-coded `PRICES` dict, validated fail-fast:** chosen for transparency —
  a reader can see exactly what a call costs in one small file — over querying a live pricing
  API. `assert_known_model` turns a stale or mistyped model string into an upfront error before
  any request is sent, instead of a `KeyError` raised only after a paid call already succeeded.
- **Error handling via typed SDK exceptions, no custom retry loop:** the SDK already retries
  429/5xx with backoff internally. Stage 1 only needs to turn an unhandled exception into a
  readable message for `AuthenticationError`, `RateLimitError`, `APIStatusError`, and
  `APIConnectionError` — not reimplement retry logic.
- **Warn, don't silently truncate, on `stop_reason == "max_tokens"`:** printing a cut-off response
  as if it were complete would hide a real failure mode from the learner.

## Rejected

- A general-purpose cost/pricing client: over-engineering for a stage about a first API call.
- Silently ignoring `max_tokens` truncation: hides a real failure mode instead of teaching it.
- A custom retry/backoff loop: duplicates behavior the SDK already provides by default.
