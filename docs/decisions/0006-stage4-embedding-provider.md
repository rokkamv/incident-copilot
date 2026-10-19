# 0006: Stage 4 embedding provider

**Status:** accepted

## Decision

Stage 4 (embeddings/RAG) embeds runbooks and past incidents through a provider-neutral
`embed()` — mirroring Stage 3's `copilot.llm.complete()` pattern — with **Voyage AI**
(`voyage-4-lite`/`voyage-4`) as the primary provider and **Google AI Studio**
(`gemini-embedding-001`) as a second provider. One `embed_and_store()` function handles both
the initial 50-incident backfill and any incident added later, so it needs no rework when
Stage 7's feedback loop starts growing `data/incidents/`. Retrieved incidents are passed to
Stage 3's `triage()` as additional LLM context; they augment the Stage 2 classifier's
`category`/`team` prediction, not replace it.

## Why

- **A separate embedding provider is unavoidable:** Anthropic doesn't serve an embedding
  model, so this is new infrastructure regardless of which hosted provider is picked.
- **Voyage as primary:** it's Anthropic's own recommended embedding partner, cheapest of the
  providers evaluated (`voyage-4-lite` at $0.02/1M tokens), 32K token context, and a
  selectable output dimension (256/512/1024/2048 via Matryoshka) that doubles as the scale
  lever — start small for the 50-row dataset, grow the dimension later without a provider
  change. The first 200M tokens are free per account, so this stage costs $0 to build and
  test.
- **Google AI Studio (`gemini-embedding-001`) as the second provider:** also free-tier-first
  (zero cost to prove out both adapters), and pairing Anthropic + Google is a cleaner
  two-provider story than reaching for OpenAI again, which already plays "provider #2" in
  Stage 3's `copilot.llm`. Known constraint: 2048-token max input, tighter than Voyage's 32K —
  fine for the current `error_text` (capped at 4000 chars, ~800-1000 tokens) but worth
  revisiting if full runbook documents are embedded later, not just short error strings.
- **`embed_and_store()` as one reusable function, not a one-off backfill script:** the same
  path needs to run for the 50-incident seed now and for any new incident Stage 7 later adds
  via a feedback loop (explicitly deferred in decision 0004) — building it provider-neutral
  and dataset-size-agnostic from the start avoids a rewrite when that happens.
- **Augments, not replaces, the Stage 2 classifier:** `category`/`team` is a small, fixed
  label space the classifier already handles well; retrieval's job is grounding root-cause/
  next-steps reasoning in similar past incidents (with citations), not re-deciding category
  or team.

## Rejected

- **OpenAI (`text-embedding-3-small`/`-large`) as the embedding provider:** cheap and
  capable, but the project's `openai` provider slot (`.env.example`'s `LLM_BASE_URL`) is
  currently wired to Groq's OpenAI-compatible chat endpoint for Stage 3, and Groq doesn't
  serve an embeddings endpoint — reusing that slot would need a second, separate OpenAI
  credential anyway, losing the "already have the key" advantage. It would also make two
  stages both reach for OpenAI as "provider #2," less illustrative than pairing with Google.
- **Jina AI:** newest-generation pricing wasn't confirmable from public docs at decision
  time (pay-per-token, no published rate) — parked as a possible future swap-in, given its
  REST-only (no dedicated SDK) integration shape, once pricing is confirmed directly.
- **Local/open-source embeddings (e.g. `sentence-transformers`):** zero cost and no API key,
  but skips practicing the multi-provider pattern this project deliberately exercises (same
  spirit as decision 0005), and the hosted options' free tiers already make this stage free
  at this dataset size anyway.
- **Voyage-only (single provider):** rejected to keep proving the provider-neutral
  abstraction works against two real backends, not just one, consistent with Stage 3.
