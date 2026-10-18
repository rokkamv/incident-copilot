# 0005: Multi-provider LLM abstraction

**Status:** accepted

## Decision

A new module, `src/copilot/llm.py`, normalizes LLM calls across providers (Anthropic and
OpenAI) behind a single `complete()` function. Stage 1 (`stages/01_first_call.py`) and
Stage 3 (`src/copilot/triage.py`) both call through it instead of constructing a provider
SDK client directly. `LLM_PROVIDER`, `LLM_MODEL`, and `LLM_API_KEY` are all mandatory
settings with no default, validated at process startup by pydantic itself (no default
provider, no default model, for either provider). Pricing moves out of source code into a
configurable `data/pricing.json`, loaded from a path itself set via `PRICING_FILE`.

## Why

- **Extends, doesn't retract, `0001-stack.md`:** `0001` chose the Anthropic SDK as the
  project's LLM integration. This ADR adds OpenAI as a second supported provider without
  reversing that choice - Anthropic is still fully supported, just no longer assumed.
- **No default provider or model, for either provider:** earlier work on this project
  researched "current" OpenAI model names and got back fabricated-looking results (names
  like "gpt-6-astra" that don't match any real OpenAI naming convention). That's concrete
  proof an AI assistant's recalled "current best model" isn't trustworthy enough to bake
  into code as a default - for *any* provider, not just the new one. `LLM_PROVIDER`/
  `LLM_MODEL`/`LLM_API_KEY` are required pydantic-settings fields with no `=` default, so
  `Settings()` raises a clear `ValidationError` at import time if any is missing - the app
  fails at startup, not on the first request.
- **`COPILOT_MODEL` -> `LLM_MODEL`, `ANTHROPIC_API_KEY`/`OPENAI_API_KEY` -> `LLM_API_KEY`:**
  since the provider is already one switch (`LLM_PROVIDER`), a second provider-specific
  model field or API-key field is redundant. One field each, passed explicitly to whichever
  SDK client gets constructed (`anthropic.Anthropic(api_key=...)` /
  `openai.OpenAI(api_key=...)`), rather than relying on each SDK's own same-named
  environment-variable auto-detection.
- **Chat Completions over the Responses API (OpenAI):** Chat Completions is explicitly
  "supported indefinitely" by OpenAI and maps much more directly onto Anthropic's Messages
  API (one call, one response, `tool_calls` vs `tool_use`) than the newer, event-oriented
  Responses API - this keeps the adapter code small.
- **A provider-neutral `ToolSpec`:** `triage.py`'s tool definition is plain
  `name`/`description`/`schema`, with no provider-specific wrapping. Each adapter
  (`_to_anthropic_tool`/`_to_openai_tool`) translates it into that SDK's wire format, so
  `triage.py` never needs to know which provider is active.
- **Refusal/truncation/usage normalization:** the two SDKs use different vocabulary for the
  same concepts, collapsed into `LLMResponse`:

  | Concept | Anthropic | OpenAI | `LLMResponse` field |
  |---|---|---|---|
  | Refusal | `stop_reason == "refusal"` | `message.refusal is not None` | `refused` |
  | Truncation | `stop_reason == "max_tokens"` | `finish_reason == "length"` | `truncated` |
  | Tool call succeeded | `stop_reason == "tool_use"`, `content` block `.input` (dict) | `finish_reason == "tool_calls"`, `tool_calls[0].function.arguments` (JSON **string**) | `tool_input` (always a parsed dict - OpenAI's string is `json.loads`'d inside the adapter) |
  | Input/output tokens | `usage.input_tokens`/`.output_tokens` | `usage.prompt_tokens`/`.completion_tokens` | `input_tokens`/`output_tokens` |

- **`LLMMalformedToolCallError`:** OpenAI's tool-call arguments are a JSON string the model
  could (rarely) get wrong even under a well-formed schema; this is a real failure mode
  Anthropic's SDK doesn't have (it hands back an already-parsed dict), so it gets its own
  normalized exception rather than being silently swallowed or crashing with a raw
  `json.JSONDecodeError`.
- **Pricing moves to `data/pricing.json`, not fetched, not hardcoded in Python:** neither
  provider exposes a queryable pricing API - both publish prices only on static docs pages.
  A third-party aggregator (e.g. a community-maintained pricing JSON) exists but just
  relocates the trust problem onto a different party and adds a network dependency. Keeping
  pricing in a small, explicitly-sourced data file (not a Python dict literal) means anyone
  can update it without touching code, and `cost.load_prices()` prints the file path it
  loaded from at startup, so the source is never silently assumed. OpenAI has no entries
  yet and won't until real, verified numbers are added - `estimate_cost` returns `None`
  (logged, not computed) for any provider/model without a pricing entry, rather than
  guessing. `assert_known_model` only fails fast for a provider that *has* a pricing table
  but is missing the requested model (catches a likely typo); a provider with no table at
  all (e.g. `"openai"` today) is a deliberate no-op, since there's nothing to validate and
  it must not block using the provider.
- **Optional `LLM_BASE_URL`:** both SDKs accept a `base_url` override, which matters beyond
  Anthropic/OpenAI themselves - e.g. Groq serves OpenAI-compatible open-weight models
  (such as `gpt-oss-120b`) at `https://api.groq.com/openai/v1` rather than OpenAI's own
  endpoint, with a Groq-issued key (not an OpenAI key) in `LLM_API_KEY`. Left unset, each
  client uses its SDK's normal default endpoint - this was discovered as a real need
  during manual verification, not speculated in advance.
- **Cross-provider model-resolution guard:** with one `LLM_MODEL` field (not two
  provider-specific ones), overriding `--provider` at the CLI without also passing
  `--model` would otherwise silently send a model name configured for a *different*
  provider. `llm.resolve_model()` rejects this combination with a clear error instead.

## Rejected

- A default provider (e.g. falling back to Anthropic) or a default model for either
  provider: the explicit point of this change is "don't assume, be explicit" - a default
  undermines that for the sake of convenience.
- The OpenAI Responses API: newer and recommended by OpenAI for new projects generally, but
  its shape diverges further from Anthropic's Messages API (event-oriented output, flatter
  tool definitions) and would have meant more normalization code for no benefit here.
  Chat Completions' "indefinitely supported" status removes the usual pressure to adopt the
  newer API early.
- A shared/provider-namespaced `PRICES` structure that includes an empty `"openai"` key:
  rejected because an empty namespace invites someone to later paste in a guessed number,
  exactly what this ADR's pricing policy forbids. Absence of the key is the explicit
  decision.
- Fetching pricing from a third-party aggregator at runtime: relocates trust rather than
  resolving it, and adds a network dependency and caching/refresh complexity this stage
  doesn't need.
- `assert_known_model` treating an entirely-missing provider table the same as a
  missing-model-within-a-table: would make the newly-added OpenAI provider unusable, since
  it has no pricing table at all today.
