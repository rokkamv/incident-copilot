# 0008: Stage 6 tool-use agent loop

**Status:** accepted

## Decision

Stage 6 (`src/copilot/agent.py`) is a genuine multi-turn, multi-tool, auto-tool-choice agent
loop - not another forced single tool call. The model is offered three tools at once
(`classify_incident`, `search_similar_incidents`, `record_triage_findings`) with free choice,
and decides itself which to call, in what order, how many times, across multiple turns, until
it calls `record_triage_findings` to finish. This is built on new infrastructure in
`copilot.llm` (`complete_turn()`, `ToolCall`, `AgentTurn`, `append_assistant_turn()`,
`append_tool_result()`) alongside the existing `complete()`, which stays untouched. Both
Anthropic and OpenAI are supported, consistent with every stage so far. Stage 6 ships both a
CLI (`stages/06_agent_triage.py`) and a new endpoint (`POST /triage/agent`) - `/triage`'s own
route, model, and behavior are completely untouched.

## Why

- **Multi-turn over single-shot**: Stage 3 already proved forced single-tool-use
  (`TRIAGE_TOOL`). The model freely deciding which/whether/how-many tools to call, reacting to
  each result before deciding what to do next, is the actual subject of this stage per the
  README ("Tools and agents") and can't be taught by a forced call - there would be nothing
  for the model to decide.
- **Both providers**: continues decision 0005's standing policy. Verified directly against
  the installed SDKs (not recalled) rather than assumed: Anthropic 1.12.1's
  `tool_choice={"type": "auto", "disable_parallel_tool_use": True}` and OpenAI 3.35.1's
  `tool_choice="auto"` / `parallel_tool_calls=False` both support auto choice, parallel-call
  suppression, and tool-result feedback with near-identical semantics, despite very different
  wire shapes (Anthropic re-sends `response.content` verbatim as the next assistant turn;
  OpenAI's `tool_calls` message field has to be reconstructed by hand).
- **Exactly one tool call per model turn**: both SDKs default to allowing parallel tool calls;
  this stage disables that explicitly. Keeps `AgentTurn.tool_call` a single value instead of a
  list, so the loop in `agent.py` stays a plain step-by-step state machine. The cost is real -
  the model can't classify and search in the same turn, just across two consecutive ones - but
  not having to fan out/fan in per turn for a 3-tool stage is worth that cost.
- **Read-only tools only**: `classify_incident` and `search_similar_incidents` wrap Stage 2/4's
  existing deterministic functions directly, so every tool call this loop makes is fully
  reproducible outside the loop too. No fictitious "action" tool (e.g. paging on-call) was
  added - this stage is scoped to orchestration/decision-making, not simulated side effects.
  `classify_incident` deliberately takes no arguments (the real `error_text` is always bound
  server-side) so the model can't feed the classifier a paraphrased or hallucinated string;
  `search_similar_incidents` exposes `query`/`k` because letting the model choose how to search
  is the actual point of offering it that tool.
- **A stray text-only turn gets nudged, not failed**: a model occasionally responding with
  plain text instead of a tool call (e.g. thinking out loud) isn't treated as an error - the
  loop appends a one-line reminder and continues, still bounded by `max_turns`. Failing
  immediately on one harmless aside would be too brittle for a conversational loop.
- **A new endpoint now, with nullable `category`/`team`**: an explicit product decision, not a
  side effect of this exercise. `AgentTriageResponse.category`/`.team` are `str | None`, unlike
  `TriageResponse`'s always-present `str` - the API surface itself makes visible that this
  endpoint's guarantees are weaker than `/triage`'s.
- **The core lesson - reliability vs autonomy**: Stage 3's `category`/`team` and retrieval
  usage are unconditional and deterministic; the classifier and (when configured) retrieval
  are always called, in a fixed order. Stage 6's are *optional and model-chosen* - the model
  might skip classification entirely, search zero-to-many times with queries it invents, or
  occasionally wander before finishing. Autonomy is gained; a guarantee is spent. This is the
  trade-off the stage exists to teach, not a defect to paper over.

## Rejected

- Forcing `classify_incident` as a mandatory first call: would guarantee `category`/`team`
  again, but defeats the lesson - there would be nothing left for the model to decide.
- A fourth, mocked "action" tool (e.g. simulated paging/ticketing): out of scope - this stage
  is about orchestration and decision-making over tools that already exist and are read-only,
  not about simulating new side effects.
- Merging `complete()` and `complete_turn()` into one generalized function: forced-single-tool
  with a plain string vs. auto-choice-multi-tool with full conversation history differ enough
  that one parameterized function would obscure both, for marginal line-count savings. The
  piece that genuinely is identical (SDK-exception-to-`LLMError` translation) was factored out
  into shared `_call_anthropic`/`_call_openai` helpers instead.
- Allowing parallel tool calls: real added complexity (`AgentTurn.tool_call` becoming a list,
  the loop needing to fan out/fan in per turn) for a stage with only three tools.
- Failing immediately on a stray text-only turn: too brittle - one harmless aside from the
  model shouldn't end the whole run when the turn budget already bounds how long it can wander.
- CLI-only, no endpoint: considered (mirroring Stage 2's own CLI-only precedent before Stage 3
  wired it into the API), but decided against - the endpoint ships now, with its nullable
  fields as the explicit, visible signal of this stage's trade-off, rather than deferring that
  product decision.
