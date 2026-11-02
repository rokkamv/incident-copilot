# 0007: Stage 5 eval harness

**Status:** accepted

## Decision

Stage 5 (`stages/05_run_eval.py`) scores real `triage()` output against a hand-labeled golden
set (`data/evals/`, 15 synthetic cases, see `copilot.evals.load_eval_cases()`) two different
ways: `category`/`team` by exact match against the real Stage 2 classifier's prediction, and
`root_causes` quality by a second LLM call - an "LLM-as-judge" (`copilot.eval_harness.judge_root_cause()`)
comparing the candidate output against each case's human-written `expected_root_cause`, using
the same forced-tool-use pattern `copilot.triage.TRIAGE_TOOL` already uses. `tests/test_eval.py`
verifies the harness's scoring/aggregation logic with a mocked `llm.complete()` and runs in CI
on every push; the real scored run against real LLM calls is a manual, opt-in CLI command.

## Why

- **category/team stay exact-match, not judged:** that label space is already small, fixed, and
  deterministic (Stage 2's classifier owns it) - paying for an LLM call or risking its variance
  to re-check a fact the classifier already states precisely would be pure waste.
- **root_causes needs a judge, not exact match:** it's open-ended natural-language reasoning,
  so there's no string to compare against directly. An LLM-as-judge (reusing `llm.complete()`
  and `ToolSpec`, the same abstraction Stage 3 already built) is the realistic, teachable
  approach - and avoids the brittleness of keyword/rubric matching against free text.
- **Three-way verdict (`correct`/`partially_correct`/`incorrect`), not binary:** mirrors
  `TRIAGE_TOOL`'s existing confidence-enum precedent, and yields two honestly different metrics
  instead of one lossy pass/fail number: `root_cause_exact_pass_rate` (strict, counts only
  `correct`) and `root_cause_mean_score` (weighted mean, `correct`=1.0/`partially_correct`=0.5/
  `incorrect`=0.0). Both are reported - nothing is hidden behind a single headline number.
- **CI runs the harness tested, not the harness run for real:** the existing CI job already uses
  a fake `LLM_API_KEY=ci-test-key-not-real`, and no test before this one ever made a real LLM
  call. Every eval case here would cost two real LLM calls (one to triage, one to judge) -
  running that automatically on every push would spend real API quota and add latency/flakiness
  risk for no CI-gating benefit a mocked test doesn't already provide. `tests/test_eval.py`
  mocks `llm.complete()` (dispatched by `tool.name`, since both `triage()` and
  `judge_root_cause()` go through the same function) to verify the scoring/aggregation math
  itself; the real scored run (`stages/05_run_eval.py`) stays manual and opt-in, mirroring how
  Stage 1/3's real-API verification already works.
- **Self-evaluation bias is a named, accepted limitation, not a silent gap:** judging with the
  same configured model that generated the triage output is a known weakness (a model may be
  systematically lenient toward its own reasoning style). This stage doesn't build around it -
  no separate required judge-model config - it's documented here as an explicit, accepted
  tradeoff for this stage's scope.
- **`run_eval()`'s report mirrors `classifier.evaluate()`'s dict-of-scalar-metrics shape, plus a
  `cases` list:** the scalar keys (`category_accuracy`, `team_accuracy`, ...) match Stage 2's
  precedent directly; `cases` is the one intentional departure, because a wrong judge verdict
  needs to be inspectable per-case to be useful (unlike a CV fold score, which is only useful
  aggregated).
- **Runbooks explicitly stay out of scope for this plan:** `data/README.md` stubs `runbooks/`
  as existing "Stage 4 onward," but Stage 4 never built it - only `data/incidents/` ever got
  embedded. Rather than building runbooks retroactively to evaluate a system that was never
  built, Stage 5 evaluates what actually exists today (classifier + LLM root-cause/next-steps +
  incident-only retrieval). Runbooks remain a possible later stage if a concrete case shows
  incident-only retrieval giving worse `next_steps` than a runbook would.

## Rejected

- Embedding-similarity scoring for root cause (e.g. cosine similarity between the candidate and
  reference text): a weak proxy for whether the *diagnosis* is actually correct, not just
  whether the wording is similar.
- A separately-configured judge model (distinct from `LLM_PROVIDER`/`LLM_MODEL`): would remove
  the self-evaluation bias, but forces a second required model configuration this stage doesn't
  want to mandate yet - parked as possible future work rather than built now.
- Running the real eval automatically in CI on every push: real cost and non-determinism for a
  check a mocked harness test already covers (the scoring logic itself, not the model's actual
  accuracy on any given day).
- Human-only grading: accurate, but not a repeatable, automatable harness - defeats the purpose
  of having a Stage 5 at all.
- A single binary pass/fail verdict: loses the correct/partially_correct distinction that two
  separate, honest metrics preserve.
- Building out runbooks now to evaluate the originally-stubbed full system: see Why above -
  deferred rather than retroactively building a feature this plan isn't about, just to evaluate it.
