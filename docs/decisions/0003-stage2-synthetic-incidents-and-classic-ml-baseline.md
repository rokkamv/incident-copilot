# 0003: Stage 2 synthetic incidents and classic ML baseline

**Status:** accepted

## Decision

Stage 2 (`stages/02_classify_incident.py`) predicts two labels — `category` (7-way) and `team`
(6-way) — from an incident's `error_text`, using two independent TF-IDF + logistic regression
scikit-learn pipelines. It trains on 50 synthetic incidents under `data/incidents/` (one JSON
file per incident) and reports accuracy via stratified 5-fold cross-validation rather than a
single train/test split.

## Why

- **Dual labels (`category` + `team`), not one:** on-call triage needs both "what kind of
  problem" and "who owns it." A single label would under-specify what a real triage flow (and
  Stage 3's `/triage` endpoint) needs to surface.
- **Classic linear ML, not deep learning:** a deliberate contrast to Stage 1's LLM call — this
  baseline is instant, free to run, fully inspectable (you can print a pipeline's
  `coef_` directly), and needs no API key. It establishes a floor that later stages (RAG in Stage
  4, evaluation in Stage 5) have to beat to justify their added cost and complexity.
- **Two independent pipelines, not one multi-output model:** at n=50 with differently-sized label
  spaces (7 vs. 6 classes), a shared/multi-output estimator adds indirection with no real benefit.
  Two plain pipelines are more teachable in isolation and stay fully decoupled for future tuning.
- **Synthetic data, one JSON file per incident:** no real incident corpus exists, and the
  project's synthetic-data-only policy (`data/README.md`) already rules one out. A small, fixed
  taxonomy (7 categories x 6 teams, ~10 services) with templated generation keeps labels clean and
  keeps the lesson about ML mechanics, not data cleaning. The `category` -> `team` mapping is
  deliberately not 1:1 (`third-party` maps to either `payments` or `app-backend` depending on
  which service is involved), so `team` prediction is a genuinely separate task from `category`,
  not a disguised lookup table.
- **Stratified 5-fold CV over a single train/test split:** at n=50, an 80/20 split's test set is
  only ~10 rows spread across a 7-way and a 6-way label space — one unlucky split can swing
  accuracy by a huge margin. 5-fold CV uses every row as a test example exactly once. The dataset's
  label distribution (every category >= 6 rows, every team >= 5 rows) was chosen specifically to
  support this without scikit-learn erroring on an under-populated class.

## Rejected

- Deep learning / a fine-tuned transformer: overkill for 50 rows, and defeats the point of a
  "classic ML baseline" to contrast against Stage 1's LLM approach.
- A single multi-output classifier or a shared vectorizer across both labels: unnecessary
  indirection at this dataset size.
- A single random train/test split: too noisy at n=50 across two multi-class label spaces.
- pandas for loading the data: 50 JSON records are handled directly by stdlib `json`; pulling in
  pandas would add a heavy dependency and an unrelated API surface to a lesson about scikit-learn
  pipelines, not dataframes.
- Real or scraped incident data: none exists, and the project is synthetic-data-only by policy.
