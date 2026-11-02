# Data

Synthetic data only. Never put real customer, payment or policy data here.

- `incidents/` will hold about 50 fake past incidents (Stage 2 onward).
- `runbooks/` will hold about 10 runbooks in Markdown (Stage 4 onward).
- `evals/` holds the scored test set (Stage 5): 15 JSON files, each with `error_text`,
  `expected_category`, `expected_team`, and a human-written `expected_root_cause`.
