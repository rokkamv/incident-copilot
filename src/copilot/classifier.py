"""Classic ML baseline: TF-IDF + logistic regression, predicting category and team.

Two independent pipelines are trained (not a shared/multi-output model) since
category (7 classes) and team (6 classes) are differently-sized label spaces
with no benefit from sharing a vectorizer at this dataset size.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

LABELS = ("category", "team")


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )


def train(incidents: list[dict]) -> dict[str, Pipeline]:
    texts = [incident["error_text"] for incident in incidents]
    models = {}
    for label in LABELS:
        pipeline = build_pipeline()
        pipeline.fit(texts, [incident[label] for incident in incidents])
        models[label] = pipeline
    return models


def evaluate(incidents: list[dict], n_splits: int = 5, random_state: int = 42) -> dict[str, float]:
    texts = [incident["error_text"] for incident in incidents]
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    scores = {}
    for label in LABELS:
        targets = [incident[label] for incident in incidents]
        fold_scores = cross_val_score(build_pipeline(), texts, targets, cv=cv)
        scores[f"{label}_accuracy"] = fold_scores.mean()
    return scores


def predict(models: dict[str, Pipeline], error_text: str) -> dict[str, str]:
    # predict() is batch-oriented: wrapping error_text in a list and taking [0]
    # unwraps the single resulting prediction back into a plain string.
    return {label: models[label].predict([error_text])[0] for label in LABELS}
