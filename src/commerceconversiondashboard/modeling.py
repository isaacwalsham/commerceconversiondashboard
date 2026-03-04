"""Model selection utilities."""

from __future__ import annotations

from typing import Dict

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def get_candidate_models(random_state: int = 42) -> Dict[str, object]:
    """Return candidate estimators for comparison."""
    return {
        "logistic_regression": LogisticRegression(
            solver="liblinear",
            class_weight="balanced",
            max_iter=2000,
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=350,
            max_depth=12,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        ),
    }
