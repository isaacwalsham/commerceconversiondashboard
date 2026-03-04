"""Model evaluation and business metric helpers."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
    roc_auc_score,
)


def classification_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, float]:
    """Compute core classifier metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
    }


def confusion_stats(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, int]:
    """Return confusion matrix values as a dictionary."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def threshold_business_table(
    y_true: pd.Series,
    y_prob: np.ndarray,
    conversion_value: float = 60.0,
    contact_cost: float = 0.75,
    min_threshold: float = 0.05,
    max_threshold: float = 0.95,
    step: float = 0.01,
) -> pd.DataFrame:
    """Compute a threshold table with business value estimates."""
    rows = []
    thresholds = np.arange(min_threshold, max_threshold + step, step)

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)

        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())

        targeted = tp + fp
        precision = tp / targeted if targeted else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        expected_value = (tp * conversion_value) - (targeted * contact_cost)

        rows.append(
            {
                "threshold": round(float(threshold), 4),
                "targeted_sessions": targeted,
                "tp": tp,
                "fp": fp,
                "precision": precision,
                "recall": recall,
                "expected_value": expected_value,
            }
        )

    return pd.DataFrame(rows)


def optimal_threshold_from_value(table: pd.DataFrame) -> Tuple[float, float]:
    """Return threshold that maximizes expected value."""
    idx = int(table["expected_value"].idxmax())
    row = table.loc[idx]
    return float(row["threshold"]), float(row["expected_value"])


def precision_recall_points(y_true: pd.Series, y_prob: np.ndarray) -> pd.DataFrame:
    """Return precision-recall curve points for charting."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    # sklearn returns one fewer threshold than precision/recall entries
    padded_thresholds = np.append(thresholds, 1.0)
    return pd.DataFrame(
        {
            "precision": precision,
            "recall": recall,
            "threshold": padded_thresholds,
        }
    )


def roc_curve_points(y_true: pd.Series, y_prob: np.ndarray) -> pd.DataFrame:
    """Return ROC curve points for charting."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    return pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds})


def calibration_table(
    y_true: pd.Series,
    y_prob: np.ndarray,
    bins: int = 10,
) -> pd.DataFrame:
    """Return quantile-binned calibration summary."""
    frame = pd.DataFrame({"y_true": y_true.values, "y_prob": y_prob})
    frame["bin"] = pd.qcut(frame["y_prob"], q=bins, labels=False, duplicates="drop")

    grouped = (
        frame.groupby("bin", dropna=True)
        .agg(
            avg_predicted_prob=("y_prob", "mean"),
            observed_conversion_rate=("y_true", "mean"),
            sessions=("y_true", "size"),
            conversions=("y_true", "sum"),
        )
        .reset_index()
    )

    grouped["bin"] = grouped["bin"].astype(int) + 1
    grouped["calibration_gap"] = (
        grouped["observed_conversion_rate"] - grouped["avg_predicted_prob"]
    )
    return grouped.sort_values("bin")


def decile_lift_table(y_true: pd.Series, y_prob: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Return decile lift chart data sorted by model score."""
    frame = pd.DataFrame({"y_true": y_true.values, "y_prob": y_prob}).sort_values(
        "y_prob", ascending=False
    )
    frame["rank"] = np.arange(1, len(frame) + 1)
    frame["decile"] = pd.qcut(frame["rank"], q=bins, labels=False, duplicates="drop") + 1

    total_conversions = max(int(frame["y_true"].sum()), 1)
    baseline_rate = float(frame["y_true"].mean())

    grouped = (
        frame.groupby("decile", dropna=True)
        .agg(
            sessions=("y_true", "size"),
            conversions=("y_true", "sum"),
            avg_score=("y_prob", "mean"),
        )
        .reset_index()
        .sort_values("decile")
    )
    grouped["conversion_rate"] = grouped["conversions"] / grouped["sessions"]
    grouped["lift_vs_baseline"] = grouped["conversion_rate"] / max(baseline_rate, 1e-9)
    grouped["cumulative_conversions"] = grouped["conversions"].cumsum()
    grouped["cumulative_capture_rate"] = grouped["cumulative_conversions"] / total_conversions
    return grouped
