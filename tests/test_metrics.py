import numpy as np
import pandas as pd

from commerceconversiondashboard.metrics import (
    calibration_table,
    classification_metrics,
    decile_lift_table,
    optimal_threshold_from_value,
    threshold_business_table,
)


def test_threshold_business_table_contains_expected_columns():
    y_true = pd.Series([0, 1, 0, 1, 1, 0])
    y_prob = np.array([0.1, 0.7, 0.4, 0.8, 0.6, 0.2])

    table = threshold_business_table(y_true=y_true, y_prob=y_prob)

    expected_cols = {
        "threshold",
        "targeted_sessions",
        "tp",
        "fp",
        "precision",
        "recall",
        "expected_value",
    }
    assert expected_cols.issubset(set(table.columns))


def test_optimal_threshold_is_in_range():
    y_true = pd.Series([0, 1, 0, 1, 1, 0])
    y_prob = np.array([0.1, 0.7, 0.4, 0.8, 0.6, 0.2])

    table = threshold_business_table(y_true=y_true, y_prob=y_prob)
    threshold, value = optimal_threshold_from_value(table)

    assert 0.0 <= threshold <= 1.0
    assert isinstance(value, float)


def test_classification_metrics_contains_brier_score():
    y_true = pd.Series([0, 1, 0, 1, 1, 0, 1, 0])
    y_prob = np.array([0.1, 0.8, 0.25, 0.7, 0.6, 0.3, 0.9, 0.2])
    y_pred = (y_prob >= 0.5).astype(int)

    result = classification_metrics(y_true=y_true, y_pred=y_pred, y_prob=y_prob)

    assert "brier_score" in result
    assert 0.0 <= result["brier_score"] <= 1.0


def test_calibration_table_expected_shape():
    y_true = pd.Series([0, 1, 0, 1, 1, 0, 1, 0, 1, 0])
    y_prob = np.array([0.08, 0.83, 0.21, 0.77, 0.67, 0.32, 0.89, 0.19, 0.58, 0.44])

    table = calibration_table(y_true=y_true, y_prob=y_prob, bins=5)

    expected_cols = {
        "bin",
        "avg_predicted_prob",
        "observed_conversion_rate",
        "sessions",
        "conversions",
        "calibration_gap",
    }
    assert expected_cols.issubset(table.columns)
    assert len(table) <= 5


def test_decile_lift_table_has_lift_column():
    y_true = pd.Series([0, 1, 0, 1, 1, 0, 0, 1, 0, 1] * 5)
    y_prob = np.linspace(0.01, 0.99, len(y_true))

    table = decile_lift_table(y_true=y_true, y_prob=y_prob, bins=10)

    assert "lift_vs_baseline" in table.columns
    assert table["decile"].min() == 1
