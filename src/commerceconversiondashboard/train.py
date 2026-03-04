"""Training orchestration for conversion prediction."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from .data import (
    download_online_shoppers_dataset,
    load_dataset,
    persist_split_data,
    split_features_target,
)
from .features import build_preprocessor
from .metrics import (
    calibration_table,
    classification_metrics,
    confusion_stats,
    decile_lift_table,
    optimal_threshold_from_value,
    precision_recall_points,
    roc_curve_points,
    threshold_business_table,
)
from .modeling import get_candidate_models
from .paths import (
    CALIBRATION_FILE,
    EVAL_FILE,
    EXECUTIVE_SUMMARY_FILE,
    FEATURE_IMPORTANCE_FILE,
    FIGURES_DIR,
    LIFT_TABLE_FILE,
    METADATA_FILE,
    MODEL_CARD_FILE,
    MODEL_FILE,
    MODELS_DIR,
    PREDICTIONS_FILE,
    REPORTS_DIR,
    ROC_CURVE_FILE,
)

SCORING = {
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
}


@dataclass
class TrainingSummary:
    """Top-level training outputs for CLI/automation use."""

    selected_model: str
    cv_pr_auc: float
    cv_roc_auc: float
    test_pr_auc: float
    test_roc_auc: float
    optimal_threshold: float
    optimal_expected_value: float


def _build_model_card_markdown(
    *,
    selected_model: str,
    metrics_default: dict,
    metrics_optimal: dict,
    dataset_rows: int,
    positive_rate: float,
    optimal_threshold: float,
) -> str:
    generated_at = datetime.now(timezone.utc).isoformat()
    return f"""# Model Card: Conversion Project

## 1. Model Details
- Model type: `{selected_model}`
- Objective: predict session-level purchase conversion
- Training timestamp (UTC): {generated_at}

## 2. Data
- Dataset: UCI Online Shoppers Purchasing Intention
- Rows: {dataset_rows:,}
- Observed conversion rate: {positive_rate:.2%}

## 3. Preprocessing
- Numeric features: median imputation + standardization
- Categorical features: mode imputation + one-hot encoding
- Split strategy: stratified train/test
- Model selection: 5-fold stratified cross-validation (PR-AUC primary)

## 4. Performance
- ROC-AUC (test, threshold 0.50): {metrics_default["roc_auc"]:.3f}
- PR-AUC (test, threshold 0.50): {metrics_default["pr_auc"]:.3f}
- F1 (test, threshold 0.50): {metrics_default["f1"]:.3f}
- Brier score (test): {metrics_default["brier_score"]:.4f}
- Optimized threshold (value-based): {optimal_threshold:.2f}
- F1 (test, optimized threshold): {metrics_optimal["f1"]:.3f}

## 5. Intended Use
- Rank sessions by purchase likelihood
- Compare threshold trade-offs in one place
- Use as a learning project for classification workflows

## 6. Limitations
- Single public dataset may not represent all verticals/regions
- No causal inference: scores estimate likelihood, not treatment effect
- Data drift (seasonality, channel changes) can reduce performance

## 7. Monitoring Plan
- Weekly: score distribution drift and conversion-rate drift
- Monthly: calibration check and decile lift stability
- Quarterly: retrain and compare metrics against previous runs
"""


def _build_executive_summary_markdown(
    *,
    selected_model: str,
    metrics_default: dict,
    optimal_threshold: float,
    optimal_expected_value: float,
    top_decile_rate: float,
    top_decile_lift: float,
    top_decile_capture: float,
) -> str:
    return f"""# Executive Summary

## Headline
The selected model (`{selected_model}`) ranks likely converters well and gives a useful threshold for targeting.

## Key Metrics
- PR-AUC: {metrics_default["pr_auc"]:.3f}
- ROC-AUC: {metrics_default["roc_auc"]:.3f}
- Brier score: {metrics_default["brier_score"]:.4f}
- Best value threshold: {optimal_threshold:.2f}
- Max expected value (relative units): {optimal_expected_value:,.1f}

## Prioritization Signal
- Top decile conversion rate: {top_decile_rate:.2%}
- Top decile lift vs baseline: {top_decile_lift:.2f}x
- Cumulative conversions captured by top decile: {top_decile_capture:.2%}

## Recommended Actions
1. Use `{optimal_threshold:.2f}` as a starting threshold for \"likely to convert\".
2. Use decile rank to focus attention on top-scoring sessions first.
3. Track calibration monthly; recalibrate or retrain if gap widens.

## Caveats
- This is a predictive model, not a causal model.
- Expected-value results depend on the conversion value/contact cost assumptions.
"""


def _ensure_output_dirs() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def _build_pipeline(preprocessor, estimator) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", clone(preprocessor)),
            ("classifier", estimator),
        ]
    )


def _run_cross_validation(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int,
) -> Tuple[pd.DataFrame, Dict[str, Pipeline]]:
    preprocessor = build_preprocessor(x_train)
    candidates = get_candidate_models(random_state=random_state)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    cv_rows = []
    pipelines: Dict[str, Pipeline] = {}

    for name, estimator in candidates.items():
        pipeline = _build_pipeline(preprocessor=preprocessor, estimator=estimator)
        pipelines[name] = pipeline

        scores = cross_validate(
            pipeline,
            x_train,
            y_train,
            scoring=SCORING,
            cv=cv,
            n_jobs=1,
            return_train_score=False,
        )

        row = {"model": name}
        for metric in SCORING:
            metric_scores = scores[f"test_{metric}"]
            row[f"{metric}_mean"] = float(np.mean(metric_scores))
            row[f"{metric}_std"] = float(np.std(metric_scores))
        cv_rows.append(row)

    results = pd.DataFrame(cv_rows).sort_values(by="pr_auc_mean", ascending=False)
    return results, pipelines


def _extract_feature_importance(fitted_pipeline: Pipeline) -> pd.DataFrame:
    preprocessor = fitted_pipeline.named_steps["preprocessor"]
    classifier = fitted_pipeline.named_steps["classifier"]

    feature_names = preprocessor.get_feature_names_out()

    if hasattr(classifier, "feature_importances_"):
        values = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        values = np.abs(classifier.coef_.ravel())
    else:
        values = np.zeros(len(feature_names))

    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": values,
        }
    ).sort_values("importance", ascending=False)

    return importance


def run_training_pipeline(
    random_state: int = 42,
    force_download: bool = False,
    data_path: Path | None = None,
    dataset_name: str | None = None,
) -> TrainingSummary:
    """Run full train/evaluate/save workflow."""
    _ensure_output_dirs()

    if data_path is None:
        raw_file = download_online_shoppers_dataset(force=force_download)
        resolved_dataset_name = dataset_name or "UCI Online Shoppers (Auto-download)"
    else:
        raw_file = Path(data_path)
        resolved_dataset_name = dataset_name or raw_file.stem

    df = load_dataset(raw_file)
    split_data = split_features_target(df=df, random_state=random_state)
    persist_split_data(split_data)

    cv_results, pipelines = _run_cross_validation(
        x_train=split_data.x_train,
        y_train=split_data.y_train,
        random_state=random_state,
    )

    selected_model = cv_results.iloc[0]["model"]
    selected_pipeline = pipelines[selected_model]
    selected_pipeline.fit(split_data.x_train, split_data.y_train)

    y_prob = selected_pipeline.predict_proba(split_data.x_test)[:, 1]
    y_pred_default = (y_prob >= 0.5).astype(int)

    metrics_default = classification_metrics(
        y_true=split_data.y_test,
        y_pred=y_pred_default,
        y_prob=y_prob,
    )
    confusion_default = confusion_stats(split_data.y_test, y_pred_default)

    value_table = threshold_business_table(
        y_true=split_data.y_test,
        y_prob=y_prob,
        conversion_value=60.0,
        contact_cost=0.75,
    )
    optimal_threshold, optimal_expected_value = optimal_threshold_from_value(value_table)

    y_pred_optimal = (y_prob >= optimal_threshold).astype(int)
    metrics_optimal = classification_metrics(
        y_true=split_data.y_test,
        y_pred=y_pred_optimal,
        y_prob=y_prob,
    )
    confusion_optimal = confusion_stats(split_data.y_test, y_pred_optimal)

    pr_points = precision_recall_points(split_data.y_test, y_prob)
    roc_points = roc_curve_points(split_data.y_test, y_prob)
    calibration_points = calibration_table(split_data.y_test, y_prob)
    lift_table = decile_lift_table(split_data.y_test, y_prob)

    predictions = split_data.x_test.copy()
    predictions["actual_conversion"] = split_data.y_test.values
    predictions["predicted_probability"] = y_prob
    predictions["predicted_default"] = y_pred_default
    predictions["predicted_optimal"] = y_pred_optimal

    feature_importance = _extract_feature_importance(selected_pipeline)

    cv_results.to_csv(REPORTS_DIR / "cv_results.csv", index=False)
    value_table.to_csv(REPORTS_DIR / "threshold_value_table.csv", index=False)
    pr_points.to_csv(REPORTS_DIR / "precision_recall_curve.csv", index=False)
    roc_points.to_csv(ROC_CURVE_FILE, index=False)
    calibration_points.to_csv(CALIBRATION_FILE, index=False)
    lift_table.to_csv(LIFT_TABLE_FILE, index=False)
    predictions.to_csv(PREDICTIONS_FILE, index=False)
    feature_importance.to_csv(FEATURE_IMPORTANCE_FILE, index=False)

    top_decile = lift_table.iloc[0]

    metadata = {
        "model_name": selected_model,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_columns": split_data.x_train.columns.tolist(),
        "default_threshold": 0.5,
        "optimal_threshold": optimal_threshold,
        "data_source_name": resolved_dataset_name,
        "data_source_file": str(raw_file),
    }

    summary_payload = {
        "dataset": {
            "rows": int(len(df)),
            "columns": int(df.shape[1]),
            "positive_rate": float(df["Revenue"].mean()),
            "source_name": resolved_dataset_name,
            "source_file": str(raw_file),
        },
        "cross_validation": cv_results.to_dict(orient="records"),
        "selected_model": selected_model,
        "metrics_default_threshold": metrics_default,
        "metrics_optimal_threshold": metrics_optimal,
        "confusion_default_threshold": confusion_default,
        "confusion_optimal_threshold": confusion_optimal,
        "optimal_threshold": optimal_threshold,
        "optimal_expected_value": optimal_expected_value,
        "calibration_mean_abs_gap": float(calibration_points["calibration_gap"].abs().mean()),
        "top_decile_stats": {
            "conversion_rate": float(top_decile["conversion_rate"]),
            "lift_vs_baseline": float(top_decile["lift_vs_baseline"]),
            "cumulative_capture_rate": float(top_decile["cumulative_capture_rate"]),
        },
    }

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(selected_pipeline, MODEL_FILE)

    METADATA_FILE.write_text(json.dumps(metadata, indent=2))
    EVAL_FILE.write_text(json.dumps(summary_payload, indent=2))
    MODEL_CARD_FILE.write_text(
        _build_model_card_markdown(
            selected_model=selected_model,
            metrics_default=metrics_default,
            metrics_optimal=metrics_optimal,
            dataset_rows=int(len(df)),
            positive_rate=float(df["Revenue"].mean()),
            optimal_threshold=float(optimal_threshold),
        )
    )
    EXECUTIVE_SUMMARY_FILE.write_text(
        _build_executive_summary_markdown(
            selected_model=selected_model,
            metrics_default=metrics_default,
            optimal_threshold=float(optimal_threshold),
            optimal_expected_value=float(optimal_expected_value),
            top_decile_rate=float(top_decile["conversion_rate"]),
            top_decile_lift=float(top_decile["lift_vs_baseline"]),
            top_decile_capture=float(top_decile["cumulative_capture_rate"]),
        )
    )

    return TrainingSummary(
        selected_model=selected_model,
        cv_pr_auc=float(cv_results.iloc[0]["pr_auc_mean"]),
        cv_roc_auc=float(cv_results.iloc[0]["roc_auc_mean"]),
        test_pr_auc=float(metrics_default["pr_auc"]),
        test_roc_auc=float(metrics_default["roc_auc"]),
        optimal_threshold=float(optimal_threshold),
        optimal_expected_value=float(optimal_expected_value),
    )


def run_and_save_summary_json(output_path: Path) -> None:
    """Convenience utility for script entrypoints."""
    summary = run_training_pipeline()
    output_path.write_text(json.dumps(asdict(summary), indent=2))
