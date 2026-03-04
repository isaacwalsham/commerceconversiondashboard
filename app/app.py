"""Streamlit dashboard for website conversion prediction."""

from __future__ import annotations

from dataclasses import dataclass
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from commerceconversiondashboard.inference import load_metadata, load_model
from commerceconversiondashboard.paths import (
    CALIBRATION_FILE,
    EVAL_FILE,
    EXECUTIVE_SUMMARY_FILE,
    FEATURE_IMPORTANCE_FILE,
    LIFT_TABLE_FILE,
    METADATA_FILE,
    MODEL_CARD_FILE,
    MODEL_FILE,
    PREDICTIONS_FILE,
    RAW_DATA_FILE,
    REPORTS_DIR,
    ROC_CURVE_FILE,
)
from commerceconversiondashboard.train import run_training_pipeline

st.set_page_config(
    page_title="Commerce Conversion Intelligence",
    page_icon="📈",
    layout="wide",
)

PROFILE_OPTIONS = [
    "Low Intent (10th percentile)",
    "Median Intent (50th percentile)",
    "High Intent (90th percentile)",
]

PREFERRED_SCENARIO_FEATURES = [
    "PageValues",
    "BounceRates",
    "ExitRates",
    "ProductRelated",
    "ProductRelated_Duration",
    "Administrative_Duration",
    "Informational_Duration",
    "SpecialDay",
    "Weekend",
]

SOURCE_OPTION_BUNDLED = "Built-in datasets"
SOURCE_OPTION_UPLOAD = "Upload a CSV"
SOURCE_OPTION_DOWNLOAD = "Auto-download from UCI"
SOURCE_OPTIONS = [SOURCE_OPTION_BUNDLED, SOURCE_OPTION_UPLOAD, SOURCE_OPTION_DOWNLOAD]


@dataclass(frozen=True)
class BundledDatasetOption:
    dataset_id: str
    name: str
    file: str
    description: str
    path: Path


def _percent(value: float, digits: int = 2) -> str:
    return f"{value * 100:.{digits}f}%"


@st.cache_resource(show_spinner=False)
def _cached_model():
    return load_model()


@st.cache_data(show_spinner=False)
def _load_reports() -> dict:
    evaluation = json.loads(EVAL_FILE.read_text())
    cv_results = pd.read_csv(REPORTS_DIR / "cv_results.csv")
    threshold_table = pd.read_csv(REPORTS_DIR / "threshold_value_table.csv")
    pr_curve = pd.read_csv(REPORTS_DIR / "precision_recall_curve.csv")
    roc_curve = pd.read_csv(ROC_CURVE_FILE)
    calibration_curve = pd.read_csv(CALIBRATION_FILE)
    lift_table = pd.read_csv(LIFT_TABLE_FILE)
    predictions = pd.read_csv(PREDICTIONS_FILE)
    feature_importance = pd.read_csv(FEATURE_IMPORTANCE_FILE)
    metadata = load_metadata()
    model_card = MODEL_CARD_FILE.read_text() if MODEL_CARD_FILE.exists() else ""
    executive_summary = EXECUTIVE_SUMMARY_FILE.read_text() if EXECUTIVE_SUMMARY_FILE.exists() else ""

    return {
        "evaluation": evaluation,
        "cv_results": cv_results,
        "threshold_table": threshold_table,
        "pr_curve": pr_curve,
        "roc_curve": roc_curve,
        "calibration_curve": calibration_curve,
        "lift_table": lift_table,
        "predictions": predictions,
        "feature_importance": feature_importance,
        "metadata": metadata,
        "model_card": model_card,
        "executive_summary": executive_summary,
    }


def _artifacts_ready() -> bool:
    required = [
        MODEL_FILE,
        METADATA_FILE,
        EVAL_FILE,
        PREDICTIONS_FILE,
        FEATURE_IMPORTANCE_FILE,
        ROC_CURVE_FILE,
        CALIBRATION_FILE,
        LIFT_TABLE_FILE,
        MODEL_CARD_FILE,
        EXECUTIVE_SUMMARY_FILE,
    ]
    return all(p.exists() for p in required)


def _clear_cached_artifacts() -> None:
    _load_reports.clear()
    _cached_model.clear()


def _safe_rerun() -> None:
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


def _list_bundled_dataset_options() -> list[BundledDatasetOption]:
    bundled_dir = PROJECT_ROOT / "data" / "bundled"
    manifest_path = bundled_dir / "datasets_manifest.json"

    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text())
        items = []
        for entry in payload.get("datasets", []):
            if not all(key in entry for key in ["id", "name", "file", "description"]):
                continue
            path = bundled_dir / str(entry["file"])
            if not path.exists():
                continue
            items.append(
                BundledDatasetOption(
                    dataset_id=str(entry["id"]),
                    name=str(entry["name"]),
                    file=str(entry["file"]),
                    description=str(entry["description"]),
                    path=path,
                )
            )
        if items:
            return items

    return [
        BundledDatasetOption(
            dataset_id=path.stem,
            name=path.stem.replace("_", " ").title(),
            file=path.name,
            description="Bundled dataset file.",
            path=path,
        )
        for path in sorted(bundled_dir.glob("*.csv"))
    ]


def _dataset_training_controls(key_prefix: str, button_label: str) -> None:
    bundled = _list_bundled_dataset_options()

    source_option = st.radio(
        "Dataset source",
        SOURCE_OPTIONS,
        index=0,
        key=f"{key_prefix}_source_option",
    )

    data_path = None
    dataset_name = None

    if source_option == SOURCE_OPTION_BUNDLED:
        if not bundled:
            st.warning("No built-in datasets found in `data/bundled/`.")
        else:
            label_to_dataset = {
                f"{dataset.name} ({dataset.file})": dataset for dataset in bundled
            }
            selected_label = st.selectbox(
                "Choose a built-in dataset",
                list(label_to_dataset.keys()),
                key=f"{key_prefix}_bundled_choice",
            )
            selected_dataset = label_to_dataset[selected_label]
            data_path = selected_dataset.path
            dataset_name = selected_dataset.name
            st.caption(selected_dataset.description)

    elif source_option == SOURCE_OPTION_UPLOAD:
        uploaded_file = st.file_uploader(
            "Upload a CSV",
            type=["csv"],
            accept_multiple_files=False,
            key=f"{key_prefix}_upload_file",
        )
        if uploaded_file is not None:
            RAW_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
            target_path = RAW_DATA_FILE.parent / f"{key_prefix}_{uploaded_file.name}"
            target_path.write_bytes(uploaded_file.getvalue())
            data_path = target_path
            dataset_name = uploaded_file.name

    train_clicked = st.button(button_label, key=f"{key_prefix}_train_button")
    if not train_clicked:
        return

    if source_option == SOURCE_OPTION_BUNDLED and data_path is None:
        st.error("Pick a built-in dataset before training.")
        return

    if source_option == SOURCE_OPTION_UPLOAD and data_path is None:
        st.error("Upload a CSV file before training.")
        return

    try:
        with st.spinner("Training model..."):
            summary = run_training_pipeline(data_path=data_path, dataset_name=dataset_name)
        _clear_cached_artifacts()
        st.success(
            (
                f"Training complete. Model: {summary.selected_model} | "
                f"PR-AUC: {summary.test_pr_auc:.3f} | ROC-AUC: {summary.test_roc_auc:.3f}"
            )
        )
        _safe_rerun()
    except RuntimeError as error:
        st.error(str(error))


def _render_confusion_matrix(confusion: dict, title: str) -> go.Figure:
    matrix = np.array([[confusion["tn"], confusion["fp"]], [confusion["fn"], confusion["tp"]]])

    fig = px.imshow(
        matrix,
        text_auto=True,
        x=["Pred: No Convert", "Pred: Convert"],
        y=["Actual: No Convert", "Actual: Convert"],
        color_continuous_scale="Blues",
        title=title,
        aspect="equal",
    )
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=55, b=10))
    return fig


def _render_probability_gauge(probability: float, threshold: float) -> go.Figure:
    color = "#16a34a" if probability >= threshold else "#f97316"
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%", "font": {"size": 36}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, threshold * 100], "color": "#e2e8f0"},
                    {"range": [threshold * 100, 100], "color": "#bfdbfe"},
                ],
                "threshold": {
                    "line": {"color": "#0f172a", "width": 3},
                    "thickness": 0.9,
                    "value": threshold * 100,
                },
            },
            title={"text": "Predicted Conversion Probability"},
        )
    )
    fig.update_layout(height=300, margin=dict(l=8, r=8, t=70, b=8))
    return fig


def _select_profile_row(source: pd.DataFrame, profile_name: str) -> pd.Series:
    sorted_data = source.sort_values("predicted_probability").reset_index(drop=True)

    if profile_name == PROFILE_OPTIONS[0]:
        idx = int(len(sorted_data) * 0.10)
    elif profile_name == PROFILE_OPTIONS[2]:
        idx = int(len(sorted_data) * 0.90)
    else:
        idx = int(len(sorted_data) * 0.50)

    idx = int(np.clip(idx, 0, len(sorted_data) - 1))
    return sorted_data.iloc[idx]


def _scenario_feature_list(source: pd.DataFrame) -> list[str]:
    numeric_cols = source.select_dtypes(include=["number"]).columns.tolist()
    ordered = [c for c in PREFERRED_SCENARIO_FEATURES if c in numeric_cols]
    remaining = [c for c in numeric_cols if c not in ordered]
    return (ordered + remaining)[:6]


def _apply_percent_adjustment(value: float, pct_change: float, clamp_01: bool = False) -> float:
    adjusted = float(value) * (1.0 + pct_change / 100.0)
    if clamp_01:
        return float(np.clip(adjusted, 0.0, 1.0))
    return adjusted


def _predict_probability(model, payload: pd.DataFrame) -> float:
    return float(model.predict_proba(payload)[:, 1][0])


def _scenario_sensitivity_table(model, baseline_payload: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    base_score = _predict_probability(model, baseline_payload)
    rows = []

    for feature in features:
        value = float(baseline_payload.iloc[0][feature])
        down_payload = baseline_payload.copy()
        up_payload = baseline_payload.copy()

        clamp_01 = feature.lower().endswith("rates") or feature.lower() in {"weekend", "specialday"}
        down_payload.loc[:, feature] = _apply_percent_adjustment(value, -10.0, clamp_01=clamp_01)
        up_payload.loc[:, feature] = _apply_percent_adjustment(value, 10.0, clamp_01=clamp_01)

        down_score = _predict_probability(model, down_payload)
        up_score = _predict_probability(model, up_payload)

        rows.append(
            {
                "feature": feature,
                "delta_down": down_score - base_score,
                "delta_up": up_score - base_score,
            }
        )

    if not rows:
        return pd.DataFrame(columns=["feature", "delta_down", "delta_up"])

    return pd.DataFrame(rows).sort_values("delta_up", ascending=False)


def _downsample_curve(curve: pd.DataFrame, max_points: int = 1200) -> pd.DataFrame:
    if len(curve) <= max_points:
        return curve
    idx = np.linspace(0, len(curve) - 1, max_points).astype(int)
    return curve.iloc[idx].copy()


def main() -> None:
    st.title("Commerce Conversion Dashboard")
    st.caption("A side project for predicting whether a shopping session converts.")

    if not _artifacts_ready():
        st.warning("No trained model artifacts found yet.")
        st.info("Pick a dataset and train the model.")
        _dataset_training_controls(key_prefix="initial", button_label="Train model")
        st.stop()

    data = _load_reports()

    evaluation = data["evaluation"]
    cv_results = data["cv_results"]
    threshold_table = data["threshold_table"]
    pr_curve = _downsample_curve(data["pr_curve"], max_points=1000)
    roc_curve = _downsample_curve(data["roc_curve"], max_points=1000)
    calibration_curve = data["calibration_curve"]
    lift_table = data["lift_table"]
    predictions = data["predictions"]
    feature_importance = data["feature_importance"]
    metadata = data["metadata"]
    model_card = data["model_card"]
    executive_summary = data["executive_summary"]

    metrics_default = evaluation["metrics_default_threshold"]
    optimal_threshold = float(evaluation["optimal_threshold"])
    top_decile = evaluation.get("top_decile_stats", {})

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Model", evaluation["selected_model"].replace("_", " ").title())
    m2.metric("PR-AUC", f"{metrics_default['pr_auc']:.3f}")
    m3.metric("ROC-AUC", f"{metrics_default['roc_auc']:.3f}")
    m4.metric("Brier", f"{metrics_default.get('brier_score', 0.0):.4f}")
    m5.metric("Best Threshold", f"{optimal_threshold:.2f}")
    m6.metric("Top Decile Lift", f"{float(top_decile.get('lift_vs_baseline', 0.0)):.2f}x")

    with st.sidebar:
        st.markdown("### Data")
        st.caption(f"Current dataset: {evaluation['dataset'].get('source_name', 'Unknown')}")
        with st.expander("Train on another dataset"):
            _dataset_training_controls(key_prefix="sidebar", button_label="Retrain")

    section = st.sidebar.radio(
        "Section",
        ["Overview", "Performance", "Value & Lift", "Scenario Lab", "Report Center"],
    )

    if section == "Overview":
        left, right = st.columns([1.3, 1])

        with left:
            distribution_fig = px.histogram(
                predictions,
                x="predicted_probability",
                color="actual_conversion",
                nbins=30,
                barmode="overlay",
                color_discrete_sequence=["#1d4ed8", "#f97316"],
                labels={
                    "predicted_probability": "Predicted Probability",
                    "actual_conversion": "Actual Conversion",
                },
                title="Score Distribution by True Outcome",
            )
            distribution_fig.add_vline(x=optimal_threshold, line_dash="dash", line_color="#0f172a")
            st.plotly_chart(distribution_fig, use_container_width=True)

        with right:
            top_features = feature_importance.head(15).sort_values("importance", ascending=True)
            feature_fig = px.bar(
                top_features,
                x="importance",
                y="feature",
                orientation="h",
                title="Top Feature Drivers",
                color="importance",
                color_continuous_scale="YlGnBu",
            )
            feature_fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(feature_fig, use_container_width=True)

        st.info(
            (
                f"Rows: {evaluation['dataset']['rows']:,} | Columns: {evaluation['dataset']['columns']} | "
                f"Observed conversion rate: {_percent(evaluation['dataset']['positive_rate'])}"
            )
        )

    elif section == "Performance":
        c1, c2 = st.columns(2)

        with c1:
            cv_plot = cv_results.melt(
                id_vars="model",
                value_vars=["pr_auc_mean", "roc_auc_mean", "f1_mean"],
                var_name="metric",
                value_name="score",
            )
            cv_fig = px.bar(
                cv_plot,
                x="model",
                y="score",
                color="metric",
                barmode="group",
                title="Cross-Validation Comparison",
            )
            st.plotly_chart(cv_fig, use_container_width=True)

        with c2:
            roc_fig = px.line(
                roc_curve,
                x="fpr",
                y="tpr",
                title="ROC Curve",
                color_discrete_sequence=["#0284c7"],
            )
            roc_fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(color="#94a3b8", dash="dot"))
            st.plotly_chart(roc_fig, use_container_width=True)

        p1, p2 = st.columns(2)
        with p1:
            pr_fig = px.line(
                pr_curve,
                x="recall",
                y="precision",
                title="Precision-Recall Curve",
                color_discrete_sequence=["#0f766e"],
            )
            st.plotly_chart(pr_fig, use_container_width=True)

        with p2:
            cal_fig = go.Figure()
            cal_fig.add_trace(
                go.Scatter(
                    x=calibration_curve["avg_predicted_prob"],
                    y=calibration_curve["observed_conversion_rate"],
                    mode="lines+markers",
                    name="Model",
                    line=dict(color="#f97316", width=3),
                )
            )
            cal_fig.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode="lines",
                    name="Perfect",
                    line=dict(color="#64748b", dash="dash"),
                )
            )
            cal_fig.update_layout(
                title="Calibration Reliability Curve",
                xaxis_title="Mean Predicted Probability",
                yaxis_title="Observed Conversion Rate",
            )
            st.plotly_chart(cal_fig, use_container_width=True)

        cm1, cm2 = st.columns(2)
        with cm1:
            st.plotly_chart(
                _render_confusion_matrix(
                    evaluation["confusion_default_threshold"],
                    "Confusion Matrix (Threshold = 0.50)",
                ),
                use_container_width=True,
            )
        with cm2:
            st.plotly_chart(
                _render_confusion_matrix(
                    evaluation["confusion_optimal_threshold"],
                    f"Confusion Matrix (Threshold = {optimal_threshold:.2f})",
                ),
                use_container_width=True,
            )

    elif section == "Value & Lift":
        curve_fig = go.Figure()
        curve_fig.add_trace(
            go.Scatter(
                x=threshold_table["threshold"],
                y=threshold_table["expected_value"],
                mode="lines",
                name="Expected Value",
                line=dict(color="#0284c7", width=3),
            )
        )
        curve_fig.add_vline(
            x=optimal_threshold,
            line_dash="dash",
            line_color="#0f172a",
            annotation_text=f"Optimal {optimal_threshold:.2f}",
            annotation_position="top right",
        )
        curve_fig.update_layout(
            title="Threshold vs Expected Value",
            xaxis_title="Probability Threshold",
            yaxis_title="Expected Value (Relative Units)",
        )
        st.plotly_chart(curve_fig, use_container_width=True)

        selected_threshold = st.slider(
            "Explore threshold trade-off",
            min_value=0.05,
            max_value=0.95,
            value=float(round(optimal_threshold, 2)),
            step=0.01,
        )

        nearest = threshold_table.iloc[(threshold_table["threshold"] - selected_threshold).abs().argsort()[:1]]
        point = nearest.iloc[0]

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Targeted Sessions", f"{int(point['targeted_sessions']):,}")
        k2.metric("Precision", _percent(float(point["precision"])))
        k3.metric("Recall", _percent(float(point["recall"])))
        k4.metric("Expected Value", f"{float(point['expected_value']):,.1f}")

        d1, d2 = st.columns(2)
        with d1:
            decile_chart = px.bar(
                lift_table,
                x="decile",
                y="lift_vs_baseline",
                color="conversion_rate",
                title="Decile Lift (1 = highest intent)",
            )
            st.plotly_chart(decile_chart, use_container_width=True)

        with d2:
            capture_chart = px.line(
                lift_table,
                x="decile",
                y="cumulative_capture_rate",
                markers=True,
                title="Cumulative Conversion Capture",
            )
            capture_chart.update_yaxes(tickformat=".0%")
            st.plotly_chart(capture_chart, use_container_width=True)

    elif section == "Scenario Lab":
        st.subheader("Scenario Simulator")

        model = _cached_model()
        feature_cols = metadata["feature_columns"]
        source = predictions[feature_cols + ["predicted_probability"]].copy()

        profile_choice = st.selectbox("Base profile", PROFILE_OPTIONS, index=1)
        selected_row = _select_profile_row(source, profile_choice)
        baseline_payload = selected_row[feature_cols].to_frame().T.copy()

        scenario_features = _scenario_feature_list(source[feature_cols])
        adjustments = {}

        c1, c2, c3 = st.columns(3)
        cols = [c1, c2, c3]
        for idx, feature in enumerate(scenario_features):
            with cols[idx % 3]:
                adjustments[feature] = st.slider(
                    f"{feature} (%)",
                    min_value=-50,
                    max_value=50,
                    value=0,
                    step=5,
                    key=f"adj_{feature}",
                )

        adjusted_payload = baseline_payload.copy()
        for feature, pct in adjustments.items():
            original = float(adjusted_payload.iloc[0][feature])
            clamp_01 = feature.lower().endswith("rates") or feature.lower() in {"weekend", "specialday"}
            adjusted = _apply_percent_adjustment(original, pct, clamp_01=clamp_01)

            if pd.api.types.is_integer_dtype(source[feature]):
                adjusted_payload.loc[:, feature] = int(round(adjusted))
            else:
                adjusted_payload.loc[:, feature] = adjusted

        baseline_prob = _predict_probability(model, baseline_payload)
        adjusted_prob = _predict_probability(model, adjusted_payload)
        delta = adjusted_prob - baseline_prob

        g1, g2 = st.columns([1.2, 1])
        with g1:
            st.plotly_chart(
                _render_probability_gauge(adjusted_prob, float(metadata.get("optimal_threshold", 0.5))),
                use_container_width=True,
            )
        with g2:
            st.metric("Baseline Probability", _percent(baseline_prob))
            st.metric("Scenario Probability", _percent(adjusted_prob))
            st.metric("Delta", f"{delta:+.2%}")
            st.metric(
                "Suggested Action",
                "Likely converter" if adjusted_prob >= optimal_threshold else "Lower intent session",
            )

        sensitivity = _scenario_sensitivity_table(model, baseline_payload, scenario_features)
        if sensitivity.empty:
            st.info("No numeric features available for sensitivity analysis.")
        else:
            sensitivity_long = sensitivity.melt(
                id_vars="feature",
                value_vars=["delta_down", "delta_up"],
                var_name="direction",
                value_name="delta_probability",
            )
            sensitivity_long["direction"] = sensitivity_long["direction"].map(
                {"delta_down": "-10% move", "delta_up": "+10% move"}
            )
            tornado = px.bar(
                sensitivity_long,
                y="feature",
                x="delta_probability",
                color="direction",
                barmode="group",
                orientation="h",
                title="Local Sensitivity (one-at-a-time ±10%)",
                color_discrete_sequence=["#475569", "#0ea5e9"],
            )
            tornado.update_xaxes(tickformat="+.1%")
            st.plotly_chart(tornado, use_container_width=True)

    else:
        st.subheader("Downloads")

        left, right = st.columns(2)
        with left:
            st.markdown("### Executive Summary")
            st.text_area("Executive Summary", value=executive_summary, height=400)
            st.download_button(
                label="Download Executive Summary (.md)",
                data=executive_summary,
                file_name="executive_summary.md",
                mime="text/markdown",
            )

        with right:
            st.markdown("### Model Card")
            st.text_area("Model Card", value=model_card, height=400)
            st.download_button(
                label="Download Model Card (.md)",
                data=model_card,
                file_name="model_card.md",
                mime="text/markdown",
            )

        eval_payload = json.dumps(evaluation, indent=2)
        st.download_button(
            label="Download Evaluation JSON",
            data=eval_payload,
            file_name="evaluation_summary.json",
            mime="application/json",
        )


if __name__ == "__main__":
    main()
