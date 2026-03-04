"""Minimal fallback app for environment/browser troubleshooting."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"
EVAL_FILE = REPORTS_DIR / "evaluation_summary.json"
PRED_FILE = REPORTS_DIR / "test_predictions.csv"
FEATURE_FILE = REPORTS_DIR / "feature_importance.csv"

st.set_page_config(page_title="Conversion Predictor (Lite)", layout="wide")
st.title("Conversion Predictor (Lite)")

if not EVAL_FILE.exists():
    st.error("No trained artifacts found. Run: python scripts/train_model.py --data-path data/raw/online_shoppers_intention.csv")
    st.stop()

summary = json.loads(EVAL_FILE.read_text())
preds = pd.read_csv(PRED_FILE)
features = pd.read_csv(FEATURE_FILE)

m1, m2, m3 = st.columns(3)
metrics = summary["metrics_default_threshold"]
m1.metric("PR-AUC", f"{metrics['pr_auc']:.3f}")
m2.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
m3.metric("Optimal Threshold", f"{summary['optimal_threshold']:.2f}")

st.subheader("Probability Distribution")
fig = px.histogram(preds, x="predicted_probability", color="actual_conversion", nbins=30)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Top Features")
st.dataframe(features.head(20), use_container_width=True)

st.success("Lite app rendered successfully.")
