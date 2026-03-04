"""Inference helpers for app and scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import joblib
import pandas as pd

from .paths import METADATA_FILE, MODEL_FILE


def load_model(model_path: Path = MODEL_FILE):
    """Load serialized model pipeline."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model file missing: {model_path}")
    return joblib.load(model_path)


def load_metadata(metadata_path: Path = METADATA_FILE) -> Dict[str, object]:
    """Load model metadata for app defaults."""
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file missing: {metadata_path}")
    return json.loads(metadata_path.read_text())


def predict_probabilities(df: pd.DataFrame):
    """Generate conversion probabilities for input dataframe."""
    model = load_model()
    probabilities = model.predict_proba(df)[:, 1]
    return probabilities
