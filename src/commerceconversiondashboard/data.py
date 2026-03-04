"""Data ingestion and preprocessing utilities."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import pandas as pd
import requests
from requests import RequestException
from sklearn.model_selection import train_test_split

from .paths import (
    DATA_BUNDLED_DIR,
    DATA_PROCESSED_DIR,
    DATA_RAW_DIR,
    DATASET_MANIFEST_FILE,
    RAW_DATA_FILE,
    TEST_DATA_FILE,
    TRAIN_DATA_FILE,
)

DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00468/online_shoppers_intention.csv"
MONTH_ORDER = ["Jan", "Feb", "Mar", "Apr", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


@dataclass(frozen=True)
class SplitData:
    """Container for train/test splits."""

    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


@dataclass(frozen=True)
class BundledDataset:
    """Metadata for built-in project datasets."""

    dataset_id: str
    name: str
    file: str
    description: str

    @property
    def path(self) -> Path:
        return DATA_BUNDLED_DIR / self.file


def ensure_data_dirs() -> None:
    """Create required data folders if missing."""
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    DATA_BUNDLED_DIR.mkdir(parents=True, exist_ok=True)


def list_bundled_datasets() -> list[BundledDataset]:
    """Return available built-in dataset options."""
    ensure_data_dirs()

    if DATASET_MANIFEST_FILE.exists():
        payload = json.loads(DATASET_MANIFEST_FILE.read_text())
        datasets = []
        for entry in payload.get("datasets", []):
            if not all(key in entry for key in ["id", "name", "file", "description"]):
                continue
            datasets.append(
                BundledDataset(
                    dataset_id=str(entry["id"]),
                    name=str(entry["name"]),
                    file=str(entry["file"]),
                    description=str(entry["description"]),
                )
            )
        return [d for d in datasets if d.path.exists()]

    csv_files = sorted(DATA_BUNDLED_DIR.glob("*.csv"))
    return [
        BundledDataset(
            dataset_id=path.stem,
            name=path.stem.replace("_", " ").title(),
            file=path.name,
            description="Bundled dataset file.",
        )
        for path in csv_files
    ]


def bundled_dataset_path(dataset_id: str) -> Path:
    """Resolve a bundled dataset id to an existing file path."""
    for dataset in list_bundled_datasets():
        if dataset.dataset_id == dataset_id:
            return dataset.path
    raise ValueError(f"Unknown bundled dataset id: {dataset_id}")


def download_online_shoppers_dataset(force: bool = False, destination: Path = RAW_DATA_FILE) -> Path:
    """Download the UCI Online Shoppers dataset if missing."""
    ensure_data_dirs()
    if destination.exists() and not force:
        return destination

    try:
        response = requests.get(DATASET_URL, timeout=60)
        response.raise_for_status()
        destination.write_bytes(response.content)
        return destination
    except RequestException as exc:
        message = (
            "Could not download the dataset automatically. "
            f"Please place `online_shoppers_intention.csv` at `{destination}` "
            "or pass a local file path into the training script."
        )
        raise RuntimeError(message) from exc


def _normalize_boolean(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.astype(int)

    normalized = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": 1, "false": 0, "1": 1, "0": 0, "yes": 1, "no": 0})
    )
    return normalized.fillna(0).astype(int)


def load_dataset(path: Path = RAW_DATA_FILE) -> pd.DataFrame:
    """Read and normalize raw dataset."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    df = pd.read_csv(path)
    if "Revenue" not in df.columns:
        raise ValueError("Expected column 'Revenue' in dataset")

    if "Weekend" in df.columns:
        df["Weekend"] = _normalize_boolean(df["Weekend"])

    df["Revenue"] = _normalize_boolean(df["Revenue"])

    if "Month" in df.columns:
        df["Month"] = pd.Categorical(df["Month"], categories=MONTH_ORDER, ordered=True)

    if "VisitorType" in df.columns:
        df["VisitorType"] = df["VisitorType"].astype(str).str.strip()

    return df


def split_features_target(
    df: pd.DataFrame,
    target_column: str = "Revenue",
    test_size: float = 0.2,
    random_state: int = 42,
) -> SplitData:
    """Split dataframe into stratified train/test feature and target sets."""
    x = df.drop(columns=[target_column])
    y = df[target_column].astype(int)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return SplitData(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)


def persist_split_data(split_data: SplitData) -> None:
    """Write train/test CSVs to disk for reproducibility."""
    ensure_data_dirs()

    train_df = split_data.x_train.copy()
    train_df["Revenue"] = split_data.y_train

    test_df = split_data.x_test.copy()
    test_df["Revenue"] = split_data.y_test

    train_df.to_csv(TRAIN_DATA_FILE, index=False)
    test_df.to_csv(TEST_DATA_FILE, index=False)
