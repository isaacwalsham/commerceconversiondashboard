"""Feature preprocessing utilities."""

from __future__ import annotations

from typing import List, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


FeatureLists = Tuple[List[str], List[str]]


def get_feature_lists(df: pd.DataFrame) -> FeatureLists:
    """Return numeric and categorical feature names."""
    numeric_features = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [c for c in df.columns if c not in numeric_features]
    return numeric_features, categorical_features


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    """Build a robust preprocessing transformer."""
    numeric_features, categorical_features = get_feature_lists(df)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )
    return preprocessor
