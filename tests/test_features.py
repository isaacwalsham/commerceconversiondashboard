import pandas as pd

from commerceconversiondashboard.features import build_preprocessor, get_feature_lists


def test_feature_lists_identify_numeric_and_categorical():
    df = pd.DataFrame(
        {
            "num_1": [1.0, 2.0, 3.0],
            "num_2": [0, 1, 0],
            "cat_1": ["A", "B", "A"],
        }
    )

    numeric, categorical = get_feature_lists(df)

    assert "num_1" in numeric
    assert "num_2" in numeric
    assert "cat_1" in categorical


def test_preprocessor_transforms_dataframe():
    df = pd.DataFrame(
        {
            "num": [1.0, 2.0, 3.0, 4.0],
            "cat": ["x", "y", "x", "z"],
        }
    )

    preprocessor = build_preprocessor(df)
    transformed = preprocessor.fit_transform(df)

    assert transformed.shape[0] == len(df)
    assert transformed.shape[1] >= 2
