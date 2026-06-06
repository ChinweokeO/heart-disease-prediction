import pandas as pd
from pathlib import Path


def _find_dataset_file():
    repo_root = Path(__file__).resolve().parents[1]
    candidate_patterns = ["*heart*.csv", "*heart*.txt"]
    dataset_paths = []
    for pattern in candidate_patterns:
        dataset_paths.extend(repo_root.rglob(pattern))
    if not dataset_paths:
        raise FileNotFoundError(
            "Could not find a heart disease dataset file in the repository root. "
            "Expected a CSV or TXT containing 'heart' in the name."
        )
    return dataset_paths[0]


def _load_dataset():
    dataset_path = _find_dataset_file()
    return pd.read_csv(dataset_path)


def _infer_target_column(df):
    for candidate in ("target", "num", "heart_disease", "disease", "class"):
        if candidate in df.columns:
            return candidate
    raise ValueError(
        "No target column found in dataset. Expected one of "
        "'target', 'num', 'heart_disease', 'disease', or 'class'."
    )


def test_expected_columns_present():
    df = _load_dataset()
    expected_columns = {
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal",
    }
    missing = sorted(expected_columns.difference(df.columns))
    assert not missing, f"Dataset is missing expected columns: {missing}"


def test_target_variable_expected_values():
    df = _load_dataset()
    target_column = _infer_target_column(df)
    values = sorted(df[target_column].dropna().unique())
    assert values, f"Target column '{target_column}' is empty."
    allowed_values = {0, 1, 2, 3, 4}
    assert set(values).issubset(allowed_values), (
        f"Target column '{target_column}' contains unexpected values {values}. "
        f"Expected subset of {sorted(allowed_values)}."
    )
    assert all(float(val).is_integer() for val in values), (
        f"Target column '{target_column}' contains non-integer classification labels: {values}."
    )


def test_numeric_features_within_expected_ranges():
    df = _load_dataset()
    expected_ranges = {
        "age": (20, 90),
        "trestbps": (80, 220),
        "chol": (100, 600),
        "thalach": (50, 260),
        "oldpeak": (0.0, 6.5),
    }
    for col, (min_value, max_value) in expected_ranges.items():
        assert col in df.columns, f"Expected numeric feature '{col}' is not present."
        col_min = float(df[col].min())
        col_max = float(df[col].max())
        assert min_value <= col_min <= max_value, (
            f"Column '{col}' has minimum value {col_min} outside expected range "
            f"[{min_value}, {max_value}] ."
        )
        assert min_value <= col_max <= max_value, (
            f"Column '{col}' has maximum value {col_max} outside expected range "
            f"[{min_value}, {max_value}] ."
        )
