import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(heart_disease, target_column=None):
    """
    Load dataset from CSV and optionally binarize the target column.
    """
    df = pd.read_csv(heart_disease)
    if target_column is not None and target_column in df.columns:
        df[target_column] = (df[target_column] > 0).astype(int)
    return df


def clean_data(df, numeric_columns, categorical_columns):
    """
    Handle missing values in numeric and categorical columns.
    """
    df = df.copy()

    # Fill numeric missing values with median
    for col in numeric_columns:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Fill categorical missing values with mode
    for col in categorical_columns:
        if col in df.columns:
            mode_val = df[col].mode()
            if len(mode_val) > 0:
                df[col] = df[col].fillna(mode_val[0])

    return df


def handle_missing_values(df):
    """
    Remove rows with missing values from the DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    return df.dropna()


def encode_categorical_variables(df, columns=None):
    """
    One-hot encode categorical columns.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

    df = df.copy()

    if columns is None:
        columns = df.select_dtypes(include=[str, "category"]).columns.tolist()

    return pd.get_dummies(
        df,
        columns=columns,
        drop_first=True,
        dtype=int
    )


def preprocess_data(df, numeric_columns, categorical_columns):
    """
    Complete preprocessing pipeline.
    """
    df = clean_data(
        df,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns
    )

    df = encode_categorical_variables(
        df,
        columns=categorical_columns
    )

    return df


def split_data(df, target_column, test_size, random_state):
    """
    Split data into train and test sets.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )