import pandas as pd

from src.preprocess import preprocess_data


def test_missing_values_are_filled():

    df = pd.DataFrame({
        "age": [50, None, 70]
    })

    result = preprocess_data(df)

    assert result["age"].isnull().sum() == 0