import pandas as pd
import pytest

from src.preprocessing import (
    encode_categorical_variables,
    handle_missing_values,
)


def test_handle_missing_values_removes_missing_rows():
    input_df = pd.DataFrame(
        {
            "age": [25, None, 45],
            "sex": ["M", "F", "M"],
            "cholesterol": [200, 180, None],
        }
    )

    result = handle_missing_values(input_df)

    expected = input_df.dropna().reset_index(drop=True)
    pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)


def test_handle_missing_values_leaves_non_missing_data_unchanged():
    input_df = pd.DataFrame(
        {
            "age": [30, 50],
            "sex": ["F", "M"],
            "cholesterol": [180, 210],
        }
    )

    result = handle_missing_values(input_df)

    pd.testing.assert_frame_equal(result, input_df)


def test_encode_categorical_variables_creates_dummy_columns():
    input_df = pd.DataFrame(
        {
            "sex": ["M", "F", "M"],
            "chest_pain": ["typical", "asymptomatic", "nonanginal"],
            "age": [55, 60, 45],
        }
    )

    result = encode_categorical_variables(input_df)

    assert "sex" not in result.columns
    assert "chest_pain" not in result.columns
    assert any(col.startswith("sex_") for col in result.columns)
    assert any(col.startswith("chest_pain_") for col in result.columns)
    assert result["age"].tolist() == [55, 60, 45]


def test_encode_categorical_variables_preserves_numeric_columns():
    input_df = pd.DataFrame(
        {
            "sex": ["F", "M"],
            "age": [40, 70],
            "cholesterol": [190, 230],
        }
    )

    result = encode_categorical_variables(input_df)

    assert result["age"].tolist() == [40, 70]
    assert result["cholesterol"].tolist() == [190, 230]


def test_handle_missing_values_does_not_modify_original_dataframe():
    input_df = pd.DataFrame(
        {
            "age": [22, None],
            "sex": ["M", "F"],
        }
    )
    original_copy = input_df.copy(deep=True)

    _ = handle_missing_values(input_df)

    pd.testing.assert_frame_equal(input_df, original_copy)


def test_encode_categorical_variables_raises_type_error_for_non_dataframe_input():
    with pytest.raises(TypeError):
        encode_categorical_variables([{"sex": "M"}])


def test_handle_missing_values_raises_type_error_for_invalid_input():
    with pytest.raises(TypeError):
        handle_missing_values("not a dataframe")
