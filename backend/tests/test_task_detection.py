import pandas as pd
import pytest

from ml.pipeline import detect_task_type


def test_detects_binary_classification():
    dataframe = pd.DataFrame(
        {
            "age": [20, 30, 40, 50],
            "churn": [0, 1, 0, 1],
        }
    )

    result = detect_task_type(
        dataframe,
        "churn",
    )

    assert result == "classification"


def test_detects_categorical_classification():
    dataframe = pd.DataFrame(
        {
            "age": [20, 30, 40],
            "segment": ["A", "B", "A"],
        }
    )

    result = detect_task_type(
        dataframe,
        "segment",
    )

    assert result == "classification"


def test_detects_regression():
    dataframe = pd.DataFrame(
        {
            "age": [20, 30, 40, 50],
            "salary": [25000.5, 32000.2, 41000.8, 55000.4],
        }
    )

    result = detect_task_type(
        dataframe,
        "salary",
    )

    assert result == "regression"


def test_raises_error_for_missing_target():
    dataframe = pd.DataFrame(
        {
            "age": [20, 30, 40],
        }
    )

    with pytest.raises(ValueError):
        detect_task_type(
            dataframe,
            "churn",
        )