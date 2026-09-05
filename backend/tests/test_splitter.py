import pandas as pd
import pytest

from ml.pipeline import split_features_target


def test_split_features_and_target():
    dataframe = pd.DataFrame(
        {
            "age": [20, 30, 40, 50, 60],
            "salary": [20000, 30000, 40000, 50000, 60000],
            "churn": [0, 1, 0, 1, 0],
        }
    )

    X_train, X_test, y_train, y_test = split_features_target(
        dataframe,
        "churn",
        test_size=0.2,
        random_state=42,
    )

    assert "churn" not in X_train.columns
    assert "churn" not in X_test.columns

    assert len(X_train) == 4
    assert len(X_test) == 1

    assert len(y_train) == 4
    assert len(y_test) == 1


def test_split_is_reproducible():
    dataframe = pd.DataFrame(
        {
            "age": range(20, 30),
            "churn": [0, 1] * 5,
        }
    )

    first_split = split_features_target(
        dataframe,
        "churn",
        random_state=42,
    )

    second_split = split_features_target(
        dataframe,
        "churn",
        random_state=42,
    )

    assert first_split[0].equals(second_split[0])
    assert first_split[1].equals(second_split[1])
    assert first_split[2].equals(second_split[2])
    assert first_split[3].equals(second_split[3])


def test_split_raises_error_for_missing_target():
    dataframe = pd.DataFrame(
        {
            "age": [20, 30, 40],
        }
    )

    with pytest.raises(ValueError):
        split_features_target(
            dataframe,
            "churn",
        )


def test_split_raises_error_for_invalid_test_size():
    dataframe = pd.DataFrame(
        {
            "age": [20, 30, 40],
            "churn": [0, 1, 0],
        }
    )

    with pytest.raises(ValueError):
        split_features_target(
            dataframe,
            "churn",
            test_size=1.5,
        )