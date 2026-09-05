import pandas as pd
import pytest

from ml.models import train_models


def test_trains_classification_models():
    X_train = pd.DataFrame(
        {
            "age": [20, 25, 30, 35, 40, 45],
            "salary": [20000, 25000, 30000, 35000, 40000, 45000],
        }
    )

    y_train = pd.Series([0, 0, 1, 1, 1, 0])

    models = train_models(
        X_train,
        y_train,
        "classification",
    )

    assert "logistic_regression" in models
    assert "random_forest" in models
    assert "xgboost" in models

    for model in models.values():
        assert hasattr(model, "predict")


def test_trains_regression_models():
    X_train = pd.DataFrame(
        {
            "age": [20, 25, 30, 35, 40, 45],
            "experience": [1, 3, 5, 7, 9, 11],
        }
    )

    y_train = pd.Series(
        [25000, 30000, 35000, 40000, 45000, 50000]
    )

    models = train_models(
        X_train,
        y_train,
        "regression",
    )

    assert "linear_regression" in models
    assert "random_forest" in models
    assert "xgboost" in models

    for model in models.values():
        assert hasattr(model, "predict")


def test_rejects_unknown_task_type():
    X_train = pd.DataFrame(
        {
            "age": [20, 30, 40],
        }
    )

    y_train = pd.Series([1, 2, 3])

    with pytest.raises(ValueError):
        train_models(
            X_train,
            y_train,
            "unknown",
        )