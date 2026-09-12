import pandas as pd

from ml.models import cross_validate_models
from ml.models.trainer import (
    get_classification_models,
    get_regression_models,
)


def test_cross_validates_classification_models():
    X = pd.DataFrame(
        {
            "age": [
                20, 21, 22, 25, 26, 27,
                30, 31, 32, 35, 36, 37,
                40, 41, 42, 45, 46, 47,
            ],
            "salary": [
                20000, 21000, 22000, 25000, 26000, 27000,
                30000, 31000, 32000, 35000, 36000, 37000,
                40000, 41000, 42000, 45000, 46000, 47000,
            ],
        }
    )

    y = pd.Series(
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    )

    models = get_classification_models()

    results = cross_validate_models(
        models,
        X,
        y,
        "classification",
        cv=3,
    )

    assert "logistic_regression" in results
    assert "random_forest" in results
    assert "xgboost" in results

    for metrics in results.values():
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics


def test_cross_validates_regression_models():
    X = pd.DataFrame(
        {
            "age": [
                20, 22, 24, 26, 28, 30,
                32, 34, 36, 38, 40, 42,
                44, 46, 48, 50, 52, 54,
            ],
            "experience": [
                1, 2, 3, 4, 5, 6,
                7, 8, 9, 10, 11, 12,
                13, 14, 15, 16, 17, 18,
            ],
        }
    )

    y = pd.Series(
        [
            20000, 22000, 24000, 26000, 28000, 30000,
            32000, 34000, 36000, 38000, 40000, 42000,
            44000, 46000, 48000, 50000, 52000, 54000,
        ]
    )

    models = get_regression_models()

    results = cross_validate_models(
        models,
        X,
        y,
        "regression",
        cv=3,
    )

    assert "linear_regression" in results
    assert "random_forest" in results
    assert "xgboost" in results

    for metrics in results.values():
        assert "mae" in metrics
        assert "mse" in metrics
        assert "rmse" in metrics
        assert "r2" in metrics