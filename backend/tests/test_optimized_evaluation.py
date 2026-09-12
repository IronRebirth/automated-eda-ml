import pandas as pd
import pytest
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
)

from ml.models import evaluate_optimized_models


def test_evaluate_optimized_classification_models():
    X_train = pd.DataFrame(
        {
            "feature": [
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
            ]
        }
    )

    y_train = pd.Series(
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    )

    model = RandomForestClassifier(
        n_estimators=10,
        random_state=42,
    )

    optimization_results = {
        "random_forest": {
            "model": model,
            "best_params": {},
            "best_score": 1.0,
            "n_trials": 2,
        }
    }

    X_test = pd.DataFrame(
        {
            "feature": [2, 8, 3, 9],
        }
    )

    y_test = pd.Series([0, 1, 0, 1])

    results = evaluate_optimized_models(
        optimization_results,
        X_train,
        y_train,
        X_test,
        y_test,
        "classification",
    )

    assert "random_forest" in results
    assert "accuracy" in results["random_forest"]
    assert "precision" in results["random_forest"]
    assert "recall" in results["random_forest"]
    assert "f1" in results["random_forest"]


def test_evaluate_optimized_regression_models():
    X_train = pd.DataFrame(
        {
            "feature": [
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
            ]
        }
    )

    y_train = pd.Series(
        [
            10,
            20,
            30,
            40,
            50,
            60,
            70,
            80,
            90,
            100,
        ]
    )

    model = RandomForestRegressor(
        n_estimators=10,
        random_state=42,
    )

    optimization_results = {
        "random_forest": {
            "model": model,
            "best_params": {},
            "best_score": -1.0,
            "n_trials": 2,
        }
    }

    X_test = pd.DataFrame(
        {
            "feature": [2, 8, 3, 9],
        }
    )

    y_test = pd.Series([20, 80, 30, 90])

    results = evaluate_optimized_models(
        optimization_results,
        X_train,
        y_train,
        X_test,
        y_test,
        "regression",
    )

    assert "random_forest" in results
    assert "mae" in results["random_forest"]
    assert "mse" in results["random_forest"]
    assert "rmse" in results["random_forest"]
    assert "r2" in results["random_forest"]


def test_evaluate_optimized_models_rejects_unknown_task():
    optimization_results = {}

    with pytest.raises(ValueError):
        evaluate_optimized_models(
            optimization_results,
            pd.DataFrame({"feature": [1, 2]}),
            pd.Series([0, 1]),
            pd.DataFrame({"feature": [3, 4]}),
            pd.Series([0, 1]),
            "unknown",
        )