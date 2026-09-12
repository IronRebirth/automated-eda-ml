import pytest
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
)


def test_select_best_classification_model():
    from ml.models.model_selection import select_best_model

    optimized_models = {
        "random_forest": {
            "model": RandomForestClassifier(
                n_estimators=10,
                random_state=42,
            )
        },
        "xgboost": {
            "model": RandomForestClassifier(
                n_estimators=20,
                random_state=42,
            )
        },
    }

    optimized_evaluation = {
        "random_forest": {
            "accuracy": 0.90,
            "precision": 0.89,
            "recall": 0.90,
            "f1": 0.895,
        },
        "xgboost": {
            "accuracy": 0.95,
            "precision": 0.94,
            "recall": 0.95,
            "f1": 0.945,
        },
    }

    result = select_best_model(
        optimized_models,
        optimized_evaluation,
        "classification",
    )

    assert result["model_name"] == "xgboost"
    assert result["metric"] == "f1"
    assert result["score"] == 0.945
    assert result["model"] is optimized_models["xgboost"]["model"]


def test_select_best_regression_model():
    from ml.models.model_selection import select_best_model

    optimized_models = {
        "random_forest": {
            "model": RandomForestRegressor(
                n_estimators=10,
                random_state=42,
            )
        },
        "xgboost": {
            "model": RandomForestRegressor(
                n_estimators=20,
                random_state=42,
            )
        },
    }

    optimized_evaluation = {
        "random_forest": {
            "mae": 10.0,
            "mse": 150.0,
            "rmse": 12.25,
            "r2": 0.80,
        },
        "xgboost": {
            "mae": 8.0,
            "mse": 100.0,
            "rmse": 10.0,
            "r2": 0.90,
        },
    }

    result = select_best_model(
        optimized_models,
        optimized_evaluation,
        "regression",
    )

    assert result["model_name"] == "xgboost"
    assert result["metric"] == "rmse"
    assert result["score"] == 10.0
    assert result["model"] is optimized_models["xgboost"]["model"]


def test_select_best_model_rejects_unknown_task():
    from ml.models.model_selection import select_best_model

    with pytest.raises(ValueError):
        select_best_model(
            {},
            {},
            "unknown",
        )


def test_select_best_model_rejects_empty_evaluation():
    from ml.models.model_selection import select_best_model

    with pytest.raises(ValueError):
        select_best_model(
            {},
            {},
            "classification",
        )


def test_select_best_model_rejects_missing_metric():
    from ml.models.model_selection import select_best_model

    optimized_models = {
        "random_forest": {
            "model": RandomForestClassifier(
                n_estimators=10,
                random_state=42,
            )
        }
    }

    optimized_evaluation = {
        "random_forest": {
            "accuracy": 0.90,
        }
    }

    with pytest.raises(ValueError):
        select_best_model(
            optimized_models,
            optimized_evaluation,
            "classification",
        )