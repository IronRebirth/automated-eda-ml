import pytest

from ml.models import build_model_leaderboard


def test_classification_leaderboard():
    evaluation_results = {
        "model_a": {
            "accuracy": 0.80,
            "precision": 0.78,
            "recall": 0.79,
            "f1": 0.85,
        },
        "model_b": {
            "accuracy": 0.90,
            "precision": 0.88,
            "recall": 0.89,
            "f1": 0.75,
        },
    }

    leaderboard = build_model_leaderboard(
        evaluation_results,
        "classification",
    )

    assert leaderboard[0]["model"] == "model_a"
    assert leaderboard[0]["rank"] == 1
    assert leaderboard[1]["rank"] == 2


def test_regression_leaderboard():
    evaluation_results = {
        "model_a": {
            "mae": 1000,
            "mse": 1500000,
            "rmse": 1224.74,
            "r2": 0.90,
        },
        "model_b": {
            "mae": 2000,
            "mse": 4000000,
            "rmse": 2000,
            "r2": 0.80,
        },
    }

    leaderboard = build_model_leaderboard(
        evaluation_results,
        "regression",
    )

    assert leaderboard[0]["model"] == "model_a"
    assert leaderboard[0]["rank"] == 1
    assert leaderboard[1]["rank"] == 2


def test_leaderboard_includes_cross_validation_results():
    evaluation_results = {
        "model_a": {
            "accuracy": 0.90,
            "precision": 0.88,
            "recall": 0.89,
            "f1": 0.91,
        },
        "model_b": {
            "accuracy": 0.85,
            "precision": 0.83,
            "recall": 0.84,
            "f1": 0.86,
        },
    }

    cross_validation_results = {
        "model_a": {
            "accuracy": 0.88,
            "accuracy_std": 0.02,
            "precision": 0.87,
            "precision_std": 0.02,
            "recall": 0.88,
            "recall_std": 0.03,
            "f1": 0.89,
            "f1_std": 0.02,
        },
        "model_b": {
            "accuracy": 0.83,
            "accuracy_std": 0.03,
            "precision": 0.82,
            "precision_std": 0.03,
            "recall": 0.83,
            "recall_std": 0.04,
            "f1": 0.84,
            "f1_std": 0.03,
        },
    }

    leaderboard = build_model_leaderboard(
        evaluation_results,
        "classification",
        cross_validation_results,
    )

    assert leaderboard[0]["cv_score"] == pytest.approx(0.89)
    assert leaderboard[0]["cv_std"] == pytest.approx(0.02)
    assert "cv_metrics" in leaderboard[0]


def test_leaderboard_rejects_unknown_task():
    evaluation_results = {
        "model_a": {
            "f1": 0.90,
        },
    }

    with pytest.raises(ValueError):
        build_model_leaderboard(
            evaluation_results,
            "unknown",
        )