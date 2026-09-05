import pandas as pd
import pytest

from ml.optimization import optimize_xgboost


def test_optimize_xgboost_classification():
    X = pd.DataFrame(
        {
            "age": [
                20,
                22,
                24,
                26,
                28,
                30,
                32,
                34,
                36,
                38,
                40,
                42,
                44,
                46,
                48,
                50,
                52,
                54,
                56,
                58,
            ],
            "salary": [
                20000,
                22000,
                24000,
                26000,
                28000,
                30000,
                32000,
                34000,
                36000,
                38000,
                40000,
                42000,
                44000,
                46000,
                48000,
                50000,
                52000,
                54000,
                56000,
                58000,
            ],
        }
    )

    y = pd.Series(
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
        ]
    )

    result = optimize_xgboost(
        X,
        y,
        "classification",
        n_trials=2,
        cv=3,
    )

    assert "model" in result
    assert "best_params" in result
    assert "best_score" in result
    assert "n_trials" in result

    assert result["n_trials"] == 2
    assert result["best_score"] >= 0
    assert hasattr(result["model"], "predict")


def test_optimize_xgboost_regression():
    X = pd.DataFrame(
        {
            "age": [
                20,
                22,
                24,
                26,
                28,
                30,
                32,
                34,
                36,
                38,
                40,
                42,
                44,
                46,
                48,
                50,
                52,
                54,
                56,
                58,
            ],
            "experience": [
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
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
            ],
        }
    )

    y = pd.Series(
        [
            20000,
            22000,
            24000,
            26000,
            28000,
            30000,
            32000,
            34000,
            36000,
            38000,
            40000,
            42000,
            44000,
            46000,
            48000,
            50000,
            52000,
            54000,
            56000,
            58000,
        ]
    )

    result = optimize_xgboost(
        X,
        y,
        "regression",
        n_trials=2,
        cv=3,
    )

    assert "model" in result
    assert "best_params" in result
    assert "best_score" in result
    assert "n_trials" in result

    assert result["n_trials"] == 2
    assert result["best_score"] <= 0
    assert hasattr(result["model"], "predict")


def test_xgboost_optimizer_rejects_unknown_task():
    X = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4, 5, 6],
        }
    )

    y = pd.Series([0, 1, 0, 1, 0, 1])

    with pytest.raises(ValueError):
        optimize_xgboost(
            X,
            y,
            "unknown",
            n_trials=2,
            cv=2,
        )


def test_xgboost_optimizer_rejects_invalid_trials():
    X = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4, 5, 6],
        }
    )

    y = pd.Series([0, 1, 0, 1, 0, 1])

    with pytest.raises(ValueError):
        optimize_xgboost(
            X,
            y,
            "classification",
            n_trials=0,
            cv=2,
        )


def test_xgboost_optimizer_rejects_invalid_cv():
    X = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4, 5, 6],
        }
    )

    y = pd.Series([0, 1, 0, 1, 0, 1])

    with pytest.raises(ValueError):
        optimize_xgboost(
            X,
            y,
            "classification",
            n_trials=2,
            cv=1,
        )