import pandas as pd

from ml.models import (
    evaluate_classification,
    evaluate_regression,
)


class DummyClassificationModel:
    def predict(self, X):
        return pd.Series([0, 1, 1], index=X.index)


class DummyRegressionModel:
    def predict(self, X):
        return pd.Series([10.0, 20.0, 30.0], index=X.index)


def test_evaluate_classification():
    X_test = pd.DataFrame(
        {
            "feature": [1, 2, 3],
        }
    )

    y_test = pd.Series([0, 1, 1])

    result = evaluate_classification(
        DummyClassificationModel(),
        X_test,
        y_test,
    )

    assert result["accuracy"] == 1.0
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1"] == 1.0


def test_evaluate_regression():
    X_test = pd.DataFrame(
        {
            "feature": [1, 2, 3],
        }
    )

    y_test = pd.Series([10.0, 20.0, 30.0])

    result = evaluate_regression(
        DummyRegressionModel(),
        X_test,
        y_test,
    )

    assert result["mae"] == 0.0
    assert result["mse"] == 0.0
    assert result["rmse"] == 0.0
    assert result["r2"] == 1.0