import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from ml.models.inference import predict_from_artifact
from ml.models.persistence import save_model_artifact


def test_predict_from_classification_artifact(tmp_path):
    model = RandomForestClassifier(
        n_estimators=10,
        random_state=42,
    )

    training_data = pd.DataFrame(
        {
            "age": [20, 22, 30, 32],
            "income": [20000, 22000, 50000, 52000],
        }
    )

    target = [0, 0, 1, 1]

    preprocessing_pipeline = StandardScaler()

    preprocessing_pipeline.fit(
        training_data
    )

    transformed_training_data = (
        preprocessing_pipeline.transform(
            training_data
        )
    )

    model.fit(
        transformed_training_data,
        target,
    )

    artifact_path = (
        tmp_path / "classification_model.joblib"
    )

    save_model_artifact(
        model,
        preprocessing_pipeline,
        artifact_path,
    )

    input_data = pd.DataFrame(
        {
            "age": [21, 31],
            "income": [21000, 51000],
        }
    )

    predictions = predict_from_artifact(
        artifact_path,
        input_data,
    )

    assert len(predictions) == 2
    assert predictions[0] in {0, 1}
    assert predictions[1] in {0, 1}


def test_predict_from_regression_artifact(tmp_path):
    model = RandomForestRegressor(
        n_estimators=10,
        random_state=42,
    )

    training_data = pd.DataFrame(
        {
            "age": [20, 22, 30, 32],
            "experience": [1, 2, 10, 12],
        }
    )

    target = [20000, 22000, 50000, 52000]

    preprocessing_pipeline = StandardScaler()

    preprocessing_pipeline.fit(
        training_data
    )

    transformed_training_data = (
        preprocessing_pipeline.transform(
            training_data
        )
    )

    model.fit(
        transformed_training_data,
        target,
    )

    artifact_path = (
        tmp_path / "regression_model.joblib"
    )

    save_model_artifact(
        model,
        preprocessing_pipeline,
        artifact_path,
    )

    input_data = pd.DataFrame(
        {
            "age": [21, 31],
            "experience": [2, 11],
        }
    )

    predictions = predict_from_artifact(
        artifact_path,
        input_data,
    )

    assert len(predictions) == 2
    assert all(
        isinstance(
            prediction,
            float,
        )
        for prediction in predictions
    )


def test_predict_from_artifact_requires_dataframe(
    tmp_path,
):
    model = RandomForestClassifier(
        n_estimators=5,
        random_state=42,
    )

    preprocessing_pipeline = StandardScaler()

    training_data = [[1], [2], [3], [4]]
    target = [0, 0, 1, 1]

    preprocessing_pipeline.fit(
        training_data
    )

    transformed_training_data = (
        preprocessing_pipeline.transform(
            training_data
        )
    )

    model.fit(
        transformed_training_data,
        target,
    )

    artifact_path = (
        tmp_path / "model.joblib"
    )

    save_model_artifact(
        model,
        preprocessing_pipeline,
        artifact_path,
    )

    with pytest.raises(TypeError):
        predict_from_artifact(
            artifact_path,
            [[1]],
        )