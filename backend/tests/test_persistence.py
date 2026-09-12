from pathlib import Path

import joblib
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from ml.models.persistence import (
    load_model_artifact,
    save_model_artifact,
)


def test_save_and_load_model_artifact(tmp_path):
    model = RandomForestClassifier(
        n_estimators=10,
        random_state=42,
    )

    model.fit(
        [[1], [2], [3], [4]],
        [0, 0, 1, 1],
    )

    preprocessing_pipeline = StandardScaler()

    preprocessing_pipeline.fit(
        [[1], [2], [3], [4]],
    )

    artifact_path = tmp_path / "model.joblib"

    saved_path = save_model_artifact(
        model,
        preprocessing_pipeline,
        artifact_path,
    )

    assert saved_path == str(artifact_path)
    assert artifact_path.exists()

    artifact = load_model_artifact(
        artifact_path,
    )

    assert isinstance(artifact, dict)
    assert "model" in artifact
    assert "preprocessing" in artifact

    assert isinstance(
        artifact["model"],
        RandomForestClassifier,
    )

    assert isinstance(
        artifact["preprocessing"],
        StandardScaler,
    )

    predictions = artifact["model"].predict(
        [[1], [4]],
    )

    assert len(predictions) == 2


def test_save_model_creates_parent_directories(tmp_path):
    model = RandomForestClassifier(
        n_estimators=5,
        random_state=42,
    )

    model.fit(
        [[1], [2], [3], [4]],
        [0, 0, 1, 1],
    )

    preprocessing_pipeline = StandardScaler()

    preprocessing_pipeline.fit(
        [[1], [2], [3], [4]],
    )

    artifact_path = (
        tmp_path
        / "nested"
        / "artifacts"
        / "model.joblib"
    )

    save_model_artifact(
        model,
        preprocessing_pipeline,
        artifact_path,
    )

    assert artifact_path.exists()


def test_load_model_rejects_missing_artifact(tmp_path):
    artifact_path = (
        tmp_path / "missing_model.joblib"
    )

    with pytest.raises(FileNotFoundError):
        load_model_artifact(
            artifact_path,
        )


def test_load_model_rejects_invalid_artifact(
    tmp_path,
):
    artifact_path = (
        tmp_path / "invalid_model.joblib"
    )

    joblib.dump(
        "invalid artifact",
        artifact_path,
    )

    with pytest.raises(ValueError):
        load_model_artifact(
            artifact_path,
        )


def test_load_model_rejects_incomplete_artifact(
    tmp_path,
):
    artifact_path = (
        tmp_path / "incomplete_model.joblib"
    )

    joblib.dump(
        {"model": "model only"},
        artifact_path,
    )

    with pytest.raises(ValueError):
        load_model_artifact(
            artifact_path,
        )


def test_save_model_returns_string_path(tmp_path):
    model = RandomForestClassifier(
        n_estimators=5,
        random_state=42,
    )

    model.fit(
        [[1], [2], [3], [4]],
        [0, 0, 1, 1],
    )

    preprocessing_pipeline = StandardScaler()

    preprocessing_pipeline.fit(
        [[1], [2], [3], [4]],
    )

    artifact_path = Path(
        tmp_path / "model.joblib"
    )

    result = save_model_artifact(
        model,
        preprocessing_pipeline,
        artifact_path,
    )

    assert isinstance(result, str)
    assert result == str(artifact_path)