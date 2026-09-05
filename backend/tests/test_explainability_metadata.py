import pandas as pd
import pytest

from ml.explainability.metadata import (
    build_explainability_metadata,
)


def build_feature_importance() -> pd.DataFrame:
    """Build feature importance data for testing."""

    return pd.DataFrame(
        {
            "feature": [
                "age",
                "income",
                "tenure",
            ],
            "importance": [
                0.8,
                0.5,
                0.2,
            ],
        }
    )


def test_build_explainability_metadata():
    feature_importance = (
        build_feature_importance()
    )

    metadata = build_explainability_metadata(
        feature_importance,
        model_name="xgboost",
        task_type="classification",
        preprocessing_applied=True,
        top_features_count=2,
    )

    assert metadata == {
        "method": "SHAP",
        "model": "xgboost",
        "task_type": "classification",
        "feature_count": 3,
        "top_features_count": 2,
        "preprocessing_applied": True,
    }


def test_metadata_defaults_to_all_features():
    feature_importance = (
        build_feature_importance()
    )

    metadata = build_explainability_metadata(
        feature_importance,
        model_name="random_forest",
        task_type="regression",
    )

    assert metadata["feature_count"] == 3
    assert metadata["top_features_count"] == 3


def test_metadata_caps_top_features_count():
    feature_importance = (
        build_feature_importance()
    )

    metadata = build_explainability_metadata(
        feature_importance,
        model_name="xgboost",
        task_type="classification",
        top_features_count=10,
    )

    assert metadata["top_features_count"] == 3


def test_metadata_rejects_empty_feature_importance():
    feature_importance = pd.DataFrame(
        columns=[
            "feature",
            "importance",
        ]
    )

    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        build_explainability_metadata(
            feature_importance,
            model_name="xgboost",
            task_type="classification",
        )


def test_metadata_rejects_invalid_model_name():
    feature_importance = (
        build_feature_importance()
    )

    with pytest.raises(
        ValueError,
        match="model_name must not be empty",
    ):
        build_explainability_metadata(
            feature_importance,
            model_name="",
            task_type="classification",
        )


def test_metadata_rejects_invalid_task_type():
    feature_importance = (
        build_feature_importance()
    )

    with pytest.raises(
        ValueError,
        match="Unsupported task type",
    ):
        build_explainability_metadata(
            feature_importance,
            model_name="xgboost",
            task_type="clustering",
        )


def test_metadata_rejects_invalid_top_features_count():
    feature_importance = (
        build_feature_importance()
    )

    with pytest.raises(
        ValueError,
        match="must be at least 1",
    ):
        build_explainability_metadata(
            feature_importance,
            model_name="xgboost",
            task_type="classification",
            top_features_count=0,
        )