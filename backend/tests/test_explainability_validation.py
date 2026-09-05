import pandas as pd
import pytest

from ml.explainability.validation import (
    validate_explainability_output,
)


def build_valid_output() -> dict:
    """Build a valid explainability output for testing."""

    feature_importance = pd.DataFrame(
        {
            "feature": [
                "age",
                "income",
            ],
            "importance": [
                0.8,
                0.4,
            ],
        }
    )

    transformed_data = pd.DataFrame(
        {
            "age": [25.0, 30.0],
            "income": [50000.0, 60000.0],
        }
    )

    return {
        "feature_importance": feature_importance,
        "shap_values": object(),
        "transformed_data": transformed_data,
        "feature_names": [
            "age",
            "income",
        ],
        "summary": {
            "top_n": 2,
            "features": [
                {
                    "rank": 1,
                    "feature": "age",
                    "importance": 0.8,
                    "relative_importance": 1.0,
                    "impact": "high",
                },
                {
                    "rank": 2,
                    "feature": "income",
                    "importance": 0.4,
                    "relative_importance": 0.5,
                    "impact": "medium",
                },
            ],
        },
        "insights": [
            "'age' is the most influential feature in the model.",
        ],
        "metadata": {
            "method": "SHAP",
            "model": "xgboost",
            "task_type": "classification",
            "feature_count": 2,
            "top_features_count": 2,
            "preprocessing_applied": True,
        },
    }


def test_validate_explainability_output_accepts_valid_output():
    output = build_valid_output()

    validate_explainability_output(output)


def test_validation_rejects_missing_required_keys():
    output = build_valid_output()
    del output["summary"]

    with pytest.raises(
        ValueError,
        match="missing required keys",
    ):
        validate_explainability_output(output)


def test_validation_rejects_negative_importance():
    output = build_valid_output()

    output["feature_importance"].loc[
        0,
        "importance",
    ] = -0.1

    with pytest.raises(
        ValueError,
        match="non-negative",
    ):
        validate_explainability_output(output)


def test_validation_rejects_feature_name_mismatch():
    output = build_valid_output()

    output["feature_names"] = [
        "age",
        "wrong_feature",
    ]

    with pytest.raises(
        ValueError,
        match="columns must match",
    ):
        validate_explainability_output(output)


def test_validation_rejects_invalid_summary_rank():
    output = build_valid_output()

    output["summary"]["features"][0]["rank"] = 2

    with pytest.raises(
        ValueError,
        match="ranks must start at 1",
    ):
        validate_explainability_output(output)


def test_validation_rejects_invalid_summary_impact():
    output = build_valid_output()

    output["summary"]["features"][0]["impact"] = (
        "critical"
    )

    with pytest.raises(
        ValueError,
        match="impact must be",
    ):
        validate_explainability_output(output)


def test_validation_rejects_empty_insights():
    output = build_valid_output()

    output["insights"] = []

    with pytest.raises(
        ValueError,
        match="insights must not be empty",
    ):
        validate_explainability_output(output)


def test_validation_rejects_invalid_insight_value():
    output = build_valid_output()

    output["insights"] = [
        "",
    ]

    with pytest.raises(
        ValueError,
        match="non-empty string",
    ):
        validate_explainability_output(output)