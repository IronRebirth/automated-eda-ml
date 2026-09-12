import pytest

from ml.reporting import validate_serialized_report


def build_valid_report() -> dict:
    """Build a valid serialized analysis report."""

    return {
        "dataset": {
            "shape": {
                "rows": 100,
                "columns": 5,
            },
            "target_column": "churn",
        },
        "data_quality": None,
        "eda": None,
        "modeling": {
            "task_type": "classification",
            "evaluation": {},
            "cross_validation": {},
            "leaderboard": [],
            "optimization": {},
            "optimized_evaluation": {},
            "best_model": {},
        },
        "explainability": {
            "metadata": {
                "method": "SHAP",
            },
            "summary": {
                "top_n": 2,
            },
            "insights": [
                "age is influential."
            ],
            "feature_importance": [
                {
                    "feature": "age",
                    "importance": 0.8,
                }
            ],
        },
        "artifact": {
            "path": None,
            "available": False,
        },
    }


def test_validate_serialized_report_accepts_valid_report():
    report = build_valid_report()

    validate_serialized_report(report)


def test_validation_rejects_invalid_report_type():
    with pytest.raises(
        TypeError,
        match="report must be a dictionary",
    ):
        validate_serialized_report([])


def test_validation_rejects_missing_section():
    report = build_valid_report()

    del report["modeling"]

    with pytest.raises(
        ValueError,
        match="missing required sections",
    ):
        validate_serialized_report(report)


def test_validation_rejects_invalid_dataset_shape():
    report = build_valid_report()

    report["dataset"]["shape"] = {
        "rows": 100,
    }

    with pytest.raises(
        ValueError,
        match="dataset shape must contain",
    ):
        validate_serialized_report(report)


def test_validation_rejects_invalid_task_type():
    report = build_valid_report()

    report["modeling"]["task_type"] = "clustering"

    with pytest.raises(
        ValueError,
        match="task_type must be",
    ):
        validate_serialized_report(report)


def test_validation_rejects_invalid_leaderboard():
    report = build_valid_report()

    report["modeling"]["leaderboard"] = {}

    with pytest.raises(
        TypeError,
        match="leaderboard must be a list",
    ):
        validate_serialized_report(report)


def test_validation_rejects_invalid_insights():
    report = build_valid_report()

    report["explainability"]["insights"] = [
        "valid",
        123,
    ]

    with pytest.raises(
        TypeError,
        match="must contain only strings",
    ):
        validate_serialized_report(report)


def test_validation_rejects_invalid_feature_importance():
    report = build_valid_report()

    report["explainability"][
        "feature_importance"
    ] = [
        {
            "feature": "age",
        }
    ]

    with pytest.raises(
        ValueError,
        match="must contain 'feature' and 'importance'",
    ):
        validate_serialized_report(report)


def test_validation_rejects_available_artifact_without_path():
    report = build_valid_report()

    report["artifact"]["available"] = True

    with pytest.raises(
        ValueError,
        match="path is required",
    ):
        validate_serialized_report(report)