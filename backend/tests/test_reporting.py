import pandas as pd
import pytest

from ml.reporting import build_analysis_report


def build_pipeline_result() -> dict:
    """Build a representative pipeline result."""

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

    return {
        "task_type": "classification",
        "target_column": "churn",
        "evaluation": {
            "random_forest": {
                "accuracy": 0.90,
                "precision": 0.89,
                "recall": 0.90,
                "f1": 0.89,
            }
        },
        "cross_validation": {
            "random_forest": {
                "accuracy": 0.88,
                "f1": 0.87,
            }
        },
        "leaderboard": [
            {
                "rank": 1,
                "model": "random_forest",
                "score": 0.89,
            }
        ],
        "optimization": {
            "random_forest": {
                "best_score": 0.91,
                "n_trials": 20,
            }
        },
        "optimized_evaluation": {
            "random_forest": {
                "accuracy": 0.92,
                "f1": 0.91,
            }
        },
        "best_model": {
            "model_name": "random_forest",
            "metric": "f1",
            "score": 0.91,
        },
        "explainability": {
            "metadata": {
                "method": "SHAP",
                "model": "random_forest",
                "task_type": "classification",
                "feature_count": 2,
                "top_features_count": 2,
                "preprocessing_applied": True,
            },
            "summary": {
                "top_n": 2,
                "features": [
                    {
                        "rank": 1,
                        "feature": "age",
                        "importance": 0.8,
                        "relative_importance": 1.0,
                        "impact": "high",
                    }
                ],
            },
            "insights": [
                "'age' is the most influential feature."
            ],
            "feature_importance": feature_importance,
        },
        "artifact_path": "artifacts/model.joblib",
    }


def test_build_analysis_report():
    pipeline_result = build_pipeline_result()

    report = build_analysis_report(
        pipeline_result,
        dataset_shape=(100, 5),
        data_quality={
            "missing_values": {},
        },
        eda={
            "numerical": {},
        },
    )

    assert "dataset" in report
    assert "data_quality" in report
    assert "eda" in report
    assert "modeling" in report
    assert "explainability" in report
    assert "artifact" in report


def test_report_contains_dataset_information():
    pipeline_result = build_pipeline_result()

    report = build_analysis_report(
        pipeline_result,
        dataset_shape=(100, 5),
    )

    assert report["dataset"]["shape"] == {
        "rows": 100,
        "columns": 5,
    }

    assert (
        report["dataset"]["target_column"]
        == "churn"
    )


def test_report_contains_modeling_information():
    pipeline_result = build_pipeline_result()

    report = build_analysis_report(
        pipeline_result
    )

    modeling = report["modeling"]

    assert modeling["task_type"] == "classification"
    assert "evaluation" in modeling
    assert "cross_validation" in modeling
    assert "leaderboard" in modeling
    assert "optimization" in modeling
    assert "optimized_evaluation" in modeling
    assert "best_model" in modeling


def test_report_contains_explainability_information():
    pipeline_result = build_pipeline_result()

    report = build_analysis_report(
        pipeline_result
    )

    explainability = report[
        "explainability"
    ]

    assert explainability["metadata"][
        "method"
    ] == "SHAP"

    assert explainability["summary"][
        "top_n"
    ] == 2

    assert len(
        explainability["insights"]
    ) > 0

    assert isinstance(
        explainability["feature_importance"],
        pd.DataFrame,
    )


def test_report_contains_artifact_information():
    pipeline_result = build_pipeline_result()

    report = build_analysis_report(
        pipeline_result
    )

    artifact = report["artifact"]

    assert artifact["path"] == (
        "artifacts/model.joblib"
    )

    assert artifact["available"] is True


def test_report_handles_missing_optional_sections():
    pipeline_result = build_pipeline_result()

    report = build_analysis_report(
        pipeline_result
    )

    assert report["dataset"]["shape"] is None
    assert report["data_quality"] is None
    assert report["eda"] is None


def test_report_rejects_invalid_pipeline_result():
    with pytest.raises(
        TypeError,
        match="pipeline_result must be a dictionary",
    ):
        build_analysis_report([])


def test_report_rejects_missing_pipeline_keys():
    with pytest.raises(
        ValueError,
        match="missing required keys",
    ):
        build_analysis_report(
            {
                "task_type": "classification",
            }
        )