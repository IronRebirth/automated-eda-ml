from typing import Any


def build_analysis_report(
    pipeline_result: dict[str, Any],
    dataset_shape: tuple[int, int] | None = None,
    data_quality: dict[str, Any] | None = None,
    eda: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a unified, product-ready analysis report."""

    if not isinstance(pipeline_result, dict):
        raise TypeError(
            "pipeline_result must be a dictionary."
        )

    required_keys = {
        "task_type",
        "target_column",
        "evaluation",
        "cross_validation",
        "leaderboard",
        "optimization",
        "optimized_evaluation",
        "best_model",
        "explainability",
    }

    missing_keys = required_keys.difference(
        pipeline_result.keys()
    )

    if missing_keys:
        raise ValueError(
            "pipeline_result is missing required keys: "
            f"{sorted(missing_keys)}"
        )

    explainability = pipeline_result[
        "explainability"
    ]

    report = {
        "dataset": {
            "shape": (
                {
                    "rows": dataset_shape[0],
                    "columns": dataset_shape[1],
                }
                if dataset_shape is not None
                else None
            ),
            "target_column": pipeline_result[
                "target_column"
            ],
        },
        "data_quality": data_quality,
        "eda": eda,
        "modeling": {
            "task_type": pipeline_result[
                "task_type"
            ],
            "evaluation": pipeline_result[
                "evaluation"
            ],
            "cross_validation": pipeline_result[
                "cross_validation"
            ],
            "leaderboard": pipeline_result[
                "leaderboard"
            ],
            "optimization": pipeline_result[
                "optimization"
            ],
            "optimized_evaluation": pipeline_result[
                "optimized_evaluation"
            ],
            "best_model": pipeline_result[
                "best_model"
            ],
        },
        "explainability": {
            "metadata": explainability.get(
                "metadata"
            ),
            "summary": explainability.get(
                "summary"
            ),
            "insights": explainability.get(
                "insights"
            ),
            "feature_importance": explainability.get(
                "feature_importance"
            ),
        },
        "artifact": {
            "path": pipeline_result.get(
                "artifact_path"
            ),
            "available": (
                pipeline_result.get(
                    "artifact_path"
                )
                is not None
            ),
        },
    }

    return report