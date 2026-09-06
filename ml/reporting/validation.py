from typing import Any


def validate_serialized_report(
    report: dict[str, Any],
) -> None:
    """Validate the structure of a JSON-safe analysis report."""

    if not isinstance(report, dict):
        raise TypeError(
            "report must be a dictionary."
        )

    required_sections = {
        "dataset",
        "data_quality",
        "eda",
        "modeling",
        "explainability",
        "artifact",
    }

    missing_sections = required_sections.difference(
        report.keys()
    )

    if missing_sections:
        raise ValueError(
            "Report is missing required sections: "
            f"{sorted(missing_sections)}"
        )

    _validate_dataset(report["dataset"])
    _validate_modeling(report["modeling"])
    _validate_explainability(
        report["explainability"]
    )
    _validate_artifact(report["artifact"])


def _validate_dataset(
    dataset: Any,
) -> None:
    """Validate the dataset section."""

    if not isinstance(dataset, dict):
        raise TypeError(
            "dataset must be a dictionary."
        )

    required_keys = {
        "shape",
        "target_column",
    }

    if not required_keys.issubset(
        dataset.keys()
    ):
        raise ValueError(
            "dataset is missing required fields."
        )

    shape = dataset["shape"]

    if shape is not None:
        if not isinstance(shape, dict):
            raise TypeError(
                "dataset shape must be a dictionary."
            )

        if set(shape.keys()) != {
            "rows",
            "columns",
        }:
            raise ValueError(
                "dataset shape must contain "
                "'rows' and 'columns'."
            )

        if not isinstance(
            shape["rows"],
            int,
        ):
            raise TypeError(
                "dataset rows must be an integer."
            )

        if not isinstance(
            shape["columns"],
            int,
        ):
            raise TypeError(
                "dataset columns must be an integer."
            )

        if shape["rows"] < 1:
            raise ValueError(
                "dataset rows must be at least 1."
            )

        if shape["columns"] < 1:
            raise ValueError(
                "dataset columns must be at least 1."
            )

    target_column = dataset[
        "target_column"
    ]

    if not isinstance(
        target_column,
        str,
    ):
        raise TypeError(
            "target_column must be a string."
        )

    if not target_column.strip():
        raise ValueError(
            "target_column must not be empty."
        )


def _validate_modeling(
    modeling: Any,
) -> None:
    """Validate the modeling section."""

    if not isinstance(
        modeling,
        dict,
    ):
        raise TypeError(
            "modeling must be a dictionary."
        )

    required_keys = {
        "task_type",
        "evaluation",
        "cross_validation",
        "leaderboard",
        "optimization",
        "optimized_evaluation",
        "best_model",
    }

    if not required_keys.issubset(
        modeling.keys()
    ):
        raise ValueError(
            "modeling is missing required fields."
        )

    if modeling["task_type"] not in {
        "classification",
        "regression",
    }:
        raise ValueError(
            "modeling task_type must be "
            "'classification' or 'regression'."
        )

    if not isinstance(
        modeling["evaluation"],
        dict,
    ):
        raise TypeError(
            "evaluation must be a dictionary."
        )

    if not isinstance(
        modeling["cross_validation"],
        dict,
    ):
        raise TypeError(
            "cross_validation must be a dictionary."
        )

    if not isinstance(
        modeling["leaderboard"],
        list,
    ):
        raise TypeError(
            "leaderboard must be a list."
        )

    if not isinstance(
        modeling["optimization"],
        dict,
    ):
        raise TypeError(
            "optimization must be a dictionary."
        )

    if not isinstance(
        modeling["optimized_evaluation"],
        dict,
    ):
        raise TypeError(
            "optimized_evaluation must be a dictionary."
        )

    if not isinstance(
        modeling["best_model"],
        dict,
    ):
        raise TypeError(
            "best_model must be a dictionary."
        )


def _validate_explainability(
    explainability: Any,
) -> None:
    """Validate the serialized explainability section."""

    if not isinstance(
        explainability,
        dict,
    ):
        raise TypeError(
            "explainability must be a dictionary."
        )

    required_keys = {
        "metadata",
        "summary",
        "insights",
        "feature_importance",
    }

    if not required_keys.issubset(
        explainability.keys()
    ):
        raise ValueError(
            "explainability is missing required fields."
        )

    metadata = explainability[
        "metadata"
    ]

    if (
        metadata is not None
        and not isinstance(metadata, dict)
    ):
        raise TypeError(
            "explainability metadata "
            "must be a dictionary."
        )

    summary = explainability[
        "summary"
    ]

    if (
        summary is not None
        and not isinstance(summary, dict)
    ):
        raise TypeError(
            "explainability summary "
            "must be a dictionary."
        )

    insights = explainability[
        "insights"
    ]

    if insights is not None:
        if not isinstance(
            insights,
            list,
        ):
            raise TypeError(
                "explainability insights "
                "must be a list."
            )

        if not all(
            isinstance(
                insight,
                str,
            )
            for insight in insights
        ):
            raise TypeError(
                "explainability insights "
                "must contain only strings."
            )

    feature_importance = (
        explainability[
            "feature_importance"
        ]
    )

    if feature_importance is not None:
        if not isinstance(
            feature_importance,
            list,
        ):
            raise TypeError(
                "feature_importance must "
                "be a list."
            )

        for item in feature_importance:
            if not isinstance(
                item,
                dict,
            ):
                raise TypeError(
                    "feature_importance items "
                    "must be dictionaries."
                )

            if not {
                "feature",
                "importance",
            }.issubset(item.keys()):
                raise ValueError(
                    "feature_importance items "
                    "must contain 'feature' "
                    "and 'importance'."
                )


def _validate_artifact(
    artifact: Any,
) -> None:
    """Validate the artifact section."""

    if not isinstance(
        artifact,
        dict,
    ):
        raise TypeError(
            "artifact must be a dictionary."
        )

    required_keys = {
        "path",
        "available",
    }

    if not required_keys.issubset(
        artifact.keys()
    ):
        raise ValueError(
            "artifact is missing required fields."
        )

    path = artifact["path"]

    if path is not None and not isinstance(
        path,
        str,
    ):
        raise TypeError(
            "artifact path must be a string or None."
        )

    if not isinstance(
        artifact["available"],
        bool,
    ):
        raise TypeError(
            "artifact available must be a boolean."
        )

    if artifact["available"] and path is None:
        raise ValueError(
            "artifact path is required when "
            "artifact is available."
        )