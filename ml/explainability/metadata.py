import pandas as pd


def build_explainability_metadata(
    feature_importance: pd.DataFrame,
    model_name: str,
    task_type: str,
    preprocessing_applied: bool = True,
    top_features_count: int | None = None,
) -> dict:
    """Build metadata describing an explainability result."""

    if not isinstance(
        feature_importance,
        pd.DataFrame,
    ):
        raise TypeError(
            "feature_importance must be a pandas DataFrame."
        )

    if feature_importance.empty:
        raise ValueError(
            "feature_importance must not be empty."
        )

    required_columns = {
        "feature",
        "importance",
    }

    if not required_columns.issubset(
        feature_importance.columns
    ):
        raise ValueError(
            "feature_importance must contain "
            "'feature' and 'importance' columns."
        )

    if not isinstance(model_name, str):
        raise TypeError(
            "model_name must be a string."
        )

    if not model_name.strip():
        raise ValueError(
            "model_name must not be empty."
        )

    if task_type not in {
        "classification",
        "regression",
    }:
        raise ValueError(
            f"Unsupported task type: {task_type}"
        )

    if not isinstance(
        preprocessing_applied,
        bool,
    ):
        raise TypeError(
            "preprocessing_applied must be a boolean."
        )

    if top_features_count is not None:
        if not isinstance(
            top_features_count,
            int,
        ):
            raise TypeError(
                "top_features_count must be an integer."
            )

        if top_features_count < 1:
            raise ValueError(
                "top_features_count must be at least 1."
            )

    feature_count = len(feature_importance)

    if top_features_count is None:
        top_features_count = feature_count

    return {
        "method": "SHAP",
        "model": model_name,
        "task_type": task_type,
        "feature_count": feature_count,
        "top_features_count": min(
            top_features_count,
            feature_count,
        ),
        "preprocessing_applied": (
            preprocessing_applied
        ),
    }