import pandas as pd


def validate_explainability_output(
    explainability: dict,
) -> None:
    """Validate the structure and consistency of explainability output."""

    if not isinstance(explainability, dict):
        raise TypeError(
            "explainability must be a dictionary."
        )

    required_keys = {
        "feature_importance",
        "shap_values",
        "transformed_data",
        "feature_names",
        "summary",
        "insights",
    }

    missing_keys = required_keys - explainability.keys()

    if missing_keys:
        raise ValueError(
            "Explainability output is missing required "
            f"keys: {sorted(missing_keys)}"
        )

    feature_importance = explainability[
        "feature_importance"
    ]

    if not isinstance(
        feature_importance,
        pd.DataFrame,
    ):
        raise TypeError(
            "feature_importance must be a pandas DataFrame."
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

    if feature_importance.empty:
        raise ValueError(
            "feature_importance must not be empty."
        )

    if feature_importance["feature"].isna().any():
        raise ValueError(
            "feature_importance contains missing feature names."
        )

    if not pd.api.types.is_numeric_dtype(
        feature_importance["importance"]
    ):
        raise TypeError(
            "feature_importance importance values must be numeric."
        )

    if feature_importance["importance"].isna().any():
        raise ValueError(
            "feature_importance contains missing importance values."
        )

    if (
        feature_importance["importance"] < 0
    ).any():
        raise ValueError(
            "feature_importance importance values "
            "must be non-negative."
        )

    feature_names = explainability[
        "feature_names"
    ]

    if not isinstance(feature_names, list):
        raise TypeError(
            "feature_names must be a list."
        )

    if not feature_names:
        raise ValueError(
            "feature_names must not be empty."
        )

    transformed_data = explainability[
        "transformed_data"
    ]

    if not isinstance(
        transformed_data,
        pd.DataFrame,
    ):
        raise TypeError(
            "transformed_data must be a pandas DataFrame."
        )

    if transformed_data.shape[1] != len(
        feature_names
    ):
        raise ValueError(
            "The number of transformed data columns "
            "must match the number of feature names."
        )

    if transformed_data.columns.tolist() != feature_names:
        raise ValueError(
            "transformed_data columns must match "
            "feature_names."
        )

    summary = explainability["summary"]

    if not isinstance(summary, dict):
        raise TypeError(
            "summary must be a dictionary."
        )

    if "top_n" not in summary:
        raise ValueError(
            "summary must contain 'top_n'."
        )

    if "features" not in summary:
        raise ValueError(
            "summary must contain 'features'."
        )

    summary_features = summary["features"]

    if not isinstance(summary_features, list):
        raise TypeError(
            "summary features must be a list."
        )

    if summary["top_n"] != len(summary_features):
        raise ValueError(
            "summary top_n must match the number "
            "of summary features."
        )

    for expected_rank, feature in enumerate(
        summary_features,
        start=1,
    ):
        if not isinstance(feature, dict):
            raise TypeError(
                "Each summary feature must be a dictionary."
            )

        required_feature_keys = {
            "rank",
            "feature",
            "importance",
            "relative_importance",
            "impact",
        }

        if not required_feature_keys.issubset(
            feature.keys()
        ):
            raise ValueError(
                "Each summary feature must contain "
                "rank, feature, importance, "
                "relative_importance, and impact."
            )

        if feature["rank"] != expected_rank:
            raise ValueError(
                "Summary feature ranks must start at 1 "
                "and increase sequentially."
            )

        if not isinstance(
            feature["importance"],
            (int, float),
        ):
            raise TypeError(
                "Summary importance values must be numeric."
            )

        if feature["importance"] < 0:
            raise ValueError(
                "Summary importance values must be non-negative."
            )

        if not isinstance(
            feature["relative_importance"],
            (int, float),
        ):
            raise TypeError(
                "Summary relative importance values "
                "must be numeric."
            )

        if not 0 <= feature[
            "relative_importance"
        ] <= 1:
            raise ValueError(
                "Summary relative importance values "
                "must be between 0 and 1."
            )

        if feature["impact"] not in {
            "high",
            "medium",
            "low",
        }:
            raise ValueError(
                "Summary impact must be 'high', "
                "'medium', or 'low'."
            )

    insights = explainability["insights"]

    if not isinstance(insights, list):
        raise TypeError(
            "insights must be a list."
        )

    if not insights:
        raise ValueError(
            "insights must not be empty."
        )

    if not all(
        isinstance(insight, str) and insight.strip()
        for insight in insights
    ):
        raise ValueError(
            "Every explainability insight must be "
            "a non-empty string."
        )