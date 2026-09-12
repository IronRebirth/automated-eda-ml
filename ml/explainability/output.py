import pandas as pd


def build_explainability_summary(
    feature_importance: pd.DataFrame,
    top_n: int = 10,
) -> dict:
    """Build a structured, UI-ready explainability summary."""

    if not isinstance(feature_importance, pd.DataFrame):
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

    if top_n < 1:
        raise ValueError(
            "top_n must be at least 1."
        )

    data = (
        feature_importance
        .head(top_n)
        .copy()
        .reset_index(drop=True)
    )

    maximum_importance = data["importance"].max()

    features = []

    for rank, row in enumerate(
        data.itertuples(index=False),
        start=1,
    ):
        importance = float(
            row.importance
        )

        if maximum_importance == 0:
            relative_importance = 0.0
        else:
            relative_importance = (
                importance
                / maximum_importance
            )

        if relative_importance >= 0.66:
            impact = "high"
        elif relative_importance >= 0.33:
            impact = "medium"
        else:
            impact = "low"

        features.append(
            {
                "rank": rank,
                "feature": str(row.feature),
                "importance": importance,
                "relative_importance": (
                    relative_importance
                ),
                "impact": impact,
            }
        )

    return {
        "top_n": len(features),
        "features": features,
    }