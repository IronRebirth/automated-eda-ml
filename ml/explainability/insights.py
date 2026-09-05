import pandas as pd


def generate_explainability_insights(
    feature_importance: pd.DataFrame,
    top_n: int = 5,
) -> list[str]:
    """Generate human-readable insights from SHAP feature importance."""

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
        .reset_index(drop=True)
    )

    insights = []

    top_feature = data.iloc[0]

    insights.append(
        f"'{top_feature['feature']}' is the most "
        "influential feature in the model."
    )

    if len(data) >= 2:
        second_feature = data.iloc[1]

        if second_feature["importance"] > 0:
            ratio = (
                top_feature["importance"]
                / second_feature["importance"]
            )

            if ratio >= 2:
                insights.append(
                    f"'{top_feature['feature']}' has more "
                    "than twice the SHAP importance of "
                    f"'{second_feature['feature']}'."
                )

    total_importance = data["importance"].sum()

    if total_importance > 0:
        top_feature_share = (
            top_feature["importance"]
            / total_importance
        )

        if top_feature_share >= 0.5:
            insights.append(
                f"'{top_feature['feature']}' accounts for "
                "more than half of the total importance "
                "among the top features analyzed."
            )

    if len(data) >= 3:
        lowest_feature = data.iloc[-1]

        insights.append(
            f"Among the top features analyzed, "
            f"'{lowest_feature['feature']}' has the "
            "lowest SHAP importance."
        )

    return insights