import pandas as pd


def generate_eda_insights(
    df: pd.DataFrame,
    numerical_analysis: dict,
    categorical_analysis: dict,
    correlation_analysis: dict,
    target_analysis: dict | None = None,
) -> list[str]:
    """Generate human-readable insights from EDA results."""

    insights = []

    # Numerical insights
    for column, stats in numerical_analysis.items():
        if stats["skewness"] > 1:
            insights.append(
                f"{column} is strongly right-skewed "
                f"(skewness: {stats['skewness']})."
            )
        elif stats["skewness"] < -1:
            insights.append(
                f"{column} is strongly left-skewed "
                f"(skewness: {stats['skewness']})."
            )

    # Categorical insights
    for column, stats in categorical_analysis.items():
        if stats["most_frequent_percentage"] >= 80:
            insights.append(
                f"{column} is highly dominated by "
                f"'{stats['most_frequent']}' "
                f"({stats['most_frequent_percentage']}% of values)."
            )

    # Correlation insights
    for correlation in correlation_analysis["strong_correlations"]:
        feature_a = correlation["feature_a"]
        feature_b = correlation["feature_b"]
        value = correlation["correlation"]

        relationship = (
            "positive"
            if value > 0
            else "negative"
        )

        insights.append(
            f"{feature_a} and {feature_b} have a strong "
            f"{relationship} correlation ({value})."
        )

    # Target insights
    if target_analysis is not None:
        missing_count = target_analysis["missing_count"]

        if missing_count > 0:
            insights.append(
                f"The target column "
                f"'{target_analysis['column']}' contains "
                f"{missing_count} missing value(s)."
            )

        if target_analysis["task_type"] == "classification":
            distribution = target_analysis["distribution"]

            if distribution:
                total = sum(distribution.values())
                majority_count = max(distribution.values())
                majority_percentage = (
                    majority_count / total
                ) * 100

                if majority_percentage >= 80:
                    insights.append(
                        f"The target '{target_analysis['column']}' "
                        f"is highly imbalanced, with the majority "
                        f"class representing "
                        f"{majority_percentage:.2f}% of observations."
                    )

    return insights