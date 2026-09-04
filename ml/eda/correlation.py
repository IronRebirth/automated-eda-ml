import pandas as pd


def analyze_correlations(df: pd.DataFrame) -> dict:
    """Analyze correlations between numerical columns."""

    numerical_columns = df.select_dtypes(
        include=["number"]
    ).columns

    if len(numerical_columns) < 2:
        return {
            "matrix": {},
            "strong_correlations": [],
        }

    correlation_matrix = df[numerical_columns].corr()

    strong_correlations = []

    for i, column_a in enumerate(numerical_columns):
        for column_b in numerical_columns[i + 1 :]:
            correlation = correlation_matrix.loc[column_a, column_b]

            if pd.isna(correlation):
                continue

            if abs(correlation) >= 0.7:
                strong_correlations.append(
                    {
                        "feature_a": column_a,
                        "feature_b": column_b,
                        "correlation": round(float(correlation), 4),
                    }
                )

    return {
        "matrix": {
            column: {
                other_column: round(
                    float(correlation_matrix.loc[column, other_column]),
                    4,
                )
                for other_column in numerical_columns
            }
            for column in numerical_columns
        },
        "strong_correlations": strong_correlations,
    }