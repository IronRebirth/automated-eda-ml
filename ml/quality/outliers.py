import pandas as pd


def analyze_outliers(df: pd.DataFrame) -> dict:
    """Detect outliers in numerical columns using the IQR method."""

    results = {}

    numerical_columns = df.select_dtypes(include=["number"]).columns

    for column in numerical_columns:
        series = df[column].dropna()

        if series.empty:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)

        outlier_mask = (series < lower_bound) | (series > upper_bound)
        outlier_count = int(outlier_mask.sum())

        results[column] = {
            "outlier_count": outlier_count,
            "outlier_percentage": round(
                (outlier_count / len(series)) * 100,
                2,
            ),
            "lower_bound": round(float(lower_bound), 4),
            "upper_bound": round(float(upper_bound), 4),
        }

    return results