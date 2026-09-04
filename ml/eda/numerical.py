import pandas as pd


def analyze_numerical_columns(df: pd.DataFrame) -> dict:
    """Generate descriptive statistics for numerical columns."""

    numerical_columns = df.select_dtypes(
        include=["number"]
    ).columns

    results = {}

    for column in numerical_columns:
        series = df[column].dropna()

        if series.empty:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)

        results[column] = {
            "count": int(series.count()),
            "unique": int(series.nunique()),
            "mean": round(float(series.mean()), 4),
            "median": round(float(series.median()), 4),
            "std": round(float(series.std()), 4),
            "min": round(float(series.min()), 4),
            "q1": round(float(q1), 4),
            "q3": round(float(q3), 4),
            "max": round(float(series.max()), 4),
            "iqr": round(float(q3 - q1), 4),
            "skewness": round(float(series.skew()), 4),
        }

    return results