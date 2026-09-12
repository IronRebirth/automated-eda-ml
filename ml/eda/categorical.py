import pandas as pd


def analyze_categorical_columns(df: pd.DataFrame) -> dict:
    """Generate descriptive statistics for categorical columns."""

    categorical_columns = df.select_dtypes(
        include=["object", "string", "category", "bool"]
    ).columns

    results = {}

    for column in categorical_columns:
        series = df[column].dropna()

        if series.empty:
            continue

        value_counts = series.value_counts()

        most_frequent = value_counts.index[0]
        most_frequent_count = int(value_counts.iloc[0])

        results[column] = {
            "count": int(series.count()),
            "unique": int(series.nunique()),
            "most_frequent": str(most_frequent),
            "most_frequent_count": most_frequent_count,
            "most_frequent_percentage": round(
                (most_frequent_count / len(series)) * 100,
                2,
            ),
            "value_counts": {
                str(category): int(count)
                for category, count in value_counts.items()
            },
        }

    return results