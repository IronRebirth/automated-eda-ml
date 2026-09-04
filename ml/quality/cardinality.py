import pandas as pd


def analyze_cardinality(df: pd.DataFrame) -> dict:
    """Analyze the number of unique values in each column."""

    results = {}

    for column in df.columns:
        unique_count = int(df[column].nunique(dropna=True))
        total_rows = len(df)

        unique_percentage = (
            (unique_count / total_rows) * 100
            if total_rows > 0
            else 0.0
        )

        results[column] = {
            "unique_count": unique_count,
            "unique_percentage": round(unique_percentage, 2),
        }

    return results