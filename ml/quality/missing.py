import pandas as pd


def analyze_missing_values(df: pd.DataFrame) -> dict:
    """Analyze missing values in each column."""

    missing_counts = df.isna().sum()

    results = {}

    for column, count in missing_counts.items():
        if count > 0:
            percentage = (count / len(df)) * 100

            results[column] = {
                "count": int(count),
                "percentage": round(percentage, 2),
            }

    return results