import pandas as pd


def analyze_duplicates(df: pd.DataFrame) -> dict:
    """Analyze duplicate rows in a dataset."""

    duplicate_count = int(df.duplicated().sum())
    total_rows = len(df)

    percentage = (
        (duplicate_count / total_rows) * 100
        if total_rows > 0
        else 0.0
    )

    return {
        "count": duplicate_count,
        "percentage": round(percentage, 2),
    }