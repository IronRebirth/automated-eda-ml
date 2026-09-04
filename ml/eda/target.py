import pandas as pd


def analyze_target(df: pd.DataFrame, target_column: str) -> dict:
    """Analyze the target variable."""

    if target_column not in df.columns:
        raise ValueError(
            f"Target column not found: {target_column}"
        )

    series = df[target_column]

    missing_count = int(series.isna().sum())
    non_missing = series.dropna()

    if non_missing.empty:
        return {
            "column": target_column,
            "dtype": str(series.dtype),
            "task_type": "unknown",
            "missing_count": missing_count,
            "unique_count": 0,
            "distribution": {},
        }

    unique_count = int(non_missing.nunique())

    if pd.api.types.is_numeric_dtype(series):
        task_type = (
            "classification"
            if unique_count <= 20
            else "regression"
        )
    else:
        task_type = "classification"

    value_counts = non_missing.value_counts()

    distribution = {
        str(value): int(count)
        for value, count in value_counts.items()
    }

    return {
        "column": target_column,
        "dtype": str(series.dtype),
        "task_type": task_type,
        "missing_count": missing_count,
        "unique_count": unique_count,
        "distribution": distribution,
    }