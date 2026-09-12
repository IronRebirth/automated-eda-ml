import pandas as pd


def detect_task_type(
    df: pd.DataFrame,
    target_column: str,
) -> str:
    """Automatically detect whether the target is for classification or regression."""

    if target_column not in df.columns:
        raise ValueError(
            f"Target column not found: {target_column}"
        )

    target = df[target_column].dropna()

    if target.empty:
        raise ValueError(
            "Target column contains no non-missing values."
        )

    if not pd.api.types.is_numeric_dtype(target):
        return "classification"

    if pd.api.types.is_float_dtype(target):
        return "regression"

    unique_count = target.nunique()

    if unique_count <= 20:
        return "classification"

    return "regression"