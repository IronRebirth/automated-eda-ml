import pandas as pd


class DatasetProfiler:
    """Generate basic metadata and statistics for a dataset."""

    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe

    def profile(self) -> dict:
        df = self.df

        numerical_columns = df.select_dtypes(include=["number"]).columns.tolist()

        categorical_columns = df.select_dtypes(
            include=["object", "string", "category", "bool"]
        ).columns.tolist()

        datetime_columns = df.select_dtypes(include=["datetime"]).columns.tolist()

        constant_columns = [
            column for column in df.columns if df[column].nunique(dropna=False) <= 1
        ]

        return {
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": df.columns.tolist(),
            "numerical_columns": numerical_columns,
            "categorical_columns": categorical_columns,
            "datetime_columns": datetime_columns,
            "missing_values": int(df.isna().sum().sum()),
            "missing_percentage": round(
                df.isna().mean().mean() * 100,
                2,
            ),
            "duplicate_rows": int(df.duplicated().sum()),
            "constant_columns": constant_columns,
            "memory_usage_mb": round(
                df.memory_usage(deep=True).sum() / 1024**2,
                2,
            ),
        }
