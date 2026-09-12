import pandas as pd
import plotly.express as px

MAX_CATEGORIES = 15


def create_numerical_histograms(df: pd.DataFrame) -> dict:
    """Create histogram figures for numerical columns."""

    numerical_columns = df.select_dtypes(
        include=["number"]
    ).columns

    figures = {}

    for column in numerical_columns:
        figure = px.histogram(
            df,
            x=column,
            title=f"Distribution of {column}",
            labels={
                column: column.replace(
                    "_",
                    " ",
                ).title(),
            },
        )

        figures[column] = figure

    return figures


def _is_identifier_column(
    column: str,
) -> bool:
    """Return whether a column name strongly indicates an identifier."""

    normalized_name = column.lower().replace(
        "-",
        "_",
    )

    return (
        normalized_name == "id"
        or normalized_name.endswith("_id")
        or normalized_name.startswith("id_")
    )


def create_categorical_bar_charts(df: pd.DataFrame) -> dict:
    """Create bounded bar charts for useful categorical columns."""

    categorical_columns = df.select_dtypes(
        include=[
            "object",
            "string",
            "category",
            "bool",
        ]
    ).columns

    figures = {}

    for column in categorical_columns:
        series = df[column].dropna()

        if series.empty:
            continue

        if _is_identifier_column(column):
            continue

        unique_count = series.nunique()

        if unique_count > MAX_CATEGORIES:
            continue

        value_counts = (
            series
            .value_counts()
            .head(MAX_CATEGORIES)
            .reset_index()
        )

        value_counts.columns = [
            "category",
            "count",
        ]

        figure = px.bar(
            value_counts,
            x="category",
            y="count",
            title=f"Distribution of {column}",
            labels={
                "category": column.replace(
                    "_",
                    " ",
                ).title(),
                "count": "Count",
            },
        )

        figures[column] = figure

    return figures