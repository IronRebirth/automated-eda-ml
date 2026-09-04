import pandas as pd
import plotly.express as px


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
            labels={column: column.replace("_", " ").title()},
        )

        figures[column] = figure

    return figures


def create_categorical_bar_charts(df: pd.DataFrame) -> dict:
    """Create bar charts for categorical columns."""

    categorical_columns = df.select_dtypes(
        include=["object", "string", "category", "bool"]
    ).columns

    figures = {}

    for column in categorical_columns:
        value_counts = (
            df[column]
            .dropna()
            .value_counts()
            .reset_index()
        )

        value_counts.columns = ["category", "count"]

        figure = px.bar(
            value_counts,
            x="category",
            y="count",
            title=f"Distribution of {column}",
            labels={
                "category": column.replace("_", " ").title(),
                "count": "Count",
            },
        )

        figures[column] = figure

    return figures