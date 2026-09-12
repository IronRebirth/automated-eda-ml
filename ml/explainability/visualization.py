import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_feature_importance(
    feature_importance: pd.DataFrame,
    top_n: int = 20,
) -> go.Figure:
    """Create a horizontal bar chart of global SHAP feature importance."""

    if not isinstance(feature_importance, pd.DataFrame):
        raise TypeError(
            "feature_importance must be a pandas DataFrame."
        )

    required_columns = {
        "feature",
        "importance",
    }

    if not required_columns.issubset(
        feature_importance.columns
    ):
        raise ValueError(
            "feature_importance must contain "
            "'feature' and 'importance' columns."
        )

    if feature_importance.empty:
        raise ValueError(
            "feature_importance must not be empty."
        )

    if top_n < 1:
        raise ValueError(
            "top_n must be at least 1."
        )

    data = (
        feature_importance
        .head(top_n)
        .sort_values(
            "importance",
            ascending=True,
        )
    )

    figure = px.bar(
        data,
        x="importance",
        y="feature",
        orientation="h",
        title="Global SHAP Feature Importance",
        labels={
            "importance": "Mean |SHAP value|",
            "feature": "Feature",
        },
    )

    return figure


def plot_shap_summary(
    shap_values,
    feature_names: list[str],
    top_n: int = 20,
) -> go.Figure:
    """Create a SHAP summary-style scatter plot."""

    if not isinstance(feature_names, list):
        raise TypeError(
            "feature_names must be a list."
        )

    if not feature_names:
        raise ValueError(
            "feature_names must not be empty."
        )

    if top_n < 1:
        raise ValueError(
            "top_n must be at least 1."
        )

    values = shap_values.values

    if values.ndim == 3:
        values = values.mean(
            axis=2
        )

    if values.ndim != 2:
        raise ValueError(
            "shap_values must contain a 2-dimensional "
            "SHAP value array."
        )

    if values.shape[1] != len(feature_names):
        raise ValueError(
            "The number of feature names must match "
            "the number of SHAP features."
        )

    importance = (
        abs(values)
        .mean(axis=0)
    )

    ranked_indices = (
        pd.Series(importance)
        .sort_values(
            ascending=False
        )
        .head(top_n)
        .index
        .tolist()
    )

    rows = []

    for feature_index in ranked_indices:
        for sample_index in range(
            values.shape[0]
        ):
            rows.append(
                {
                    "feature": feature_names[
                        feature_index
                    ],
                    "shap_value": values[
                        sample_index,
                        feature_index,
                    ],
                }
            )

    data = pd.DataFrame(rows)

    figure = px.strip(
        data,
        x="shap_value",
        y="feature",
        orientation="h",
        title="SHAP Summary",
        labels={
            "shap_value": "SHAP value",
            "feature": "Feature",
        },
    )

    return figure