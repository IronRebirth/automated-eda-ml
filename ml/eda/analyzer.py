import pandas as pd

from ml.eda.categorical import analyze_categorical_columns
from ml.eda.correlation import analyze_correlations
from ml.eda.insights import generate_eda_insights
from ml.eda.numerical import analyze_numerical_columns
from ml.eda.target import analyze_target
from ml.eda.visualization import (
    create_categorical_bar_charts,
    create_numerical_histograms,
)


class EDAAnalyzer:
    """Run the complete exploratory data analysis pipeline."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        target_column: str | None = None,
    ):
        self.df = dataframe
        self.target_column = target_column

    def analyze(self) -> dict:
        """Generate a complete EDA report."""

        numerical_analysis = analyze_numerical_columns(self.df)
        categorical_analysis = analyze_categorical_columns(self.df)
        correlation_analysis = analyze_correlations(self.df)

        target_analysis = None

        if self.target_column is not None:
            target_analysis = analyze_target(
                self.df,
                self.target_column,
            )

        insights = generate_eda_insights(
            self.df,
            numerical_analysis,
            categorical_analysis,
            correlation_analysis,
            target_analysis,
        )

        numerical_figures = create_numerical_histograms(self.df)
        categorical_figures = create_categorical_bar_charts(self.df)

        return {
            "numerical": numerical_analysis,
            "categorical": categorical_analysis,
            "correlations": correlation_analysis,
            "target": target_analysis,
            "insights": insights,
            "visualizations": {
                "numerical": numerical_figures,
                "categorical": categorical_figures,
            },
        }