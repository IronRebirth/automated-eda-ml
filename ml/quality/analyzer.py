import pandas as pd

from ml.quality.cardinality import analyze_cardinality
from ml.quality.duplicates import analyze_duplicates
from ml.quality.missing import analyze_missing_values
from ml.quality.outliers import analyze_outliers


class DataQualityAnalyzer:
    """Run all data-quality analyzers on a dataset."""

    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe

    def analyze(self) -> dict:
        """Generate a complete data-quality report."""

        return {
            "missing_values": analyze_missing_values(self.df),
            "duplicates": analyze_duplicates(self.df),
            "cardinality": analyze_cardinality(self.df),
            "outliers": analyze_outliers(self.df),
        }