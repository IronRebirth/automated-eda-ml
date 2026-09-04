import pandas as pd

from ml.quality import DataQualityAnalyzer


def test_quality_analyzer_detects_missing_values():
    dataframe = pd.DataFrame(
        {
            "age": [20, None, 40],
            "city": ["Dhaka", "Dhaka", "Chittagong"],
        }
    )

    analyzer = DataQualityAnalyzer(dataframe)

    result = analyzer.analyze()

    assert result["missing_values"]["age"]["count"] == 1


def test_quality_analyzer_detects_duplicates():
    dataframe = pd.DataFrame(
        {
            "age": [20, 30, 20],
            "city": ["Dhaka", "Chittagong", "Dhaka"],
        }
    )

    analyzer = DataQualityAnalyzer(dataframe)

    result = analyzer.analyze()

    assert result["duplicates"]["count"] == 1


def test_quality_analyzer_detects_cardinality():
    dataframe = pd.DataFrame(
        {
            "city": ["Dhaka", "Dhaka", "Chittagong"],
        }
    )

    analyzer = DataQualityAnalyzer(dataframe)

    result = analyzer.analyze()

    assert result["cardinality"]["city"]["unique_count"] == 2


def test_quality_analyzer_detects_outliers():
    dataframe = pd.DataFrame(
        {
            "age": [20, 21, 22, 23, 24, 100],
        }
    )

    analyzer = DataQualityAnalyzer(dataframe)

    result = analyzer.analyze()

    assert result["outliers"]["age"]["outlier_count"] == 1