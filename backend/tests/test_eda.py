import pandas as pd

from ml.eda import EDAAnalyzer


def test_eda_analyzer_returns_numerical_analysis():
    dataframe = pd.DataFrame(
        {
            "age": [20, 30, 40],
            "city": ["Dhaka", "Dhaka", "Sylhet"],
        }
    )

    analyzer = EDAAnalyzer(dataframe)

    result = analyzer.analyze()

    assert "age" in result["numerical"]
    assert "city" not in result["numerical"]


def test_eda_analyzer_returns_categorical_analysis():
    dataframe = pd.DataFrame(
        {
            "age": [20, 30, 40],
            "city": ["Dhaka", "Dhaka", "Sylhet"],
        }
    )

    analyzer = EDAAnalyzer(dataframe)

    result = analyzer.analyze()

    assert "city" in result["categorical"]
    assert "age" not in result["categorical"]


def test_eda_analyzer_returns_target_analysis():
    dataframe = pd.DataFrame(
        {
            "age": [20, 30, 40],
            "churn": [0, 1, 0],
        }
    )

    analyzer = EDAAnalyzer(
        dataframe,
        target_column="churn",
    )

    result = analyzer.analyze()

    assert result["target"]["column"] == "churn"
    assert result["target"]["task_type"] == "classification"


def test_eda_analyzer_returns_correlations():
    dataframe = pd.DataFrame(
        {
            "age": [20, 30, 40],
            "salary": [20000, 30000, 40000],
        }
    )

    analyzer = EDAAnalyzer(dataframe)

    result = analyzer.analyze()

    assert "matrix" in result["correlations"]
    assert "age" in result["correlations"]["matrix"]
    assert "salary" in result["correlations"]["matrix"]


def test_eda_analyzer_returns_visualizations():
    dataframe = pd.DataFrame(
        {
            "age": [20, 30, 40],
            "city": ["Dhaka", "Dhaka", "Sylhet"],
        }
    )

    analyzer = EDAAnalyzer(dataframe)

    result = analyzer.analyze()

    assert "age" in result["visualizations"]["numerical"]
    assert "city" in result["visualizations"]["categorical"]


def test_eda_analyzer_generates_insights():
    dataframe = pd.DataFrame(
        {
            "age": [20, 21, 22, 23, 24, 100],
            "city": [
                "Dhaka",
                "Dhaka",
                "Dhaka",
                "Dhaka",
                "Dhaka",
                "Sylhet",
            ],
        }
    )

    analyzer = EDAAnalyzer(dataframe)

    result = analyzer.analyze()

    assert isinstance(result["insights"], list)