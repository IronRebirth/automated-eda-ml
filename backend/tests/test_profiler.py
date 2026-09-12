import pandas as pd

from ml.profiling import DatasetProfiler


def test_profiler_counts_rows_and_columns():
    dataframe = pd.DataFrame(
        {
            "age": [20, 30, 40],
            "city": ["Dhaka", "Chittagong", "Dhaka"],
        }
    )

    profiler = DatasetProfiler(dataframe)

    result = profiler.profile()

    assert result["rows"] == 3
    assert result["columns"] == 2


def test_profiler_detects_missing_values():
    dataframe = pd.DataFrame(
        {
            "age": [20, None, 40],
        }
    )

    profiler = DatasetProfiler(dataframe)

    result = profiler.profile()

    assert result["missing_values"] == 1


def test_profiler_detects_constant_columns():
    dataframe = pd.DataFrame(
        {
            "age": [20, 30, 40],
            "country": ["BD", "BD", "BD"],
        }
    )

    profiler = DatasetProfiler(dataframe)

    result = profiler.profile()

    assert "country" in result["constant_columns"]
