import pandas as pd
import pytest

from ml.reporting import serialize_analysis_report


def build_report() -> dict:
    """Build a report containing common non-JSON Python objects."""

    return {
        "dataset": {
            "shape": {
                "rows": 100,
                "columns": 4,
            },
        },
        "modeling": {
            "task_type": "classification",
            "score": 0.91,
            "metrics": {
                "accuracy": 0.90,
                "f1": 0.91,
            },
        },
        "explainability": {
            "feature_importance": pd.DataFrame(
                {
                    "feature": [
                        "age",
                        "income",
                    ],
                    "importance": [
                        0.8,
                        0.4,
                    ],
                }
            ),
            "values": pd.Series(
                [1, 2, 3]
            ),
        },
    }


def test_serialize_analysis_report():
    report = build_report()

    serialized = serialize_analysis_report(
        report
    )

    assert isinstance(
        serialized,
        dict,
    )

    assert serialized["dataset"]["shape"] == {
        "rows": 100,
        "columns": 4,
    }

    assert serialized["modeling"]["task_type"] == (
        "classification"
    )


def test_dataframe_is_serialized_to_records():
    report = build_report()

    serialized = serialize_analysis_report(
        report
    )

    feature_importance = serialized[
        "explainability"
    ]["feature_importance"]

    assert feature_importance == [
        {
            "feature": "age",
            "importance": 0.8,
        },
        {
            "feature": "income",
            "importance": 0.4,
        },
    ]


def test_series_is_serialized_to_list():
    report = build_report()

    serialized = serialize_analysis_report(
        report
    )

    assert serialized[
        "explainability"
    ]["values"] == [1, 2, 3]


def test_nested_values_are_serialized():
    report = {
        "items": [
            {
                "data": pd.DataFrame(
                    {
                        "feature": ["age"],
                        "importance": [0.8],
                    }
                )
            }
        ]
    }

    serialized = serialize_analysis_report(
        report
    )

    assert serialized["items"][0]["data"] == [
        {
            "feature": "age",
            "importance": 0.8,
        }
    ]


def test_tuple_is_serialized_to_list():
    report = {
        "values": (
            "age",
            "income",
        )
    }

    serialized = serialize_analysis_report(
        report
    )

    assert serialized["values"] == [
        "age",
        "income",
    ]


def test_numpy_scalars_are_serialized():
    import numpy as np

    report = {
        "float_value": np.float64(0.91),
        "int_value": np.int64(5),
    }

    serialized = serialize_analysis_report(
        report
    )

    assert serialized["float_value"] == 0.91
    assert serialized["int_value"] == 5


def test_invalid_report_is_rejected():
    with pytest.raises(
        TypeError,
        match="report must be a dictionary",
    ):
        serialize_analysis_report([])


def test_original_report_is_not_modified():
    report = build_report()

    original_dataframe = report[
        "explainability"
    ]["feature_importance"]

    serialize_analysis_report(report)

    assert isinstance(
        original_dataframe,
        pd.DataFrame,
    )