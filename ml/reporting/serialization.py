from typing import Any

import pandas as pd


def serialize_analysis_report(
    report: dict[str, Any],
) -> dict[str, Any]:
    """Convert an analysis report into JSON-serializable data."""

    if not isinstance(report, dict):
        raise TypeError(
            "report must be a dictionary."
        )

    serialized = _serialize_value(report)

    if not isinstance(serialized, dict):
        raise TypeError(
            "Serialized report must be a dictionary."
        )

    return serialized


def _serialize_value(value: Any) -> Any:
    """Recursively convert supported values into JSON-safe objects."""

    if isinstance(value, pd.DataFrame):
        return value.to_dict(
            orient="records"
        )

    if isinstance(value, pd.Series):
        return value.tolist()

    if isinstance(value, dict):
        return {
            str(key): _serialize_value(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            _serialize_value(item)
            for item in value
        ]

    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass

    return value