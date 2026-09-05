import pandas as pd
import pytest

from ml.explainability import (
    build_explainability_summary,
)


def create_feature_importance():
    return pd.DataFrame(
        {
            "feature": [
                "contract_month_to_month",
                "tenure",
                "monthly_charges",
                "age",
            ],
            "importance": [
                0.80,
                0.50,
                0.25,
                0.10,
            ],
        }
    )


def test_build_explainability_summary():
    result = build_explainability_summary(
        create_feature_importance()
    )

    assert result["top_n"] == 4
    assert len(result["features"]) == 4

    first_feature = result["features"][0]

    assert first_feature["rank"] == 1
    assert first_feature["feature"] == (
        "contract_month_to_month"
    )
    assert first_feature["importance"] == 0.80
    assert first_feature["relative_importance"] == 1.0
    assert first_feature["impact"] == "high"


def test_summary_respects_top_n():
    result = build_explainability_summary(
        create_feature_importance(),
        top_n=2,
    )

    assert result["top_n"] == 2
    assert len(result["features"]) == 2
    assert result["features"][0]["rank"] == 1
    assert result["features"][1]["rank"] == 2


def test_summary_classifies_impact_levels():
    result = build_explainability_summary(
        create_feature_importance()
    )

    impacts = [
        feature["impact"]
        for feature in result["features"]
    ]

    assert impacts == [
        "high",
        "medium",
        "low",
        "low",
    ]


def test_summary_handles_zero_importance():
    feature_importance = pd.DataFrame(
        {
            "feature": [
                "feature_a",
                "feature_b",
            ],
            "importance": [
                0.0,
                0.0,
            ],
        }
    )

    result = build_explainability_summary(
        feature_importance
    )

    assert result["features"][0][
        "relative_importance"
    ] == 0.0

    assert result["features"][0][
        "impact"
    ] == "low"


def test_summary_rejects_invalid_data():
    with pytest.raises(
        ValueError,
        match=(
            "feature_importance must contain "
            "'feature' and 'importance' columns."
        ),
    ):
        build_explainability_summary(
            pd.DataFrame(
                {
                    "feature": ["age"],
                }
            )
        )


def test_summary_rejects_invalid_top_n():
    with pytest.raises(
        ValueError,
        match="top_n must be at least 1.",
    ):
        build_explainability_summary(
            create_feature_importance(),
            top_n=0,
        )