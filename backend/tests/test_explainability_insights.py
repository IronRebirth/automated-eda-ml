import pandas as pd
import pytest

from ml.explainability import (
    generate_explainability_insights,
)


def create_feature_importance():
    return pd.DataFrame(
        {
            "feature": [
                "contract_type",
                "tenure",
                "monthly_charges",
                "age",
            ],
            "importance": [
                0.80,
                0.30,
                0.15,
                0.05,
            ],
        }
    )


def test_generate_explainability_insights():
    insights = generate_explainability_insights(
        create_feature_importance()
    )

    assert len(insights) == 4

    assert insights[0] == (
        "'contract_type' is the most "
        "influential feature in the model."
    )


def test_insights_detect_dominant_feature():
    insights = generate_explainability_insights(
        create_feature_importance()
    )

    assert (
        "'contract_type' has more than twice the "
        "SHAP importance of 'tenure'."
        in insights
    )


def test_insights_detect_top_feature_share():
    feature_importance = pd.DataFrame(
        {
            "feature": [
                "feature_a",
                "feature_b",
            ],
            "importance": [
                0.80,
                0.20,
            ],
        }
    )

    insights = generate_explainability_insights(
        feature_importance
    )

    assert (
        "'feature_a' accounts for more than half "
        "of the total importance among the top "
        "features analyzed."
        in insights
    )


def test_insights_identify_lowest_top_feature():
    insights = generate_explainability_insights(
        create_feature_importance()
    )

    assert (
        "Among the top features analyzed, "
        "'age' has the lowest SHAP importance."
        in insights
    )


def test_insights_respects_top_n():
    insights = generate_explainability_insights(
        create_feature_importance(),
        top_n=2,
    )

    assert len(insights) == 3

    assert all(
        "monthly_charges" not in insight
        for insight in insights
    )


def test_insights_reject_invalid_data():
    with pytest.raises(
        ValueError,
        match=(
            "feature_importance must contain "
            "'feature' and 'importance' columns."
        ),
    ):
        generate_explainability_insights(
            pd.DataFrame(
                {
                    "feature": ["age"],
                }
            )
        )


def test_insights_reject_invalid_top_n():
    with pytest.raises(
        ValueError,
        match="top_n must be at least 1.",
    ):
        generate_explainability_insights(
            create_feature_importance(),
            top_n=0,
        )