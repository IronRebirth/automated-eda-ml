import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from ml.explainability import (
    explain_model,
    plot_feature_importance,
    plot_shap_summary,
)


def create_dataset():
    return pd.DataFrame(
        {
            "age": [
                20,
                22,
                24,
                26,
                28,
                30,
                32,
                34,
                36,
                38,
            ],
            "salary": [
                20000,
                22000,
                24000,
                26000,
                28000,
                30000,
                32000,
                34000,
                36000,
                38000,
            ],
            "city_Dhaka": [
                1,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
            ],
            "city_Sylhet": [
                0,
                0,
                1,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
            ],
            "city_Chittagong": [
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                1,
                0,
            ],
        }
    )


def create_target():
    return pd.Series(
        [
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            1,
        ]
    )


def test_plot_feature_importance_returns_figure():
    dataframe = create_dataset()

    target = create_target()

    model = RandomForestClassifier(
        n_estimators=20,
        random_state=42,
    )

    model.fit(
        dataframe,
        target,
    )

    result = explain_model(
        model,
        dataframe,
    )

    figure = plot_feature_importance(
        result["feature_importance"]
    )

    assert figure is not None
    assert len(figure.data) == 1
    assert figure.layout.title.text == (
        "Global SHAP Feature Importance"
    )


def test_plot_feature_importance_respects_top_n():
    feature_importance = pd.DataFrame(
        {
            "feature": [
                "feature_1",
                "feature_2",
                "feature_3",
            ],
            "importance": [
                0.8,
                0.5,
                0.2,
            ],
        }
    )

    figure = plot_feature_importance(
        feature_importance,
        top_n=2,
    )

    assert len(
        figure.data[0].y
    ) == 2


def test_plot_shap_summary_returns_figure():
    dataframe = create_dataset()

    target = create_target()

    model = RandomForestClassifier(
        n_estimators=20,
        random_state=42,
    )

    model.fit(
        dataframe,
        target,
    )

    result = explain_model(
        model,
        dataframe,
    )

    figure = plot_shap_summary(
        result["shap_values"],
        dataframe.columns.tolist(),
    )

    assert figure is not None
    assert len(figure.data) == 1
    assert figure.layout.title.text == (
        "SHAP Summary"
    )


def test_plot_feature_importance_rejects_invalid_data():
    try:
        plot_feature_importance(
            pd.DataFrame(
                {
                    "feature": ["age"],
                }
            )
        )
    except ValueError as error:
        assert str(error) == (
            "feature_importance must contain "
            "'feature' and 'importance' columns."
        )
    else:
        raise AssertionError(
            "Expected ValueError."
        )


def test_plot_shap_summary_rejects_feature_mismatch():
    dataframe = create_dataset()

    target = create_target()

    model = RandomForestClassifier(
        n_estimators=20,
        random_state=42,
    )

    model.fit(
        dataframe,
        target,
    )

    result = explain_model(
        model,
        dataframe,
    )

    try:
        plot_shap_summary(
            result["shap_values"],
            ["age"],
        )
    except ValueError as error:
        assert str(error) == (
            "The number of feature names must match "
            "the number of SHAP features."
        )
    else:
        raise AssertionError(
            "Expected ValueError."
        )