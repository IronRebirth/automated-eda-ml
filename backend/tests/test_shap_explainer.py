import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from ml.explainability import (
    explain_model,
    explain_preprocessed_model,
)
from ml.pipeline.preprocessing import (
    build_preprocessing_pipeline,
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
            "city": [
                "Dhaka",
                "Dhaka",
                "Sylhet",
                "Dhaka",
                "Chittagong",
                "Dhaka",
                "Sylhet",
                "Dhaka",
                "Chittagong",
                "Dhaka",
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


def test_explain_model_returns_feature_importance():
    dataframe = create_dataset()

    target = create_target()

    model = RandomForestClassifier(
        n_estimators=20,
        random_state=42,
    )

    model.fit(
        pd.get_dummies(dataframe),
        target,
    )

    encoded_data = pd.get_dummies(
        dataframe
    )

    result = explain_model(
        model,
        encoded_data,
    )

    assert "feature_importance" in result
    assert "shap_values" in result

    feature_importance = result[
        "feature_importance"
    ]

    assert list(
        feature_importance.columns
    ) == [
        "feature",
        "importance",
    ]

    assert len(feature_importance) == len(
        encoded_data.columns
    )

    assert all(
        feature_importance["importance"] >= 0
    )


def test_explain_preprocessed_model_handles_categorical_features():
    dataframe = create_dataset()

    target = create_target()

    preprocessing_pipeline = (
        build_preprocessing_pipeline(
            dataframe
        )
    )

    transformed_data = (
        preprocessing_pipeline.fit_transform(
            dataframe
        )
    )

    model = RandomForestClassifier(
        n_estimators=20,
        random_state=42,
    )

    model.fit(
        transformed_data,
        target,
    )

    result = explain_preprocessed_model(
        model,
        preprocessing_pipeline,
        dataframe,
    )

    assert "feature_importance" in result
    assert "shap_values" in result
    assert "transformed_data" in result
    assert "feature_names" in result

    assert len(
        result["feature_names"]
    ) == transformed_data.shape[1]

    assert result[
        "transformed_data"
    ].shape == transformed_data.shape

    assert len(
        result["feature_importance"]
    ) == transformed_data.shape[1]


def test_explain_model_rejects_non_dataframe():
    model = RandomForestClassifier(
        n_estimators=10,
        random_state=42,
    )

    invalid_data = [
        [1, 2],
        [3, 4],
    ]

    try:
        explain_model(
            model,
            invalid_data,
        )
    except TypeError as error:
        assert str(error) == (
            "X must be a pandas DataFrame."
        )
    else:
        raise AssertionError(
            "Expected TypeError."
        )


def test_explain_model_rejects_empty_dataframe():
    dataframe = pd.DataFrame(
        columns=[
            "age",
            "salary",
        ]
    )

    model = RandomForestClassifier(
        n_estimators=10,
        random_state=42,
    )

    try:
        explain_model(
            model,
            dataframe,
        )
    except ValueError as error:
        assert str(error) == (
            "X must not be empty."
        )
    else:
        raise AssertionError(
            "Expected ValueError."
        )


def test_explain_preprocessed_model_rejects_non_dataframe():
    model = RandomForestClassifier(
        n_estimators=10,
        random_state=42,
    )

    preprocessing_pipeline = (
        build_preprocessing_pipeline(
            pd.DataFrame(
                {
                    "age": [20, 30],
                }
            )
        )
    )

    try:
        explain_preprocessed_model(
            model,
            preprocessing_pipeline,
            [[20]],
        )
    except TypeError as error:
        assert str(error) == (
            "X must be a pandas DataFrame."
        )
    else:
        raise AssertionError(
            "Expected TypeError."
        )