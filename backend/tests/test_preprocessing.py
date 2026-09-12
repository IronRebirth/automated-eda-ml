import pandas as pd

from ml.pipeline import build_preprocessing_pipeline


def test_preprocessing_handles_numerical_and_categorical_columns():
    dataframe = pd.DataFrame(
        {
            "age": [20, 30, None, 40],
            "salary": [20000, 30000, 40000, None],
            "city": ["Dhaka", "Sylhet", None, "Dhaka"],
        }
    )

    pipeline = build_preprocessing_pipeline(dataframe)

    transformed = pipeline.fit_transform(dataframe)

    assert transformed.shape[0] == 4
    assert transformed.shape[1] == 4


def test_preprocessing_handles_unseen_categories():
    train = pd.DataFrame(
        {
            "age": [20, 30, 40],
            "city": ["Dhaka", "Dhaka", "Sylhet"],
        }
    )

    test = pd.DataFrame(
        {
            "age": [25],
            "city": ["Chittagong"],
        }
    )

    pipeline = build_preprocessing_pipeline(train)

    pipeline.fit(train)

    transformed = pipeline.transform(test)

    assert transformed.shape[0] == 1
    assert transformed.shape[1] == 3


def test_preprocessing_handles_only_numerical_columns():
    dataframe = pd.DataFrame(
        {
            "age": [20, 30, 40],
            "salary": [20000, 30000, 40000],
        }
    )

    pipeline = build_preprocessing_pipeline(dataframe)

    transformed = pipeline.fit_transform(dataframe)

    assert transformed.shape == (3, 2)


def test_preprocessing_handles_only_categorical_columns():
    dataframe = pd.DataFrame(
        {
            "city": ["Dhaka", "Sylhet", "Dhaka"],
        }
    )

    pipeline = build_preprocessing_pipeline(dataframe)

    transformed = pipeline.fit_transform(dataframe)

    assert transformed.shape == (3, 2)