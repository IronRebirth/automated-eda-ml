import pandas as pd

from ml.pipeline import MLPipeline


def test_ml_pipeline_includes_hyperparameter_optimization():
    dataframe = pd.DataFrame(
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
                40,
                42,
                44,
                46,
                48,
                50,
                52,
                54,
                56,
                58,
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
                40000,
                42000,
                44000,
                46000,
                48000,
                50000,
                52000,
                54000,
                56000,
                58000,
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
                "Sylhet",
                "Dhaka",
                "Chittagong",
                "Dhaka",
                "Sylhet",
                "Dhaka",
                "Chittagong",
                "Dhaka",
                "Sylhet",
                "Dhaka",
            ],
            "churn": [
                0,
                0,
                0,
                0,
                0,
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
                1,
                1,
                1,
                1,
                1,
            ],
        }
    )

    pipeline = MLPipeline(
        dataframe,
        target_column="churn",
        test_size=0.2,
        random_state=42,
        cv=3,
        optimization_trials=2,
    )

    result = pipeline.run()

    assert "optimization" in result

    assert "random_forest" in result["optimization"]
    assert "xgboost" in result["optimization"]

    random_forest_result = result["optimization"][
        "random_forest"
    ]

    xgboost_result = result["optimization"]["xgboost"]

    assert "model" in random_forest_result
    assert "best_params" in random_forest_result
    assert "best_score" in random_forest_result
    assert "n_trials" in random_forest_result

    assert "model" in xgboost_result
    assert "best_params" in xgboost_result
    assert "best_score" in xgboost_result
    assert "n_trials" in xgboost_result

    assert random_forest_result["n_trials"] == 2
    assert xgboost_result["n_trials"] == 2

    assert hasattr(
        random_forest_result["model"],
        "predict",
    )

    assert hasattr(
        xgboost_result["model"],
        "predict",
    )


def test_ml_pipeline_optimization_supports_regression():
    dataframe = pd.DataFrame(
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
                40,
                42,
                44,
                46,
                48,
                50,
                52,
                54,
                56,
                58,
            ],
            "experience": [
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
            ],
            "salary": [
                20000.5,
                22000.2,
                24000.8,
                26000.4,
                28000.1,
                30000.7,
                32000.3,
                34000.9,
                36000.6,
                38000.8,
                40000.2,
                42000.5,
                44000.7,
                46000.3,
                48000.9,
                50000.4,
                52000.6,
                54000.1,
                56000.8,
                58000.5,
            ],
        }
    )

    pipeline = MLPipeline(
        dataframe,
        target_column="salary",
        test_size=0.2,
        random_state=42,
        cv=3,
        optimization_trials=2,
    )

    result = pipeline.run()

    assert result["task_type"] == "regression"

    assert "optimization" in result
    assert "random_forest" in result["optimization"]
    assert "xgboost" in result["optimization"]

    assert result["optimization"]["random_forest"][
        "n_trials"
    ] == 2

    assert result["optimization"]["xgboost"][
        "n_trials"
    ] == 2