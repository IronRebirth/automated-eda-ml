import pandas as pd

from ml.pipeline import MLPipeline


def test_ml_pipeline_runs_end_to_end():
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
    )

    result = pipeline.run()

    assert result["task_type"] == "classification"
    assert result["target_column"] == "churn"

    assert "logistic_regression" in result["models"]
    assert "random_forest" in result["models"]
    assert "xgboost" in result["models"]

    assert "logistic_regression" in result["evaluation"]
    assert "random_forest" in result["evaluation"]
    assert "xgboost" in result["evaluation"]

    assert "cross_validation" in result

    assert "logistic_regression" in result["cross_validation"]
    assert "random_forest" in result["cross_validation"]
    assert "xgboost" in result["cross_validation"]

    assert len(result["leaderboard"]) == 3

    assert "explainability" in result

    explainability = result["explainability"]

    assert "feature_importance" in explainability
    assert "shap_values" in explainability
    assert "transformed_data" in explainability
    assert "feature_names" in explainability

    assert len(
        explainability["feature_importance"]
    ) == len(
        explainability["feature_names"]
    )

    assert (
        explainability["transformed_data"].shape[1]
        == len(explainability["feature_names"])
    )

    assert all(
        explainability["feature_importance"]["importance"]
        >= 0
    )

    # 5.7: structured explainability summary
    assert "summary" in explainability

    summary = explainability["summary"]

    assert "top_n" in summary
    assert "features" in summary
    assert summary["top_n"] > 0
    assert len(summary["features"]) == summary["top_n"]

    first_feature = summary["features"][0]

    assert "rank" in first_feature
    assert "feature" in first_feature
    assert "importance" in first_feature
    assert "relative_importance" in first_feature
    assert "impact" in first_feature

    assert first_feature["rank"] == 1
    assert first_feature["importance"] >= 0
    assert 0 <= first_feature["relative_importance"] <= 1
    assert first_feature["impact"] in {
        "high",
        "medium",
        "low",
    }

    # 5.7: human-readable explainability insights
    assert "insights" in explainability

    insights = explainability["insights"]

    assert isinstance(insights, list)
    assert len(insights) > 0
    assert all(
        isinstance(insight, str)
        for insight in insights
    )

    # 5.9: explainability metadata
    assert "metadata" in explainability

    metadata = explainability["metadata"]

    assert metadata["method"] == "SHAP"
    assert metadata["model"] == result[
        "best_model"
    ]["model_name"]
    assert metadata["task_type"] == "classification"
    assert metadata["feature_count"] == len(
        explainability["feature_names"]
    )
    assert metadata["top_features_count"] == (
        summary["top_n"]
    )
    assert metadata["preprocessing_applied"] is True


def test_ml_pipeline_detects_regression():
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
            ],
            "salary": [
                20000.5,
                25000.2,
                30000.8,
                35000.4,
                40000.1,
                45000.7,
                50000.3,
                55000.9,
                60000.6,
                65000.8,
            ],
        }
    )

    pipeline = MLPipeline(
        dataframe,
        target_column="salary",
        test_size=0.2,
        random_state=42,
    )

    result = pipeline.run()

    assert result["task_type"] == "regression"
    assert result["target_column"] == "salary"

    assert "linear_regression" in result["models"]
    assert "random_forest" in result["models"]
    assert "xgboost" in result["models"]

    assert "linear_regression" in result["evaluation"]
    assert "random_forest" in result["evaluation"]
    assert "xgboost" in result["evaluation"]

    assert "cross_validation" in result

    assert "linear_regression" in result["cross_validation"]
    assert "random_forest" in result["cross_validation"]
    assert "xgboost" in result["cross_validation"]

    assert len(result["leaderboard"]) == 3

    assert "explainability" in result

    explainability = result["explainability"]

    assert "summary" in explainability
    assert "insights" in explainability

    assert explainability["summary"]["top_n"] > 0
    assert len(explainability["summary"]["features"]) > 0

    assert isinstance(
        explainability["insights"],
        list,
    )
    assert len(explainability["insights"]) > 0

    # 5.9: explainability metadata
    assert "metadata" in explainability

    metadata = explainability["metadata"]

    assert metadata["method"] == "SHAP"
    assert metadata["model"] == result[
        "best_model"
    ]["model_name"]
    assert metadata["task_type"] == "regression"
    assert metadata["feature_count"] == len(
        explainability["feature_names"]
    )
    assert metadata["top_features_count"] == (
        explainability["summary"]["top_n"]
    )
    assert metadata["preprocessing_applied"] is True