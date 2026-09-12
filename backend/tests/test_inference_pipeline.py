import pandas as pd

from ml.models.inference import predict_from_artifact
from ml.pipeline import MLPipeline


def test_pipeline_artifact_can_generate_predictions(
    tmp_path,
):
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

    artifact_path = (
        tmp_path
        / "artifacts"
        / "best_model.joblib"
    )

    pipeline = MLPipeline(
        dataframe,
        target_column="churn",
        test_size=0.2,
        random_state=42,
        cv=3,
        optimization_trials=2,
        artifact_path=artifact_path,
    )

    result = pipeline.run()

    predictions = predict_from_artifact(
        result["artifact_path"],
        dataframe.drop(
            columns=["churn"]
        ),
    )

    assert len(predictions) == len(dataframe)

    assert all(
        prediction in {0, 1}
        for prediction in predictions
    )