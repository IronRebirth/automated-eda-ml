from io import BytesIO

import pandas as pd
from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def test_upload_dataset():
    csv_content = (
        "name,age,city\n"
        "Alice,25,Dhaka\n"
        "Bob,30,Chittagong\n"
    )

    response = client.post(
        "/datasets/upload",
        files={
            "file": (
                "customers.csv",
                BytesIO(
                    csv_content.encode("utf-8")
                ),
                "text/csv",
            )
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "filename": "customers.csv",
        "rows": 2,
        "columns": 3,
        "column_names": [
            "name",
            "age",
            "city",
        ],
    }


def test_upload_dataset_rejects_non_csv():
    response = client.post(
        "/datasets/upload",
        files={
            "file": (
                "customers.txt",
                BytesIO(
                    b"not a csv file"
                ),
                "text/plain",
            )
        },
    )

    assert response.status_code == 400
    assert response.json() == {
        "detail": "Only CSV files are supported.",
    }


def test_upload_dataset_rejects_invalid_csv():
    response = client.post(
        "/datasets/upload",
        files={
            "file": (
                "customers.csv",
                BytesIO(
                    b"\xff\xfe\x00\x00"
                ),
                "text/csv",
            )
        },
    )

    assert response.status_code == 400
    assert "Unable to read CSV file" in (
        response.json()["detail"]
    )


def test_inspect_dataset():
    csv_content = (
        "name,age,city\n"
        "Alice,25,Dhaka\n"
        "Bob,30,Chittagong\n"
        "Charlie,,Dhaka\n"
    )

    response = client.post(
        "/datasets/inspect",
        files={
            "file": (
                "customers.csv",
                BytesIO(
                    csv_content.encode("utf-8")
                ),
                "text/csv",
            )
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["filename"] == "customers.csv"

    profile = data["profile"]

    assert profile["rows"] == 3
    assert profile["columns"] == 3
    assert profile["column_names"] == [
        "name",
        "age",
        "city",
    ]
    assert profile["numerical_columns"] == [
        "age"
    ]
    assert profile["categorical_columns"] == [
        "name",
        "city",
    ]
    assert profile["datetime_columns"] == []
    assert profile["missing_values"] == 1
    assert profile["duplicate_rows"] == 0
    assert profile["constant_columns"] == []


def test_inspect_dataset_rejects_non_csv():
    response = client.post(
        "/datasets/inspect",
        files={
            "file": (
                "customers.json",
                BytesIO(
                    b'{"name": "Alice"}'
                ),
                "application/json",
            )
        },
    )

    assert response.status_code == 400
    assert response.json() == {
        "detail": "Only CSV files are supported.",
    }


def test_quality_dataset():
    csv_content = (
        "name,age,city\n"
        "Alice,25,Dhaka\n"
        "Bob,30,Chittagong\n"
        "Alice,25,Dhaka\n"
        "Charlie,,Dhaka\n"
    )

    response = client.post(
        "/datasets/quality",
        files={
            "file": (
                "customers.csv",
                BytesIO(
                    csv_content.encode("utf-8")
                ),
                "text/csv",
            )
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["filename"] == "customers.csv"

    quality = data["quality"]

    assert set(quality.keys()) == {
        "missing_values",
        "duplicates",
        "cardinality",
        "outliers",
    }

    assert quality["missing_values"] is not None
    assert quality["duplicates"] is not None
    assert quality["cardinality"] is not None
    assert quality["outliers"] is not None


def test_quality_dataset_rejects_non_csv():
    response = client.post(
        "/datasets/quality",
        files={
            "file": (
                "customers.json",
                BytesIO(
                    b'{"name": "Alice"}'
                ),
                "application/json",
            )
        },
    )

    assert response.status_code == 400
    assert response.json() == {
        "detail": "Only CSV files are supported.",
    }


def test_eda_dataset():
    csv_content = (
        "name,age,city\n"
        "Alice,25,Dhaka\n"
        "Bob,30,Chittagong\n"
        "Charlie,35,Dhaka\n"
    )

    response = client.post(
        "/datasets/eda",
        files={
            "file": (
                "customers.csv",
                BytesIO(
                    csv_content.encode("utf-8")
                ),
                "text/csv",
            )
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["filename"] == "customers.csv"
    assert data["target_column"] is None

    eda = data["eda"]

    assert set(eda.keys()) == {
        "numerical",
        "categorical",
        "correlations",
        "target",
        "insights",
        "visualizations",
    }

    assert eda["target"] is None
    assert eda["numerical"] is not None
    assert eda["categorical"] is not None
    assert eda["correlations"] is not None
    assert eda["insights"] is not None
    assert eda["visualizations"] is not None


def test_eda_dataset_with_target():
    csv_content = (
        "age,income,city,target\n"
        "25,50000,Dhaka,0\n"
        "30,60000,Chittagong,1\n"
        "35,70000,Dhaka,1\n"
        "40,80000,Sylhet,0\n"
    )

    response = client.post(
        "/datasets/eda?target_column=target",
        files={
            "file": (
                "customers.csv",
                BytesIO(
                    csv_content.encode("utf-8")
                ),
                "text/csv",
            )
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["filename"] == "customers.csv"
    assert data["target_column"] == "target"
    assert data["eda"]["target"] is not None


def test_eda_dataset_rejects_invalid_target():
    csv_content = (
        "name,age,city\n"
        "Alice,25,Dhaka\n"
        "Bob,30,Chittagong\n"
    )

    response = client.post(
        "/datasets/eda?target_column=missing",
        files={
            "file": (
                "customers.csv",
                BytesIO(
                    csv_content.encode("utf-8")
                ),
                "text/csv",
            )
        },
    )

    assert response.status_code == 400
    assert response.json() == {
        "detail": "Target column not found: missing",
    }


def test_eda_dataset_rejects_non_csv():
    response = client.post(
        "/datasets/eda",
        files={
            "file": (
                "customers.json",
                BytesIO(
                    b'{"name": "Alice"}'
                ),
                "application/json",
            )
        },
    )

    assert response.status_code == 400
    assert response.json() == {
        "detail": "Only CSV files are supported.",
    }


def test_run_ml_pipeline():
    csv_content = (
        "age,income,churn\n"
        "25,30000,0\n"
        "30,40000,0\n"
        "35,50000,1\n"
        "40,60000,1\n"
        "45,70000,1\n"
        "50,80000,1\n"
        "28,35000,0\n"
        "32,45000,0\n"
        "38,55000,1\n"
        "42,65000,1\n"
    )

    response = client.post(
        "/datasets/run",
        files={
            "file": (
                "customers.csv",
                BytesIO(
                    csv_content.encode("utf-8")
                ),
                "text/csv",
            )
        },
        data={
            "target_column": "churn",
        },
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["filename"] == "customers.csv"

    run = payload["run"]

    assert run["task_type"] == "classification"
    assert run["target_column"] == "churn"

    assert "evaluation" in run
    assert "cross_validation" in run
    assert "leaderboard" in run
    assert "optimized_evaluation" in run
    assert "best_model" in run
    assert "explainability" in run

    assert "model_name" in run["best_model"]
    assert "metrics" in run["best_model"]

    assert "summary" in run["explainability"]
    assert "insights" in run["explainability"]
    assert "metadata" in run["explainability"]


def test_run_ml_pipeline_rejects_missing_target():
    csv_content = (
        "age,income,churn\n"
        "25,30000,0\n"
        "30,40000,1\n"
    )

    response = client.post(
        "/datasets/run",
        files={
            "file": (
                "customers.csv",
                BytesIO(
                    csv_content.encode("utf-8")
                ),
                "text/csv",
            )
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Target column is required."
    )


def test_run_ml_pipeline_rejects_invalid_target():
    csv_content = (
        "age,income,churn\n"
        "25,30000,0\n"
        "30,40000,1\n"
    )

    response = client.post(
        "/datasets/run",
        files={
            "file": (
                "customers.csv",
                BytesIO(
                    csv_content.encode("utf-8")
                ),
                "text/csv",
            )
        },
        data={
            "target_column": "missing",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Target column not found: missing"
    )


def test_run_ml_pipeline_rejects_invalid_test_size():
    csv_content = (
        "age,income,churn\n"
        "25,30000,0\n"
        "30,40000,1\n"
    )

    response = client.post(
        "/datasets/run",
        files={
            "file": (
                "customers.csv",
                BytesIO(
                    csv_content.encode("utf-8")
                ),
                "text/csv",
            )
        },
        data={
            "target_column": "churn",
            "test_size": "1.0",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "test_size must be between 0 and 1."
    )


def test_run_ml_pipeline_rejects_non_csv():
    response = client.post(
        "/datasets/run",
        files={
            "file": (
                "customers.txt",
                BytesIO(
                    b"age,income,churn\n"
                    b"25,30000,0\n"
                ),
                "text/plain",
            )
        },
        data={
            "target_column": "churn",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Only CSV files are supported."
    )


def test_predict_dataset():
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler

    from ml.models import save_model_artifact

    preprocessing = StandardScaler()

    training_data = pd.DataFrame(
        {
            "age": [20, 30, 40, 50],
        }
    )

    preprocessing.fit(training_data)

    model = LinearRegression()
    model.fit(
        preprocessing.transform(training_data),
        [20000, 30000, 40000, 50000],
    )

    artifact_path = (
        "artifacts/test_prediction_model.joblib"
    )

    save_model_artifact(
        model,
        preprocessing,
        artifact_path,
    )

    csv_content = (
        "age\n"
        "25\n"
        "35\n"
        "45\n"
    )

    response = client.post(
        "/datasets/predict",
        files={
            "file": (
                "customers.csv",
                BytesIO(
                    csv_content.encode("utf-8")
                ),
                "text/csv",
            )
        },
        data={
            "artifact_path": artifact_path,
        },
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["filename"] == "customers.csv"
    assert len(payload["predictions"]) == 3


def test_predict_dataset_rejects_missing_artifact_path():
    csv_content = (
        "age\n"
        "25\n"
        "35\n"
    )

    response = client.post(
        "/datasets/predict",
        files={
            "file": (
                "customers.csv",
                BytesIO(
                    csv_content.encode("utf-8")
                ),
                "text/csv",
            )
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Artifact path is required."
    )


def test_predict_dataset_rejects_missing_artifact():
    csv_content = (
        "age\n"
        "25\n"
        "35\n"
    )

    response = client.post(
        "/datasets/predict",
        files={
            "file": (
                "customers.csv",
                BytesIO(
                    csv_content.encode("utf-8")
                ),
                "text/csv",
            )
        },
        data={
            "artifact_path": (
                "artifacts/missing_model.joblib"
            ),
        },
    )

    assert response.status_code == 404
    assert "Model artifact not found" in (
        response.json()["detail"]
    )


def test_predict_dataset_rejects_non_csv():
    response = client.post(
        "/datasets/predict",
        files={
            "file": (
                "customers.txt",
                BytesIO(
                    b"age\n"
                    b"25\n"
                ),
                "text/plain",
            )
        },
        data={
            "artifact_path": (
                "artifacts/missing_model.joblib"
            ),
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Only CSV files are supported."
    )