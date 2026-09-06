from io import BytesIO

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