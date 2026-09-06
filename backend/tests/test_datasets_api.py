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