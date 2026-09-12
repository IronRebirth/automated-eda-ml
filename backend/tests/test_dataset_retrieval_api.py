import pandas as pd
from fastapi.testclient import TestClient

from backend.app.db.database import SessionLocal, init_db
from backend.app.main import app
from backend.app.services.datasets import (
    create_dataset_record,
    get_dataset_storage_path,
)

client = TestClient(app)


def test_get_dataset():
    init_db()

    dataframe = pd.DataFrame(
        {
            "age": [20, 30, None],
            "city": ["Dhaka", "Dhaka", "Chittagong"],
            "active": [True, False, True],
        }
    )

    db = SessionLocal()

    dataset = None

    try:
        dataset = create_dataset_record(
            db=db,
            filename="retrieval-test.csv",
            dataframe=dataframe,
        )

        dataset_id = dataset.id

        response = client.get(
            f"/datasets/{dataset_id}"
        )

        assert response.status_code == 200

        response_data = response.json()

        assert response_data["dataset_id"] == dataset_id
        assert response_data["filename"] == "retrieval-test.csv"
        assert response_data["rows"] == 3
        assert response_data["columns"] == 3
        assert response_data["created_at"] is not None

        assert "profile" in response_data

        profile = response_data["profile"]

        assert profile["rows"] == 3
        assert profile["columns"] == 3
        assert profile["missing_values"] == 1
        assert profile["duplicate_rows"] == 0

        assert len(profile["column_profiles"]) == 3

    finally:
        if dataset is not None:
            storage_path = get_dataset_storage_path(
                dataset.id
            )

            if storage_path.exists():
                storage_path.unlink()

            db.delete(dataset)
            db.commit()

        db.close()


def test_get_dataset_not_found():
    response = client.get(
        "/datasets/999999999"
    )

    assert response.status_code == 404
    assert response.json() == {
        "detail": "Dataset not found: 999999999",
    }


def test_get_dataset_when_storage_file_is_missing():
    init_db()

    dataframe = pd.DataFrame(
        {
            "value": [1, 2, 3],
        }
    )

    db = SessionLocal()

    dataset = None

    try:
        dataset = create_dataset_record(
            db=db,
            filename="missing-file.csv",
            dataframe=dataframe,
        )

        storage_path = get_dataset_storage_path(
            dataset.id
        )

        storage_path.unlink()

        response = client.get(
            f"/datasets/{dataset.id}"
        )

        assert response.status_code == 404
        assert response.json() == {
            "detail": "Stored dataset file is unavailable.",
        }

    finally:
        if dataset is not None:
            storage_path = get_dataset_storage_path(
                dataset.id
            )

            if storage_path.exists():
                storage_path.unlink()

            db.delete(dataset)
            db.commit()

        db.close()