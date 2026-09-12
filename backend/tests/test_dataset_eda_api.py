from io import BytesIO

from fastapi.testclient import TestClient

from backend.app.db.database import SessionLocal
from backend.app.db.models import Dataset
from backend.app.main import app

client = TestClient(app)


def _create_dataset() -> int:
    csv_content = (
        "name,age,city\n"
        "Alice,25,Dhaka\n"
        "Bob,30,Chittagong\n"
        "Charlie,35,Dhaka\n"
        "David,40,Sylhet\n"
    )

    dataframe_response = client.post(
        "/datasets/upload",
        files={
            "file": (
                "customers.csv",
                BytesIO(csv_content.encode("utf-8")),
                "text/csv",
            )
        },
    )

    assert dataframe_response.status_code == 200

    return dataframe_response.json()["dataset_id"]


def _delete_dataset(dataset_id: int) -> None:
    db = SessionLocal()

    try:
        dataset = db.get(
            Dataset,
            dataset_id,
        )

        if dataset is not None:
            db.delete(dataset)
            db.commit()

    finally:
        db.close()


def test_get_dataset_eda():
    dataset_id = _create_dataset()

    try:
        response = client.get(
            f"/datasets/{dataset_id}/eda"
        )

        assert response.status_code == 200

        data = response.json()

        assert data["dataset_id"] == dataset_id
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
        assert isinstance(eda["insights"], list)
        assert eda["visualizations"] is not None

    finally:
        _delete_dataset(dataset_id)


def test_get_dataset_eda_with_target():
    csv_content = (
        "age,income,city,target\n"
        "25,50000,Dhaka,0\n"
        "30,60000,Chittagong,1\n"
        "35,70000,Dhaka,1\n"
        "40,80000,Sylhet,0\n"
    )

    dataframe_response = client.post(
        "/datasets/upload",
        files={
            "file": (
                "customers.csv",
                BytesIO(csv_content.encode("utf-8")),
                "text/csv",
            )
        },
    )

    assert dataframe_response.status_code == 200

    dataset_id = dataframe_response.json()["dataset_id"]

    try:
        response = client.get(
            f"/datasets/{dataset_id}/eda",
            params={
                "target_column": "target",
            },
        )

        assert response.status_code == 200

        data = response.json()

        assert data["target_column"] == "target"
        assert data["eda"]["target"] is not None
        assert (
            data["eda"]["target"]["column"]
            == "target"
        )

    finally:
        _delete_dataset(dataset_id)


def test_get_dataset_eda_rejects_invalid_target():
    dataset_id = _create_dataset()

    try:
        response = client.get(
            f"/datasets/{dataset_id}/eda",
            params={
                "target_column": "missing",
            },
        )

        assert response.status_code == 400
        assert response.json() == {
            "detail": "Target column not found: missing",
        }

    finally:
        _delete_dataset(dataset_id)


def test_get_dataset_eda_rejects_missing_dataset():
    response = client.get(
        "/datasets/999999/eda"
    )

    assert response.status_code == 404
    assert response.json() == {
        "detail": "Dataset not found: 999999",
    }