from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from backend.app.db.database import SessionLocal, init_db
from backend.app.db.models import Dataset
from backend.app.main import app
from backend.app.services.runs import create_run_record

client = TestClient(app)


def _create_dataset(
    db: Session,
    filename: str,
    rows: int = 3,
    columns: int = 2,
) -> Dataset:
    """Create a dataset record for API testing."""

    dataset = Dataset(
        filename=filename,
        rows=rows,
        columns=columns,
    )

    db.add(dataset)
    db.commit()
    db.refresh(dataset)

    return dataset


def test_list_runs():
    init_db()

    db = SessionLocal()

    dataset = _create_dataset(
        db,
        "list-runs.csv",
    )

    run = create_run_record(
        db=db,
        dataset_id=dataset.id,
        target_column="target",
        test_size=0.2,
        random_state=42,
    )

    run_id = run.id
    dataset_id = dataset.id

    db.close()

    try:
        response = client.get("/runs")

        assert response.status_code == 200

        response_data = response.json()

        assert response_data["limit"] == 20
        assert response_data["offset"] == 0
        assert response_data["count"] >= 1

        runs = response_data["runs"]

        matching_runs = [
            item
            for item in runs
            if item["run_id"] == run_id
        ]

        assert len(matching_runs) == 1

        retrieved_run = matching_runs[0]

        assert retrieved_run["dataset_id"] == dataset_id
        assert retrieved_run["status"] == "pending"
        assert retrieved_run["target_column"] == "target"
        assert retrieved_run["test_size"] == 0.2
        assert retrieved_run["random_state"] == 42

    finally:
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


def test_list_runs_filtered_by_dataset():
    init_db()

    db = SessionLocal()

    dataset_one = _create_dataset(
        db,
        "dataset-one.csv",
    )

    dataset_two = _create_dataset(
        db,
        "dataset-two.csv",
    )

    run_one = create_run_record(
        db=db,
        dataset_id=dataset_one.id,
        target_column="target",
    )

    run_two = create_run_record(
        db=db,
        dataset_id=dataset_two.id,
        target_column="label",
    )

    dataset_one_id = dataset_one.id
    dataset_two_id = dataset_two.id
    run_one_id = run_one.id
    run_two_id = run_two.id

    db.close()

    try:
        response = client.get(
            f"/runs?dataset_id={dataset_one_id}"
        )

        assert response.status_code == 200

        response_data = response.json()

        assert response_data["count"] >= 1

        runs = response_data["runs"]

        assert all(
            item["dataset_id"] == dataset_one_id
            for item in runs
        )

        assert any(
            item["run_id"] == run_one_id
            for item in runs
        )

        assert not any(
            item["run_id"] == run_two_id
            for item in runs
        )

    finally:
        db = SessionLocal()

        try:
            dataset = db.get(
                Dataset,
                dataset_one_id,
            )

            if dataset is not None:
                db.delete(dataset)

            dataset = db.get(
                Dataset,
                dataset_two_id,
            )

            if dataset is not None:
                db.delete(dataset)

            db.commit()

        finally:
            db.close()


def test_list_runs_pagination():
    init_db()

    db = SessionLocal()

    dataset = _create_dataset(
        db,
        "pagination.csv",
    )

    run_ids = []

    for index in range(3):
        run = create_run_record(
            db=db,
            dataset_id=dataset.id,
            target_column=f"target_{index}",
        )

        run_ids.append(run.id)

    dataset_id = dataset.id

    db.close()

    try:
        response = client.get(
            "/runs?limit=2&offset=1"
        )

        assert response.status_code == 200

        response_data = response.json()

        assert response_data["limit"] == 2
        assert response_data["offset"] == 1
        assert response_data["count"] == 2
        assert len(response_data["runs"]) == 2

        returned_ids = [
            item["run_id"]
            for item in response_data["runs"]
        ]

        assert all(
            run_id in run_ids
            for run_id in returned_ids
        )

    finally:
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


def test_list_runs_rejects_invalid_pagination():
    response = client.get(
        "/runs?limit=0"
    )

    assert response.status_code == 422

    response = client.get(
        "/runs?limit=101"
    )

    assert response.status_code == 422

    response = client.get(
        "/runs?offset=-1"
    )

    assert response.status_code == 422