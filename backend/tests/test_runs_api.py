from fastapi.testclient import TestClient

from backend.app.db.database import SessionLocal, init_db
from backend.app.db.models import Dataset
from backend.app.main import app
from backend.app.services.runs import create_run_record

client = TestClient(app)


def test_get_run():
    init_db()

    db = SessionLocal()

    try:
        dataset = Dataset(
            filename="test.csv",
            rows=3,
            columns=2,
        )

        db.add(dataset)
        db.commit()
        db.refresh(dataset)

        run = create_run_record(
            db=db,
            dataset_id=dataset.id,
            target_column="target",
            test_size=0.2,
            random_state=42,
        )

        run_id = run.id
        dataset_id = dataset.id

    finally:
        db.close()

    try:
        response = client.get(
            f"/runs/{run_id}"
        )

        assert response.status_code == 200

        response_data = response.json()

        assert response_data["run_id"] == run_id
        assert response_data["dataset_id"] == dataset_id
        assert response_data["status"] == "pending"
        assert response_data["target_column"] == "target"
        assert response_data["test_size"] == 0.2
        assert response_data["random_state"] == 42
        assert response_data["created_at"] is not None
        assert response_data["completed_at"] is None

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


def test_get_run_not_found():
    response = client.get(
        "/runs/999999999"
    )

    assert response.status_code == 404
    assert response.json() == {
        "detail": "Run not found: 999999999",
    }