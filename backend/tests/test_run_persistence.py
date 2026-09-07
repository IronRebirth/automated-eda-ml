from backend.app.db.database import SessionLocal, init_db
from backend.app.db.models import Dataset, Run
from backend.app.services.runs import (
    create_run_record,
    update_run_status,
)


def test_create_run_record():
    init_db()

    db = SessionLocal()

    dataset = Dataset(
        filename="customers.csv",
        rows=3,
        columns=2,
    )

    db.add(dataset)
    db.commit()
    db.refresh(dataset)

    run = None

    try:
        run = create_run_record(
            db=db,
            dataset_id=dataset.id,
            target_column="age",
            test_size=0.2,
            random_state=42,
        )

        assert run.id is not None
        assert run.dataset_id == dataset.id
        assert run.status == "pending"
        assert run.target_column == "age"
        assert run.test_size == 0.2
        assert run.random_state == 42
        assert run.created_at is not None
        assert run.completed_at is None
    finally:
        if run is not None:
            db.delete(run)

        db.delete(dataset)
        db.commit()
        db.close()


def test_update_run_status():
    init_db()

    db = SessionLocal()

    dataset = Dataset(
        filename="sample.csv",
        rows=3,
        columns=1,
    )

    db.add(dataset)
    db.commit()
    db.refresh(dataset)

    run = None

    try:
        run = create_run_record(
            db=db,
            dataset_id=dataset.id,
        )

        updated = update_run_status(
            db=db,
            run=run,
            status="completed",
        )

        assert updated.status == "completed"
        assert updated.completed_at is not None
    finally:
        if run is not None:
            db.delete(run)

        db.delete(dataset)
        db.commit()
        db.close()


def test_run_can_be_retrieved():
    init_db()

    db = SessionLocal()

    dataset = Dataset(
        filename="sample.csv",
        rows=5,
        columns=2,
    )

    db.add(dataset)
    db.commit()
    db.refresh(dataset)

    run = None

    try:
        run = create_run_record(
            db=db,
            dataset_id=dataset.id,
            target_column="target",
        )

        retrieved = db.get(
            Run,
            run.id,
        )

        assert retrieved is not None
        assert retrieved.dataset_id == dataset.id
        assert retrieved.status == "pending"
        assert retrieved.target_column == "target"
    finally:
        if run is not None:
            db.delete(run)

        db.delete(dataset)
        db.commit()
        db.close()