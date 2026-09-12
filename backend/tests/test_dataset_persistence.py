import pandas as pd

from backend.app.db.database import SessionLocal, init_db
from backend.app.db.models import Dataset
from backend.app.services.datasets import create_dataset_record


def test_create_dataset_record():
    init_db()

    dataframe = pd.DataFrame(
        {
            "name": ["Alice", "Bob", "Charlie"],
            "age": [25, 30, 35],
        }
    )

    db = SessionLocal()

    try:
        dataset = create_dataset_record(
            db=db,
            filename="customers.csv",
            dataframe=dataframe,
        )

        assert dataset.id is not None
        assert dataset.filename == "customers.csv"
        assert dataset.rows == 3
        assert dataset.columns == 2
        assert dataset.created_at is not None
    finally:
        db.delete(dataset)
        db.commit()
        db.close()


def test_dataset_record_can_be_retrieved():
    init_db()

    dataframe = pd.DataFrame(
        {
            "feature": [1, 2, 3],
        }
    )

    db = SessionLocal()

    try:
        dataset = create_dataset_record(
            db=db,
            filename="sample.csv",
            dataframe=dataframe,
        )

        retrieved = db.get(
            Dataset,
            dataset.id,
        )

        assert retrieved is not None
        assert retrieved.filename == "sample.csv"
        assert retrieved.rows == 3
        assert retrieved.columns == 1
    finally:
        db.delete(dataset)
        db.commit()
        db.close()