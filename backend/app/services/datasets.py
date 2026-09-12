from pathlib import Path

import pandas as pd
from sqlalchemy.orm import Session

from backend.app.db.models import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_STORAGE_DIR = PROJECT_ROOT / "datasets" / "raw"


def get_dataset_storage_path(dataset_id: int) -> Path:
    """Return the filesystem path for a persisted dataset."""

    return DATASET_STORAGE_DIR / f"{dataset_id}.csv"


def create_dataset_record(
    db: Session,
    filename: str,
    dataframe: pd.DataFrame,
) -> Dataset:
    """Create a dataset record and persist its CSV data."""

    DATASET_STORAGE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset = Dataset(
        filename=filename,
        rows=len(dataframe),
        columns=len(dataframe.columns),
    )

    db.add(dataset)
    db.flush()

    storage_path = get_dataset_storage_path(dataset.id)

    try:
        dataframe.to_csv(
            storage_path,
            index=False,
        )

        db.commit()
        db.refresh(dataset)

    except Exception:
        db.rollback()

        if storage_path.exists():
            storage_path.unlink()

        raise

    return dataset


def load_dataset_record(
    dataset: Dataset,
) -> pd.DataFrame:
    """Load the persisted CSV data for a dataset record."""

    storage_path = get_dataset_storage_path(dataset.id)

    if not storage_path.exists():
        raise FileNotFoundError(
            f"Stored dataset file not found: {storage_path}"
        )

    return pd.read_csv(storage_path)