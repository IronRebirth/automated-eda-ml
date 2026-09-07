import pandas as pd
from sqlalchemy.orm import Session

from backend.app.db.models import Dataset


def create_dataset_record(
    db: Session,
    filename: str,
    dataframe: pd.DataFrame,
) -> Dataset:
    """Create and persist a dataset metadata record."""

    dataset = Dataset(
        filename=filename,
        rows=len(dataframe),
        columns=len(dataframe.columns),
    )

    db.add(dataset)
    db.commit()
    db.refresh(dataset)

    return dataset