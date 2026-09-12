import json

from fastapi import APIRouter, HTTPException
from plotly.utils import PlotlyJSONEncoder
from sqlalchemy.orm import Session

from backend.app.db.database import SessionLocal
from backend.app.db.models import Dataset
from backend.app.services.datasets import load_dataset_record
from ml.eda import EDAAnalyzer

router = APIRouter(
    prefix="/datasets",
    tags=["datasets"],
)


def _serialize_eda_visualizations(
    visualizations: dict,
) -> dict:
    """Convert Plotly figures into JSON-safe dictionaries."""

    serialized = {}

    for category, figures in visualizations.items():
        if isinstance(figures, dict):
            figure_values = figures.values()
        else:
            figure_values = figures

        serialized[category] = [
            json.loads(
                json.dumps(
                    figure,
                    cls=PlotlyJSONEncoder,
                )
            )
            for figure in figure_values
        ]

    return serialized


@router.get("/{dataset_id}/eda")
def get_dataset_eda(
    dataset_id: int,
    target_column: str | None = None,
) -> dict:
    """Run automated EDA against a persisted dataset."""

    db: Session = SessionLocal()

    try:
        dataset = db.get(
            Dataset,
            dataset_id,
        )

        if dataset is None:
            raise HTTPException(
                status_code=404,
                detail=f"Dataset not found: {dataset_id}",
            )

        try:
            dataframe = load_dataset_record(
                dataset
            )

        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=404,
                detail="Stored dataset file is unavailable.",
            ) from exc

        normalized_target = (
            target_column.strip()
            if target_column
            else None
        )

        if (
            normalized_target
            and normalized_target not in dataframe.columns
        ):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Target column not found: "
                    f"{normalized_target}"
                ),
            )

        analyzer = EDAAnalyzer(
            dataframe,
            target_column=normalized_target,
        )

        eda = analyzer.analyze()

        eda["visualizations"] = (
            _serialize_eda_visualizations(
                eda["visualizations"]
            )
        )

        return {
            "dataset_id": dataset.id,
            "filename": dataset.filename,
            "target_column": normalized_target,
            "eda": eda,
        }

    finally:
        db.close()