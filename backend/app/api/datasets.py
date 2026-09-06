import json
from io import BytesIO

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile
from plotly.utils import PlotlyJSONEncoder

from ml.eda import EDAAnalyzer
from ml.profiling import DatasetProfiler
from ml.quality import DataQualityAnalyzer

router = APIRouter(
    prefix="/datasets",
    tags=["datasets"],
)


UPLOAD_FILE = File(...)


@router.post("/upload")
async def upload_dataset(
    file: UploadFile = UPLOAD_FILE,
) -> dict:
    """Upload and inspect a CSV dataset."""

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is required.",
        )

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are supported.",
        )

    try:
        contents = await file.read()
        dataframe = pd.read_csv(
            BytesIO(contents)
        )
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Unable to read CSV file: {exc}",
        ) from exc

    return {
        "filename": file.filename,
        "rows": len(dataframe),
        "columns": len(dataframe.columns),
        "column_names": dataframe.columns.tolist(),
    }


@router.post("/inspect")
async def inspect_dataset(
    file: UploadFile = UPLOAD_FILE,
) -> dict:
    """Profile an uploaded CSV dataset."""

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is required.",
        )

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are supported.",
        )

    try:
        contents = await file.read()
        dataframe = pd.read_csv(
            BytesIO(contents)
        )
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Unable to read CSV file: {exc}",
        ) from exc

    profile = DatasetProfiler(
        dataframe
    ).profile()

    return {
        "filename": file.filename,
        "profile": profile,
    }


@router.post("/quality")
async def analyze_dataset_quality(
    file: UploadFile = UPLOAD_FILE,
) -> dict:
    """Analyze the data quality of an uploaded CSV dataset."""

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is required.",
        )

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are supported.",
        )

    try:
        contents = await file.read()
        dataframe = pd.read_csv(
            BytesIO(contents)
        )
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Unable to read CSV file: {exc}",
        ) from exc

    quality_report = DataQualityAnalyzer(
        dataframe
    ).analyze()

    return {
        "filename": file.filename,
        "quality": quality_report,
    }


def _serialize_eda_visualizations(
    visualizations: dict,
) -> dict:
    """Convert Plotly figures into JSON-safe dictionaries."""

    serialized = {}

    for category, figures in visualizations.items():
        serialized[category] = [
            json.loads(
                json.dumps(
                    figure,
                    cls=PlotlyJSONEncoder,
                )
            )
            for figure in figures
        ]

    return serialized


@router.post("/eda")
async def analyze_dataset_eda(
    file: UploadFile = UPLOAD_FILE,
    target_column: str | None = None,
) -> dict:
    """Run exploratory data analysis on an uploaded CSV dataset."""

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is required.",
        )

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are supported.",
        )

    try:
        contents = await file.read()
        dataframe = pd.read_csv(
            BytesIO(contents)
        )
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Unable to read CSV file: {exc}",
        ) from exc

    if (
        target_column is not None
        and target_column not in dataframe.columns
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Target column not found: "
                f"{target_column}"
            ),
        )

    eda_report = EDAAnalyzer(
        dataframe,
        target_column=target_column,
    ).analyze()

    eda_report["visualizations"] = (
        _serialize_eda_visualizations(
            eda_report["visualizations"]
        )
    )

    return {
        "filename": file.filename,
        "target_column": target_column,
        "eda": eda_report,
    }