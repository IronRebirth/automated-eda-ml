import json
from io import BytesIO

import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from plotly.utils import PlotlyJSONEncoder

from ml.eda import EDAAnalyzer
from ml.pipeline import MLPipeline
from ml.profiling import DatasetProfiler
from ml.quality import DataQualityAnalyzer

router = APIRouter(
    prefix="/datasets",
    tags=["datasets"],
)

UPLOAD_FILE = File(...)
TARGET_COLUMN = Form("")
TEST_SIZE = Form(0.2)
RANDOM_STATE = Form(42)


def _read_csv_file(file: UploadFile) -> pd.DataFrame:
    """Read an uploaded CSV file into a DataFrame."""

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
        contents = file.file.read()

        return pd.read_csv(
            BytesIO(contents)
        )

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Unable to read CSV file: {exc}",
        ) from exc


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


def _build_ml_run_response(
    result: dict,
) -> dict:
    """Build a JSON-safe API response from an ML pipeline result."""

    best_model = result["best_model"]
    explainability = result["explainability"]

    return {
        "task_type": result["task_type"],
        "target_column": result["target_column"],
        "evaluation": result["evaluation"],
        "cross_validation": result["cross_validation"],
        "leaderboard": result["leaderboard"],
        "optimized_evaluation": (
            result["optimized_evaluation"]
        ),
        "best_model": {
            "model_name": best_model["model_name"],
            "metrics": best_model["metrics"],
        },
        "explainability": {
            "summary": explainability["summary"],
            "insights": explainability["insights"],
            "metadata": explainability["metadata"],
        },
        "artifact_path": result["artifact_path"],
    }


@router.post("/upload")
def upload_dataset(
    file: UploadFile = UPLOAD_FILE,
) -> dict:
    """Upload a CSV dataset and return basic metadata."""

    dataframe = _read_csv_file(file)

    return {
        "filename": file.filename,
        "rows": len(dataframe),
        "columns": len(dataframe.columns),
        "column_names": dataframe.columns.tolist(),
    }


@router.post("/inspect")
def inspect_dataset(
    file: UploadFile = UPLOAD_FILE,
) -> dict:
    """Upload a CSV dataset and return its profile."""

    dataframe = _read_csv_file(file)

    profiler = DatasetProfiler(
        dataframe
    )

    return {
        "filename": file.filename,
        "profile": profiler.profile(),
    }


@router.post("/quality")
def analyze_dataset_quality(
    file: UploadFile = UPLOAD_FILE,
) -> dict:
    """Upload a CSV dataset and return data-quality analysis."""

    dataframe = _read_csv_file(file)

    analyzer = DataQualityAnalyzer(
        dataframe
    )

    return {
        "filename": file.filename,
        "quality": analyzer.analyze(),
    }


@router.post("/eda")
def analyze_dataset_eda(
    file: UploadFile = UPLOAD_FILE,
    target_column: str | None = None,
) -> dict:
    """Upload a CSV dataset and return automated EDA results."""

    dataframe = _read_csv_file(file)

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

    analyzer = EDAAnalyzer(
        dataframe,
        target_column=target_column,
    )

    eda_report = analyzer.analyze()

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


@router.post("/run")
def run_ml_pipeline(
    file: UploadFile = UPLOAD_FILE,
    target_column: str = TARGET_COLUMN,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> dict:
    """Run the automated ML pipeline on an uploaded CSV dataset."""

    dataframe = _read_csv_file(file)

    if not target_column.strip():
        raise HTTPException(
            status_code=400,
            detail="Target column is required.",
        )

    if target_column not in dataframe.columns:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Target column not found: "
                f"{target_column}"
            ),
        )

    if not 0 < test_size < 1:
        raise HTTPException(
            status_code=400,
            detail="test_size must be between 0 and 1.",
        )

    try:
        pipeline = MLPipeline(
            dataframe,
            target_column=target_column,
            test_size=test_size,
            random_state=random_state,
            cv=3,
            optimization_trials=2,
        )

        result = pipeline.run()

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"ML pipeline failed: {exc}",
        ) from exc

    return {
        "filename": file.filename,
        "run": _build_ml_run_response(result),
    }