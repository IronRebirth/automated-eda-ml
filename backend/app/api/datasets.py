import json
from io import BytesIO

import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from plotly.utils import PlotlyJSONEncoder
from sqlalchemy.orm import Session

from backend.app.db.database import SessionLocal
from backend.app.services.datasets import create_dataset_record
from ml.eda import EDAAnalyzer
from ml.models import predict_from_artifact
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
ARTIFACT_PATH = Form("")


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


def _build_unified_analysis_response(
    dataframe: pd.DataFrame,
    target_column: str | None,
    test_size: float,
    random_state: int,
) -> dict:
    """Run profiling, quality, EDA, and optional ML analysis."""

    profile = DatasetProfiler(
        dataframe
    ).profile()

    quality = DataQualityAnalyzer(
        dataframe
    ).analyze()

    eda_analyzer = EDAAnalyzer(
        dataframe,
        target_column=target_column,
    )

    eda = eda_analyzer.analyze()

    eda["visualizations"] = (
        _serialize_eda_visualizations(
            eda["visualizations"]
        )
    )

    ml_result = None

    if target_column is not None:
        pipeline = MLPipeline(
            dataframe,
            target_column=target_column,
            test_size=test_size,
            random_state=random_state,
            cv=3,
            optimization_trials=2,
        )

        result = pipeline.run()

        ml_result = _build_ml_run_response(
            result
        )

    return {
        "profile": profile,
        "quality": quality,
        "eda": eda,
        "ml": ml_result,
    }


@router.post("/upload")
def upload_dataset(
    file: UploadFile = UPLOAD_FILE,
) -> dict:
    """Upload a CSV dataset and persist its metadata."""

    dataframe = _read_csv_file(file)

    db: Session = SessionLocal()

    try:
        dataset = create_dataset_record(
            db=db,
            filename=file.filename,
            dataframe=dataframe,
        )

    except Exception as exc:
        db.rollback()

        raise HTTPException(
            status_code=500,
            detail=f"Failed to persist dataset: {exc}",
        ) from exc

    finally:
        db.close()

    return {
        "dataset_id": dataset.id,
        "filename": dataset.filename,
        "rows": dataset.rows,
        "columns": dataset.columns,
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

    normalized_target = (
        target_column.strip()
        if target_column
        else None
    )

    if normalized_target and normalized_target not in dataframe.columns:
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
        "filename": file.filename,
        "target_column": normalized_target,
        "eda": eda,
    }


@router.post("/run")
def run_ml_pipeline(
    file: UploadFile = UPLOAD_FILE,
    target_column: str = TARGET_COLUMN,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> dict:
    """Upload a CSV dataset and run the automated ML pipeline."""

    dataframe = _read_csv_file(file)

    normalized_target = target_column.strip()

    if not normalized_target:
        raise HTTPException(
            status_code=400,
            detail="Target column is required.",
        )

    if normalized_target not in dataframe.columns:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Target column not found: "
                f"{normalized_target}"
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
            target_column=normalized_target,
            test_size=test_size,
            random_state=random_state,
            cv=3,
            optimization_trials=2,
        )

        result = pipeline.run()

        return {
            "filename": file.filename,
            "run": _build_ml_run_response(
                result
            ),
        }

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


@router.post("/predict")
def predict_dataset(
    file: UploadFile = UPLOAD_FILE,
    artifact_path: str = ARTIFACT_PATH,
) -> dict:
    """Generate predictions using a saved model artifact."""

    dataframe = _read_csv_file(file)

    normalized_artifact_path = artifact_path.strip()

    if not normalized_artifact_path:
        raise HTTPException(
            status_code=400,
            detail="Artifact path is required.",
        )

    try:
        predictions = predict_from_artifact(
            normalized_artifact_path,
            dataframe,
        )

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}",
        ) from exc

    return {
        "filename": file.filename,
        "predictions": predictions.tolist(),
    }


@router.post("/analyze")
def analyze_dataset(
    file: UploadFile = UPLOAD_FILE,
    target_column: str = TARGET_COLUMN,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> dict:
    """Run unified dataset profiling, quality, EDA, and ML analysis."""

    dataframe = _read_csv_file(file)

    normalized_target = target_column.strip()

    if normalized_target and normalized_target not in dataframe.columns:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Target column not found: "
                f"{normalized_target}"
            ),
        )

    if not 0 < test_size < 1:
        raise HTTPException(
            status_code=400,
            detail="test_size must be between 0 and 1.",
        )

    try:
        analysis = _build_unified_analysis_response(
            dataframe=dataframe,
            target_column=(
                normalized_target
                if normalized_target
                else None
            ),
            test_size=test_size,
            random_state=random_state,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unified analysis failed: {exc}",
        ) from exc

    return {
        "filename": file.filename,
        "target_column": (
            normalized_target
            if normalized_target
            else None
        ),
        "analysis": analysis,
    }