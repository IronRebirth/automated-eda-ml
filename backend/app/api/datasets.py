import json
from io import BytesIO

import pandas as pd
from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from plotly.utils import PlotlyJSONEncoder
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.app.db.database import SessionLocal
from backend.app.db.models import Dataset
from backend.app.services.datasets import (
    create_dataset_record,
    load_dataset_record,
)
from backend.app.services.runs import (
    create_run_record,
    update_run_result,
    update_run_status,
)
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
COMPUTE_MODE = Form("local")


class DatasetRunRequest(BaseModel):
    """Request body for running ML on a persisted dataset."""

    target_column: str
    test_size: float = 0.2
    random_state: int = 42
    compute_mode: str = "local"


def _validate_compute_mode(
    compute_mode: str,
) -> str:
    """Validate and normalize the requested compute mode."""

    normalized_mode = compute_mode.strip().lower()

    if normalized_mode not in {
        "local",
        "cloud",
    }:
        raise HTTPException(
            status_code=400,
            detail=(
                "compute_mode must be either "
                "'local' or 'cloud'."
            ),
        )

    return normalized_mode


def _read_csv_file(
    file: UploadFile,
) -> pd.DataFrame:
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


def _run_persisted_dataset_pipeline(
    db: Session,
    dataset: Dataset,
    dataframe: pd.DataFrame,
    target_column: str,
    test_size: float,
    random_state: int,
    compute_mode: str = "local",
) -> tuple[int, dict]:
    """Run and persist an ML pipeline for an existing dataset."""

    run = create_run_record(
        db=db,
        dataset_id=dataset.id,
        target_column=target_column,
        test_size=test_size,
        random_state=random_state,
        compute_mode=compute_mode,
    )

    try:
        update_run_status(
            db=db,
            run=run,
            status="running",
        )

        pipeline = MLPipeline(
            dataframe,
            target_column=target_column,
            test_size=test_size,
            random_state=random_state,
            cv=3,
            optimization_trials=2,
        )

        result = pipeline.run()

        persisted_result = _build_ml_run_response(
            result
        )

        update_run_result(
            db=db,
            run=run,
            result=persisted_result,
        )

        return run.id, persisted_result

    except ValueError as exc:
        update_run_status(
            db=db,
            run=run,
            status="failed",
        )

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        update_run_status(
            db=db,
            run=run,
            status="failed",
        )

        raise HTTPException(
            status_code=500,
            detail=f"ML pipeline failed: {exc}",
        ) from exc


@router.get("/{dataset_id}")
def get_dataset(
    dataset_id: int,
) -> dict:
    """Retrieve a persisted dataset and its profile by ID."""

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

        profile = DatasetProfiler(
            dataframe
        ).profile()

        return {
            "dataset_id": dataset.id,
            "filename": dataset.filename,
            "rows": dataset.rows,
            "columns": dataset.columns,
            "created_at": dataset.created_at,
            "profile": profile,
        }

    finally:
        db.close()


@router.post("/upload")
def upload_dataset(
    file: UploadFile = UPLOAD_FILE,
) -> dict:
    """Upload a CSV dataset and persist its metadata and data."""

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
    compute_mode: str = COMPUTE_MODE,
) -> dict:
    """Upload a CSV dataset and run the automated ML pipeline."""

    dataframe = _read_csv_file(file)

    normalized_target = target_column.strip()
    normalized_compute_mode = _validate_compute_mode(
        compute_mode
    )

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

    db: Session = SessionLocal()
    run = None

    try:
        dataset = create_dataset_record(
            db=db,
            filename=file.filename,
            dataframe=dataframe,
        )

        run = create_run_record(
            db=db,
            dataset_id=dataset.id,
            target_column=normalized_target,
            test_size=test_size,
            random_state=random_state,
            compute_mode=normalized_compute_mode,
        )

        pipeline = MLPipeline(
            dataframe,
            target_column=normalized_target,
            test_size=test_size,
            random_state=random_state,
            cv=3,
            optimization_trials=2,
        )

        result = pipeline.run()

        persisted_result = _build_ml_run_response(
            result
        )

        update_run_result(
            db=db,
            run=run,
            result=persisted_result,
        )

        return {
            "filename": file.filename,
            "run_id": run.id,
            "compute_mode": run.compute_mode,
            "run": persisted_result,
        }

    except ValueError as exc:
        if run is not None:
            update_run_status(
                db=db,
                run=run,
                status="failed",
            )

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        if run is not None:
            update_run_status(
                db=db,
                run=run,
                status="failed",
            )

        raise HTTPException(
            status_code=500,
            detail=f"ML pipeline failed: {exc}",
        ) from exc

    finally:
        db.close()


@router.post("/{dataset_id}/run")
def run_persisted_dataset(
    dataset_id: int,
    request: DatasetRunRequest,
) -> dict:
    """Queue an ML pipeline for a persisted dataset."""

    normalized_target = request.target_column.strip()
    normalized_compute_mode = _validate_compute_mode(
        request.compute_mode
    )

    if not normalized_target:
        raise HTTPException(
            status_code=400,
            detail="Target column is required.",
        )

    if not 0 < request.test_size < 1:
        raise HTTPException(
            status_code=400,
            detail="test_size must be between 0 and 1.",
        )

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

        if normalized_target not in dataframe.columns:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Target column not found: "
                    f"{normalized_target}"
                ),
            )

        run = create_run_record(
            db=db,
            dataset_id=dataset.id,
            target_column=normalized_target,
            test_size=request.test_size,
            random_state=request.random_state,
            compute_mode=normalized_compute_mode,
        )

        return {
            "dataset_id": dataset.id,
            "run_id": run.id,
            "status": "queued",
            "compute_mode": run.compute_mode,
        }

    finally:
        db.close()


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