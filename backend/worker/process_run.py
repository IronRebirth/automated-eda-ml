import sys

from sqlalchemy.orm import Session

from backend.app.db.database import SessionLocal
from backend.app.db.models import Dataset, Run
from backend.app.services.datasets import load_dataset_record
from backend.app.services.runs import (
    update_run_result,
    update_run_status,
)
from ml.pipeline import MLPipeline


def build_ml_run_response(result: dict) -> dict:
    """Build the persisted ML result from a pipeline result."""

    best_model = result["best_model"]
    explainability = result["explainability"]

    return {
        "task_type": result["task_type"],
        "target_column": result["target_column"],
        "evaluation": result["evaluation"],
        "cross_validation": result["cross_validation"],
        "leaderboard": result["leaderboard"],
        "optimized_evaluation": result[
            "optimized_evaluation"
        ],
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


def mark_run_failed(
    db: Session,
    run: Run,
    message: str,
) -> None:
    """Mark a run as failed without crashing cleanup."""

    try:
        db.rollback()
        db.refresh(run)

        if run.status != "cancelled":
            update_run_status(
                db=db,
                run=run,
                status="failed",
            )

        print(
            message,
            flush=True,
        )

    except (
        OSError,
        RuntimeError,
        TypeError,
    ):
        db.rollback()


def process_run(
    db: Session,
    run: Run,
) -> None:
    """Execute one ML run."""

    dataset = db.get(
        Dataset,
        run.dataset_id,
    )

    if dataset is None:
        update_run_status(
            db=db,
            run=run,
            status="failed",
        )

        raise RuntimeError(
            f"Dataset not found: {run.dataset_id}"
        )

    if run.status == "cancelled":
        return

    try:
        dataframe = load_dataset_record(
            dataset
        )

        db.refresh(run)

        if run.status == "cancelled":
            return

        update_run_status(
            db=db,
            run=run,
            status="running",
        )

        pipeline = MLPipeline(
            dataframe,
            target_column=run.target_column,
            test_size=run.test_size,
            random_state=run.random_state,
            cv=3,
            optimization_trials=2,
        )

        result = pipeline.run()

        db.refresh(run)

        if run.status == "cancelled":
            print(
                f"ML run {run.id} was cancelled "
                "after pipeline execution.",
                flush=True,
            )
            return

        persisted_result = build_ml_run_response(
            result
        )

        update_run_result(
            db=db,
            run=run,
            result=persisted_result,
        )

        print(
            f"ML run {run.id} completed successfully.",
            flush=True,
        )

    except (
        ValueError,
        FileNotFoundError,
        RuntimeError,
        TypeError,
        OSError,
    ) as exc:
        mark_run_failed(
            db=db,
            run=run,
            message=(
                f"ML run {run.id} failed: {exc}"
            ),
        )

        raise


def main() -> None:
    """Process exactly one ML run by ID."""

    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python -m backend.worker.process_run <run_id>"
        )

    try:
        run_id = int(sys.argv[1])
    except ValueError as exc:
        raise SystemExit(
            "run_id must be an integer."
        ) from exc

    db: Session = SessionLocal()

    try:
        run = db.get(
            Run,
            run_id,
        )

        if run is None:
            raise SystemExit(
                f"Run not found: {run_id}"
            )

        if run.status == "cancelled":
            print(
                f"ML run {run_id} is cancelled.",
                flush=True,
            )
            return

        if run.status not in {
            "pending",
            "running",
        }:
            raise SystemExit(
                f"Run {run_id} has status "
                f"{run.status!r} and cannot be processed."
            )

        process_run(
            db=db,
            run=run,
        )

    finally:
        db.close()


if __name__ == "__main__":
    main()