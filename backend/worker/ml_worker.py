import time

from sqlalchemy.orm import Session

from backend.app.db.database import SessionLocal
from backend.app.db.models import Dataset, Run
from backend.app.services.datasets import load_dataset_record
from backend.app.services.runs import (
    update_run_progress,
    update_run_result,
    update_run_status,
)
from ml.pipeline import MLPipeline

POLL_INTERVAL_SECONDS = 2


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
        "optimized_evaluation": result["optimized_evaluation"],
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
    """Mark a run as failed without crashing the worker."""

    try:
        db.rollback()
        db.refresh(run)

        if run.status != "cancelled":
            run.current_stage = "failed"
            run.progress = min(
                run.progress,
                99,
            )
            run.stage_message = message

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


def recover_stale_runs(
    db: Session,
) -> None:
    """Mark interrupted running jobs as failed on worker startup."""

    stale_runs = (
        db.query(Run)
        .filter(
            Run.status == "running"
        )
        .all()
    )

    if not stale_runs:
        return

    for run in stale_runs:
        run.current_stage = "failed"
        run.stage_message = (
            "Worker stopped before the analysis completed."
        )

        update_run_status(
            db=db,
            run=run,
            status="failed",
        )

        print(
            f"Recovered stale ML run {run.id} "
            "as failed.",
            flush=True,
        )


def process_run(
    db: Session,
    run: Run,
) -> None:
    """Execute one pending ML run."""

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
        return

    if run.status == "cancelled":
        return

    try:
        update_run_progress(
            db=db,
            run=run,
            stage="loading",
            progress=2,
            message="Loading dataset.",
        )

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

        def report_progress(
            stage: str,
            progress: int,
            message: str,
        ) -> None:
            db.refresh(run)

            if run.status == "cancelled":
                return

            update_run_progress(
                db=db,
                run=run,
                stage=stage,
                progress=progress,
                message=message,
            )

            print(
                f"[ML RUN {run.id}] "
                f"{stage} ({progress}%): {message}",
                flush=True,
            )

        report_progress(
            "profiling",
            8,
            "Dataset loaded. Inspecting the prediction task.",
        )

        pipeline = MLPipeline(
            dataframe,
            target_column=run.target_column,
            test_size=run.test_size,
            random_state=run.random_state,
            cv=3,
            optimization_trials=2,
            progress_callback=report_progress,
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
    ) as exc:
        mark_run_failed(
            db=db,
            run=run,
            message=f"ML run {run.id} failed: {exc}",
        )

    except OSError as exc:
        mark_run_failed(
            db=db,
            run=run,
            message=f"ML run {run.id} failed: {exc}",
        )


def worker_loop() -> None:
    """Continuously process pending ML runs."""

    print(
        "ML worker started. "
        "Recovering interrupted runs...",
        flush=True,
    )

    db: Session = SessionLocal()

    try:
        recover_stale_runs(db)

    except (
        OSError,
        RuntimeError,
        TypeError,
    ) as exc:
        db.rollback()

        print(
            f"ML worker recovery error: {exc}",
            flush=True,
        )

    finally:
        db.close()

    print(
        "ML worker ready. "
        "Waiting for pending runs...",
        flush=True,
    )

    while True:
        db: Session = SessionLocal()

        try:
            run = (
                db.query(Run)
                .filter(
                    Run.status == "pending",
                    Run.compute_mode == "local",
                )
                .order_by(
                    Run.created_at.asc()
                )
                .first()
            )

            if run is not None:
                print(
                    f"Processing ML run {run.id}...",
                    flush=True,
                )

                process_run(
                    db=db,
                    run=run,
                )

        except (
            OSError,
            RuntimeError,
            TypeError,
        ) as exc:
            db.rollback()

            print(
                f"ML worker error: {exc}",
                flush=True,
            )

        finally:
            db.close()

        time.sleep(
            POLL_INTERVAL_SECONDS
        )


if __name__ == "__main__":
    worker_loop()