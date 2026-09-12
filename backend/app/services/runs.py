from datetime import UTC, datetime
from typing import Any

from sqlalchemy.orm import Session

from backend.app.db.models import Run


def create_run_record(
    db: Session,
    dataset_id: int,
    target_column: str | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    compute_mode: str = "local",
) -> Run:
    """Create and persist a pending analysis run."""

    if compute_mode not in {
        "local",
        "cloud",
    }:
        raise ValueError(
            "compute_mode must be 'local' or 'cloud'."
        )

    run = Run(
        dataset_id=dataset_id,
        status="pending",
        compute_mode=compute_mode,
        current_stage="queued",
        progress=0,
        stage_message="Waiting for the ML worker.",
        target_column=target_column,
        test_size=test_size,
        random_state=random_state,
    )

    db.add(run)
    db.commit()
    db.refresh(run)

    return run


def update_run_status(
    db: Session,
    run: Run,
    status: str,
) -> Run:
    """Update the lifecycle status of an analysis run."""

    run.status = status

    if status == "running" and run.started_at is None:
        run.started_at = datetime.now(UTC)

    if status in {
        "completed",
        "failed",
    }:
        run.completed_at = datetime.now(UTC)

    db.commit()
    db.refresh(run)

    return run


def update_run_progress(
    db: Session,
    run: Run,
    stage: str,
    progress: int,
    message: str,
) -> Run:
    """Persist the current execution stage and progress."""

    if not 0 <= progress <= 100:
        raise ValueError(
            "progress must be between 0 and 100."
        )

    run.current_stage = stage
    run.progress = progress
    run.stage_message = message

    db.commit()
    db.refresh(run)

    return run


def update_run_result(
    db: Session,
    run: Run,
    result: dict[str, Any],
) -> Run:
    """Persist the JSON-safe result of an analysis run."""

    run.result = result
    run.status = "completed"
    run.current_stage = "completed"
    run.progress = 100
    run.stage_message = "Analysis completed successfully."

    if run.started_at is None:
        run.started_at = datetime.now(UTC)

    run.completed_at = datetime.now(UTC)

    db.commit()
    db.refresh(run)

    return run