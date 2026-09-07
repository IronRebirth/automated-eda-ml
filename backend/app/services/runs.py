from datetime import UTC, datetime

from sqlalchemy.orm import Session

from backend.app.db.models import Run


def create_run_record(
    db: Session,
    dataset_id: int,
    target_column: str | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Run:
    """Create and persist a pending analysis run."""

    run = Run(
        dataset_id=dataset_id,
        status="pending",
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

    if status in {"completed", "failed"}:
        run.completed_at = datetime.now(UTC)

    db.commit()
    db.refresh(run)

    return run