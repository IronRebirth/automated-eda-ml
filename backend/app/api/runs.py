from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.db.database import SessionLocal
from backend.app.db.models import Run

router = APIRouter(
    prefix="/runs",
    tags=["runs"],
)


@router.get("")
def list_runs(
    dataset_id: int | None = None,
    limit: int = Query(
        default=20,
        ge=1,
        le=100,
    ),
    offset: int = Query(
        default=0,
        ge=0,
    ),
) -> dict:
    """List persisted analysis runs with optional dataset filtering."""

    db: Session = SessionLocal()

    try:
        statement = select(Run)

        if dataset_id is not None:
            statement = statement.where(
                Run.dataset_id == dataset_id
            )

        statement = (
            statement
            .order_by(Run.created_at.desc())
            .offset(offset)
            .limit(limit)
        )

        runs = db.scalars(statement).all()

        return {
            "runs": [
                {
                    "run_id": run.id,
                    "dataset_id": run.dataset_id,
                    "status": run.status,
                    "target_column": run.target_column,
                    "test_size": run.test_size,
                    "random_state": run.random_state,
                    "created_at": run.created_at,
                    "completed_at": run.completed_at,
                }
                for run in runs
            ],
            "limit": limit,
            "offset": offset,
            "count": len(runs),
        }

    finally:
        db.close()


@router.get("/{run_id}")
def get_run(
    run_id: int,
) -> dict:
    """Retrieve a persisted analysis run by ID."""

    db: Session = SessionLocal()

    try:
        run = db.get(
            Run,
            run_id,
        )

        if run is None:
            raise HTTPException(
                status_code=404,
                detail=f"Run not found: {run_id}",
            )

        return {
            "run_id": run.id,
            "dataset_id": run.dataset_id,
            "status": run.status,
            "target_column": run.target_column,
            "test_size": run.test_size,
            "random_state": run.random_state,
            "created_at": run.created_at,
            "completed_at": run.completed_at,
        }

    finally:
        db.close()