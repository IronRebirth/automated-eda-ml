from fastapi import APIRouter, HTTPException
from sqlalchemy.orm import Session

from backend.app.db.database import SessionLocal
from backend.app.db.models import Run

router = APIRouter(
    prefix="/runs",
    tags=["runs"],
)


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