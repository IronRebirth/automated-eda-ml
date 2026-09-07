from contextlib import asynccontextmanager

from fastapi import FastAPI

import backend.app.db.models  # noqa: F401
from backend.app.api.datasets import router as datasets_router
from backend.app.db.database import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize application resources during startup."""

    init_db()

    yield


app = FastAPI(
    title="Automated EDA & ML API",
    version="1.0.0",
    lifespan=lifespan,
)


app.include_router(datasets_router)


@app.get("/health")
def health_check() -> dict[str, str]:
    """Return the API health status."""

    return {
        "status": "ok",
    }