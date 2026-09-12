from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import backend.app.db.models  # noqa: F401
from backend.app.api.datasets import router as datasets_router
from backend.app.api.eda import router as eda_router
from backend.app.api.runs import router as runs_router
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


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(datasets_router)
app.include_router(eda_router)
app.include_router(runs_router)


@app.get("/health")
def health_check() -> dict[str, str]:
    """Return the API health status."""

    return {
        "status": "ok",
    }