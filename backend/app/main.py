from fastapi import FastAPI

from backend.app.api.datasets import router as datasets_router

app = FastAPI(
    title="Automated EDA & ML API",
    version="1.0.0",
)


app.include_router(datasets_router)


@app.get("/health")
def health_check() -> dict[str, str]:
    """Return the API health status."""

    return {
        "status": "ok",
    }