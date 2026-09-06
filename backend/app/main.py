from fastapi import FastAPI

app = FastAPI(
    title="Automated EDA & ML API",
    version="1.0.0",
)


@app.get("/health")
def health_check() -> dict[str, str]:
    """Return the API health status."""

    return {
        "status": "ok",
    }