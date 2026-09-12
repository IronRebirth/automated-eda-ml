# automated-eda-ml

Automated Exploratory Data Analysis and Machine Learning platform. Upload a CSV, get a full EDA report, and trigger an end-to-end ML pipeline — all without writing a line of code.

## Features

- **Dataset upload** — CSV ingestion with automatic schema detection and profiling
- **Exploratory Data Analysis** — distributions, correlations, missing value analysis, outlier detection
- **Data quality scoring** — cardinality checks, duplicate detection, completeness metrics
- **Automated ML pipeline** — task type detection, model selection, hyperparameter optimisation, cross-validation
- **SHAP explainability** — feature importance, structured explainability output, human-readable insights
- **Run tracking** — persistent run history with status, metrics, and artefact storage

## Architecture

```
backend/     FastAPI + PostgreSQL (Python 3.12)
frontend/    Next.js 15, TypeScript
ml/          ML engine — EDA, quality, pipeline, explainability, reporting
worker/      Dedicated ML worker process
datasets/    Sample datasets
```

## Quick start

### Prerequisites

- Docker + Docker Compose
- Python 3.12 (for local backend dev)
- Node.js 20+ (for local frontend dev)

### Docker (recommended)

```bash
docker compose up --build
```

| Service  | URL |
|---|---|
| Frontend | http://localhost:3000 |
| Backend  | http://localhost:8000 |
| Docs     | http://localhost:8000/docs |

### Local development

**Backend**

```bash
pip install -r requirements.txt
uvicorn backend.app.main:app --reload
```

**Frontend**

```bash
cd frontend
npm install
npm run dev
```

## Testing

```bash
pytest
```

179 tests across the ML engine, API layer, and database layer.

## License

MIT