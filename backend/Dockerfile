FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (e.g., for XGBoost)
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY backend/ backend/
COPY ml/ ml/

# Default command for the backend API
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
