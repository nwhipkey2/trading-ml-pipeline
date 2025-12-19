FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps (pyarrow + pandas sometimes need these)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first (better caching)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Optional ML deps that you already use
RUN pip install --no-cache-dir scikit-learn joblib pyyaml

# Copy project
COPY . /app

# Default command (override in compose)
CMD ["python", "-m", "src.run_pipeline"]
