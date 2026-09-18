# syntax=docker/dockerfile:1

# ---------- Stage 1: build a venv with all runtime dependencies ----------
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
COPY pyproject.toml ./
COPY src ./src

# Resolve/install dependencies declared in pyproject.toml, then drop the
# package itself from site-packages: the app is run straight from ./src
# (see credit_scoring/paths.py, which derives PROJECT_ROOT from file
# location and expects a src/ + models/ layout, not an installed package).
RUN pip install --upgrade pip \
    && pip install . \
    && pip uninstall -y credit-scoring

# ---------- Stage 2: minimal runtime image ----------
FROM python:3.12-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 appuser

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY src ./src
COPY models/lgbm_pipeline.joblib ./models/lgbm_pipeline.joblib

USER appuser

EXPOSE 8000

CMD ["uvicorn", "credit_scoring.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
