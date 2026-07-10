# ---------------------------------------------------------------------------
# Dockerfile - ICD-10 Multi-Agent Predictor (FastAPI on Cloud Run)
# Multi-stage build for a small, secure, production image.
# ---------------------------------------------------------------------------

# --- Stage 1: build wheels ---------------------------------------------------
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
COPY requirements.txt .

# Build wheels for all dependencies so the final image needs no compilers.
RUN pip wheel --wheel-dir=/wheels -r requirements.txt


# --- Stage 2: runtime --------------------------------------------------------
FROM python:3.11-slim AS runtime

# Prevent Python from writing .pyc files / buffering stdout (better logs).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    HF_HOME=/app/.cache/huggingface \
    PUBMEDBERT_MODEL_NAME=microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext

# Create a non-root user to run the application (security best practice).
RUN useradd --create-home --uid 1000 appuser
WORKDIR /app

# Install dependencies from pre-built wheels.
COPY --from=builder /wheels /wheels
COPY requirements.txt .
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt \
    && rm -rf /wheels

# Pre-download the PubMedBERT weights into the image (avoids cold-start pulls).
RUN python -c "from transformers import AutoTokenizer, AutoModel; \
m='${PUBMEDBERT_MODEL_NAME}'; AutoTokenizer.from_pretrained(m); AutoModel.from_pretrained(m)"

# Copy application source.
COPY . .

# Ensure the runtime user owns the app + model cache.
RUN chown -R appuser:appuser /app

# Drop privileges.
USER appuser

# Cloud Run injects $PORT (default 8080). Bind uvicorn to it.
EXPOSE 8080
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
