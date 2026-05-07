# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# Stage 1 — builder: install Python deps via uv (with [real] extras)
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

WORKDIR /build

# Install uv
RUN pip install --no-cache-dir uv==0.6.14

# Copy dependency manifests first (layer caching)
COPY pyproject.toml uv.lock ./
COPY src/ src/

# Sync production deps + real extras (FastAPI, uvicorn, asyncpg, etc.)
RUN uv sync --frozen --no-dev --extra real

# ---------------------------------------------------------------------------
# Stage 2 — runtime: lean image with only what's needed to run
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

# Create non-root user (uid 1001)
RUN addgroup --gid 1001 appgroup \
    && adduser --uid 1001 --gid 1001 --no-create-home --disabled-password --gecos "" appuser

WORKDIR /app

# Copy the installed virtual environment from builder
COPY --from=builder /build/.venv /app/.venv

# Copy application source
COPY --from=builder /build/src /app/src

# Copy the browser frontend static assets so the FastAPI backend can serve them
COPY src/voice_eval_lab/backend/static /app/src/voice_eval_lab/backend/static

# Put the venv on PATH so the `voice-eval` entrypoint resolves
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Drop to non-root
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz')"

CMD ["uv", "run", "voice-eval", "serve", "--host", "0.0.0.0", "--port", "8000"]
