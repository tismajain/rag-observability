# API service image.
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System deps kept minimal — add libs in the phase that needs them.
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

# Install Python deps as a separate, cached layer.
COPY pyproject.toml ./
COPY src ./src
COPY cli ./cli
RUN pip install -e ".[dev,eval]"

# Drop to non-root.
RUN useradd --create-home --shell /usr/sbin/nologin appuser \
 && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=6 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health/live').status==200 else 1)"

# --loop asyncio (instead of the default uvloop) lets RAGAS's nest_asyncio
# patching work — nest_asyncio refuses to patch uvloop. Small perf cost,
# unblocks in-process RAGAS evals.
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--loop", "asyncio"]
