"""FastAPI application factory.

Phase 1 wires only: settings, structured logging, request-id + timing middleware,
and a ``/health`` endpoint. Pipeline routers (``/query``, ``/traces``, ``/defects``,
``/evals``, ``/meta/query``) land in their owning phases.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI

from src.api.error_handlers import install_error_handlers
from src.api.middleware import BodySizeLimitMiddleware, RequestIDMiddleware, TimingMiddleware
from src.api.rate_limit import install_rate_limiting
from src.api.routers import defects, documents, evals, health, meta, query, traces
from src.config.logging import setup_logging
from src.config.settings import settings
from src.ingestion.indexer import init_qdrant
from src.observability.tracer import setup_tracing, shutdown_tracing
from src.pipeline.pipeline_runner import pipeline_runner
from src.storage.database import init_engine, shutdown_engine


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    setup_logging()
    setup_tracing()
    log = structlog.get_logger(__name__)
    log.info(
        "app.startup",
        app_name=settings.app_name,
        environment=settings.environment,
        llm_provider=settings.llm.provider,
        llm_model=settings.llm.model,
    )
    init_engine()  # lazy — does not connect until first query
    try:
        await init_qdrant()
    except Exception as exc:  # noqa: BLE001 — don't crash the API if Qdrant is briefly unavailable
        log.warning("app.startup.qdrant_init_failed", error=str(exc), error_type=type(exc).__name__)
    await pipeline_runner.startup()
    yield
    await pipeline_runner.shutdown()
    await shutdown_engine()
    log.info("app.shutdown")
    shutdown_tracing()


app = FastAPI(
    title="RAG Observability API",
    version="0.1.0",
    lifespan=lifespan,
    docs_url=None if settings.auth.protect_api_docs else "/docs",
    redoc_url=None if settings.auth.protect_api_docs else "/redoc",
    openapi_url=None if settings.auth.protect_api_docs else "/openapi.json",
)

# Middleware order matters: TimingMiddleware wraps RequestIDMiddleware so the
# access log it emits already sees the bound request_id. BodySizeLimitMiddleware
# runs first (outermost) so over-sized bodies are rejected before any expensive
# parsing.
app.add_middleware(TimingMiddleware)
app.add_middleware(RequestIDMiddleware)
app.add_middleware(BodySizeLimitMiddleware, max_bytes=settings.resources.max_body_bytes)

# Rate limiting + structured error responses. install_rate_limiting also adds
# the SlowAPIMiddleware; install_error_handlers wires our centralized JSON
# error envelopes.
install_rate_limiting(app)
install_error_handlers(app)

app.include_router(health.router)
app.include_router(documents.router)
app.include_router(query.router)
app.include_router(traces.router)
app.include_router(defects.router)
app.include_router(evals.router)
app.include_router(meta.router)
