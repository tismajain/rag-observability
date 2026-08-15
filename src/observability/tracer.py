"""OpenTelemetry tracer setup + accessor.

Call :func:`setup_tracing` once at process startup (the FastAPI lifespan does
this). Call :func:`shutdown_tracing` at process shutdown so the
:class:`BatchSpanProcessor` flushes any spans still in its in-memory queue.

After setup, every existing ``with tracer.start_as_current_span(...)`` call
in the codebase (ingestion, retrieval, etc.) starts emitting real spans to
the configured exporters. No code changes needed in those modules — the
forward-design pattern from earlier phases pays off here.
"""

from __future__ import annotations

import structlog
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)
from opentelemetry.trace import Tracer

from src.config.settings import settings

log = structlog.get_logger(__name__)

_DEFAULT_TRACER_NAME = "rag_observability"
_provider: TracerProvider | None = None


def setup_tracing() -> None:
    """Install a global :class:`TracerProvider` with Phoenix + optional console exporters.

    Idempotent: a second call is a no-op. This lets the FastAPI lifespan and
    standalone scripts both call it without coordination.
    """
    global _provider
    if _provider is not None:
        return

    resource = Resource.create(
        {
            "service.name": settings.app_name,
            "service.version": "0.1.0",
            "deployment.environment": settings.environment,
        }
    )
    provider = TracerProvider(resource=resource)

    # Primary exporter: OTLP gRPC to Phoenix. Batched so the user request path
    # never blocks on a flush.
    otlp_exporter = OTLPSpanExporter(
        endpoint=settings.observability.phoenix_endpoint,
        insecure=True,
    )
    provider.add_span_processor(BatchSpanProcessor(otlp_exporter))

    # Optional console exporter for local debugging. SimpleSpanProcessor flushes
    # immediately so spans appear in stdout as they close.
    if settings.observability.enable_console_exporter:
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

    trace.set_tracer_provider(provider)
    _provider = provider

    log.info(
        "observability.tracing.configured",
        endpoint=settings.observability.phoenix_endpoint,
        console_exporter=settings.observability.enable_console_exporter,
        service_name=settings.app_name,
        environment=settings.environment,
    )


def shutdown_tracing() -> None:
    """Flush and tear down the tracer provider. Idempotent."""
    global _provider
    if _provider is None:
        return
    try:
        _provider.shutdown()
    except Exception as exc:  # noqa: BLE001 — never let shutdown crash the app
        log.warning("observability.tracing.shutdown_failed", error=str(exc))
    finally:
        _provider = None


def get_tracer(name: str = _DEFAULT_TRACER_NAME) -> Tracer:
    """Return a named tracer. Safe to call before :func:`setup_tracing`.

    Without a configured provider, OTel returns a no-op tracer whose spans
    cost a few nanoseconds and emit nothing. That is how earlier phases
    instrumented themselves without depending on Phase 4 being live.
    """
    return trace.get_tracer(name)
