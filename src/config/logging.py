"""Structured logging setup.

Call :func:`setup_logging` once at application startup. After that, modules use::

    import structlog
    log = structlog.get_logger(__name__)
    log.info("event.name", key=value)

Every log entry is JSON in non-development environments and carries:

* ``timestamp`` (ISO 8601, UTC)
* ``level``
* ``logger`` (module name)
* ``trace_id`` (from the active OpenTelemetry span, or ``None`` when no span is active)
* ``environment``
* any context bound via :func:`structlog.contextvars.bind_contextvars`
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

from src.config.settings import settings


def _add_trace_id(_logger: Any, _method: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    """Attach the current OpenTelemetry trace ID, if a span is active.

    Imports ``opentelemetry`` lazily so the logger still works before the tracer
    is wired in Phase 4 (and so the import is optional during Phase 1 builds).
    """
    try:
        from opentelemetry.trace import get_current_span

        span = get_current_span()
        ctx = span.get_span_context() if span else None
        if ctx and ctx.is_valid:
            event_dict["trace_id"] = f"{ctx.trace_id:032x}"
            event_dict["span_id"] = f"{ctx.span_id:016x}"
        else:
            event_dict["trace_id"] = None
    except ImportError:
        event_dict["trace_id"] = None
    return event_dict


def _add_environment(_logger: Any, _method: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    event_dict["environment"] = settings.environment
    return event_dict


def setup_logging() -> None:
    """Configure structlog and stdlib ``logging`` for the running process."""
    is_dev = settings.environment == "development"

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        _add_environment,
        _add_trace_id,
    ]

    renderer: Any = (
        structlog.dev.ConsoleRenderer(colors=True)
        if is_dev
        else structlog.processors.JSONRenderer()
    )

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(settings.log_level.upper())
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )

    # Route stdlib loggers (uvicorn, sqlalchemy, etc.) through the same renderer
    # so everything in the process emits one consistent log format.
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=shared_processors,
            processor=renderer,
        )
    )
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(settings.log_level.upper())

    # Uvicorn ships its own loggers — let them propagate to root instead of
    # double-printing.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.handlers = []
        lg.propagate = True
