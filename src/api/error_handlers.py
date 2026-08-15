"""Centralized FastAPI error handlers.

Wraps four exception classes into the structured :class:`ErrorResponse`
envelope so every error response looks the same to a client:

* :class:`HTTPException` — explicit ``raise HTTPException(...)`` in handlers
* :class:`RequestValidationError` — Pydantic-driven 422 on bad request bodies
* :class:`StarletteHTTPException` — Starlette-level 404s and similar
* :class:`Exception` — final catch-all so internal stack traces never reach
  the wire. Instead the client gets a structured 500 with the request ID.

Every response carries ``request_id`` so an operator can grep logs + Phoenix
for the failing trace without asking the user for a screenshot.
"""

from __future__ import annotations

import structlog
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.security.redaction import redact, redact_text

log = structlog.get_logger(__name__)


def _request_id(request: Request) -> str | None:
    return request.headers.get("X-Request-ID") or getattr(request.state, "request_id", None)


def _envelope(*, error: str, detail: object, status_code: int, request: Request) -> JSONResponse:
    body = {
        "error": error,
        "detail": detail,
        "request_id": _request_id(request),
    }
    return JSONResponse(status_code=status_code, content=body)


async def _http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """Wrap explicit HTTPExceptions into the structured envelope."""
    # Common error codes get a short stable name; everything else uses
    # ``http_error`` so clients can pattern-match without parsing English.
    error_name = {
        400: "bad_request",
        401: "unauthorized",
        403: "forbidden",
        404: "not_found",
        409: "conflict",
        422: "validation_error",
        429: "rate_limit_exceeded",
        500: "internal_error",
        503: "service_unavailable",
    }.get(exc.status_code, "http_error")
    detail = redact(exc.detail if isinstance(exc.detail, (str, list)) else str(exc.detail))
    return _envelope(
        error=error_name,
        detail=detail,
        status_code=exc.status_code,
        request=request,
    )


async def _validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Replace FastAPI's default 422 body with the structured envelope."""
    # exc.errors() is a list of dicts already JSON-serializable.
    return _envelope(
        error="validation_error",
        detail=redact(exc.errors()),
        status_code=422,
        request=request,
    )


async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all so internal failures never leak stack traces to clients.

    The exception is logged with the full traceback for operators; the wire
    response shows only the exception class name (``RuntimeError`` /
    ``TimeoutError``) so a client can tell *what kind* of thing went wrong
    without exposing internals.
    """
    log.exception(
        "api.unhandled_exception",
        path=request.url.path,
        method=request.method,
        error_type=type(exc).__name__,
        error=redact_text(exc),
        request_id=_request_id(request),
    )
    return _envelope(
        error="internal_error",
        detail=f"Unexpected {type(exc).__name__}",
        status_code=500,
        request=request,
    )


def install_error_handlers(app: FastAPI) -> None:
    """Register the four exception handlers on the given app."""
    # Starlette's add_exception_handler typing is loose; the handlers are
    # narrower-typed for clarity, so cast at the registration boundary.
    app.add_exception_handler(StarletteHTTPException, _http_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(RequestValidationError, _validation_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, _unhandled_exception_handler)
