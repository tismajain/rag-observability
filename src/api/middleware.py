"""HTTP middleware: request-id injection, timing/access logging, body-size cap."""

from __future__ import annotations

import os
import time
from uuid import uuid4

import structlog
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

log = structlog.get_logger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"

# Default 1 MiB — Pydantic enforces `query` max_length=2000 already; this is
# a defense-in-depth cap to reject obviously-malicious payloads before any
# JSON parsing happens.
DEFAULT_MAX_BODY_BYTES = int(os.environ.get("MAX_BODY_BYTES", str(1 * 1024 * 1024)))


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Generate or accept a request ID and bind it to the structlog contextvars.

    Downstream log calls automatically include ``request_id`` without each
    handler having to pass it explicitly.
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        supplied = request.headers.get(REQUEST_ID_HEADER, "")
        request_id = (
            supplied
            if supplied.isascii() and supplied.isprintable() and len(supplied) <= 128
            else str(uuid4())
        )
        request.state.request_id = request_id
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)
        try:
            response = await call_next(request)
        finally:
            # Clearing here keeps the contextvars from leaking between requests
            # in the same worker process.
            pass
        response.headers[REQUEST_ID_HEADER] = request_id
        return response


class TimingMiddleware(BaseHTTPMiddleware):
    """Emit a structured access log for every HTTP request."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        start = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            duration_ms = int((time.perf_counter() - start) * 1000)
            log.info(
                "http.request",
                method=request.method,
                path=request.url.path,
                status_code=status_code,
                duration_ms=duration_ms,
            )


class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject requests whose body exceeds :data:`DEFAULT_MAX_BODY_BYTES`.

    Reads ``Content-Length`` when the client supplies it (fast path); falls
    back to consuming the stream for chunked uploads. Returns a structured
    413 envelope matching the rest of the API's error shape.
    """

    def __init__(self, app: object, max_bytes: int = DEFAULT_MAX_BODY_BYTES) -> None:
        super().__init__(app)  # type: ignore[arg-type]
        self._max = max_bytes

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                declared = int(content_length)
            except ValueError:
                declared = 0
            if declared > self._max:
                return self._too_large(declared)
        return await call_next(request)

    def _too_large(self, size: int) -> JSONResponse:
        return JSONResponse(
            status_code=413,
            content={
                "error": "payload_too_large",
                "detail": f"Request body of {size} bytes exceeds limit of {self._max} bytes",
                "request_id": None,
            },
        )
