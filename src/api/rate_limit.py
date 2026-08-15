"""Rate limiting for LLM-spending endpoints.

slowapi-based limiter. We rate-limit by client IP at the gateway layer to
protect the API-key budget from a runaway client. The limits are intentionally
generous — production-volume traffic shapes can override them via env vars.

Two pieces:

* :data:`limiter` — the module-level :class:`Limiter` instance. Routers use it
  via the ``@limiter.limit("...")`` decorator on individual handlers.
* :func:`install_rate_limiting` — wires the limiter and its 429 handler onto
  a FastAPI app. Called from the app factory.

Disabled-by-default tests: set ``RATE_LIMIT_ENABLED=false`` in the environment
and ``limiter.enabled`` flips off, so the existing FastAPI TestClient tests
keep passing without a rewrite.
"""

from __future__ import annotations

import os

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from src.config.settings import settings

# Default policies. Keep humans in mind: an analyst poking the dashboard
# shouldn't trip the limit, but a runaway script should.
DEFAULT_QUERY_RATE = os.environ.get(
    "RATE_LIMIT_QUERY", f"{settings.rate_limit.query_per_minute}/minute"
)
DEFAULT_META_QUERY_RATE = os.environ.get(
    "RATE_LIMIT_META_QUERY", f"{settings.rate_limit.meta_query_per_minute}/minute"
)


def _principal_or_ip(request: Request) -> str:
    principal = getattr(request.state, "principal", None)
    if principal is not None:
        return f"user:{principal.subject}"
    return f"ip:{get_remote_address(request)}"


_storage_uri = (
    settings.rate_limit.redis_url.get_secret_value()
    if settings.rate_limit.backend == "redis" and settings.rate_limit.redis_url is not None
    else "memory://"
)

# Module-level singleton. Routers import this directly.
limiter: Limiter = Limiter(
    key_func=_principal_or_ip,
    enabled=os.environ.get("RATE_LIMIT_ENABLED", "true").lower() != "false",
    default_limits=[f"{settings.rate_limit.per_ip_per_minute}/minute"],
    storage_uri=_storage_uri,
)


def install_rate_limiting(app: FastAPI) -> None:
    """Wire the limiter, exception handler, and middleware into the app."""
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _structured_429_handler)
    app.add_middleware(SlowAPIMiddleware)


async def _structured_429_handler(request: Request, exc: Exception) -> JSONResponse:
    """Replace slowapi's plain-text 429 with our structured ErrorResponse shape."""
    # Delegate to slowapi to compute the Retry-After header, then re-wrap the
    # body. Forward ONLY Retry-After — copying base.headers would carry over
    # slowapi's Content-Length, which corrupts our (different-sized) body.
    base = _rate_limit_exceeded_handler(request, exc)  # type: ignore[arg-type]
    forwarded_headers = {}
    if "Retry-After" in base.headers:
        forwarded_headers["Retry-After"] = base.headers["Retry-After"]
    body = {
        "error": "rate_limit_exceeded",
        "detail": str(exc),
        "request_id": request.headers.get("X-Request-ID"),
    }
    return JSONResponse(content=body, status_code=base.status_code, headers=forwarded_headers)
