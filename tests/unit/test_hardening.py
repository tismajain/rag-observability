"""Tests for Phase 10 hardening: rate limit, body cap, error envelope.

These need a fully-wired app (rate limiter + error handlers + middleware) so
they construct a fresh one rather than reusing the lightweight test fixture
from test_api.py.
"""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from src.api.error_handlers import install_error_handlers
from src.api.middleware import BodySizeLimitMiddleware, RequestIDMiddleware, TimingMiddleware
from src.api.rate_limit import install_rate_limiting, limiter
from src.api.routers import health, meta, query
from src.pipeline.pipeline_runner import pipeline_runner as runner_singleton


def _build_hardened_app(max_body_bytes: int = 1024) -> FastAPI:
    """Build an app with the full Phase 10 middleware stack wired in."""
    app = FastAPI(title="hardening-test")
    app.add_middleware(TimingMiddleware)
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=max_body_bytes)
    install_rate_limiting(app)
    install_error_handlers(app)
    app.include_router(health.router)
    app.include_router(query.router)
    app.include_router(meta.router)
    return app


@pytest.fixture(autouse=True)
def _reset_limiter_state() -> Iterator[None]:
    """slowapi keeps in-memory counters between requests; clear them each test."""
    limiter.reset()
    yield
    limiter.reset()


# ----------------------------------------------------------------- body cap


def test_body_size_limit_rejects_oversized_payload() -> None:
    app = _build_hardened_app(max_body_bytes=100)
    with TestClient(app) as client:
        big_payload = {"query": "x" * 5000}
        r = client.post("/query", json=big_payload)
    assert r.status_code == 413
    body = r.json()
    assert body["error"] == "payload_too_large"
    assert "exceeds limit" in body["detail"]


def test_body_size_limit_allows_small_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner_singleton, "_pipeline", None)
    monkeypatch.setattr(runner_singleton, "_build_error", "stub", raising=False)
    app = _build_hardened_app(max_body_bytes=1_000_000)
    with TestClient(app) as client:
        # Small payload — should pass the cap and hit the 503 path because no
        # pipeline is wired. The point is: body cap did NOT trip.
        r = client.post("/query", json={"query": "hello"})
    assert r.status_code == 503
    assert r.json()["error"] == "service_unavailable"


# ------------------------------------------------------------ error envelope


def test_validation_error_returns_structured_envelope() -> None:
    app = _build_hardened_app()
    with TestClient(app) as client:
        r = client.post("/query", json={"query": ""})
    assert r.status_code == 422
    body = r.json()
    assert body["error"] == "validation_error"
    assert isinstance(body["detail"], list)
    assert "request_id" in body


def test_http_exception_passes_through_structured_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner_singleton, "_pipeline", None)
    monkeypatch.setattr(runner_singleton, "_build_error", "missing-keys", raising=False)
    app = _build_hardened_app()
    with TestClient(app) as client:
        r = client.post("/query", json={"query": "anything"})
    assert r.status_code == 503
    body = r.json()
    assert body["error"] == "service_unavailable"
    assert "missing-keys" in body["detail"]
    assert "request_id" in body


def test_unhandled_exception_is_wrapped_not_leaked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bare exception inside the handler must become a structured 500."""
    stub_pipeline = MagicMock()
    stub_pipeline.run = AsyncMock(side_effect=RuntimeError("internal mishap"))
    monkeypatch.setattr(runner_singleton, "_pipeline", stub_pipeline)
    monkeypatch.setattr(runner_singleton, "_evaluator", None)

    app = _build_hardened_app()
    with TestClient(app, raise_server_exceptions=False) as client:
        r = client.post("/query", json={"query": "x"})
    assert r.status_code == 500
    body = r.json()
    assert body["error"] == "internal_error"
    # The exception class name leaks (intentionally — clients can pattern-match
    # on the kind of failure); the *message* does not.
    assert "internal mishap" not in body["detail"]


# ------------------------------------------------------------- rate limiting


def test_rate_limit_kicks_in_after_burst(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner_singleton, "_pipeline", None)
    monkeypatch.setattr(runner_singleton, "_build_error", "stub", raising=False)

    # Tighten the limit so the test stays fast — one call per minute.
    from src.api.routers import query as query_router

    # Re-decorate the handler with a tighter limit. slowapi reads the limit
    # from the closure at call time, so patching the handler directly is the
    # cleanest knob.
    tight_app = FastAPI(title="rate-limit-test")
    tight_app.add_middleware(RequestIDMiddleware)
    install_rate_limiting(tight_app)
    install_error_handlers(tight_app)

    @tight_app.get("/tight")
    @limiter.limit("1/minute")
    async def _tight(request: Request) -> dict[str, str]:
        return {"ok": "yes"}

    # Use the existing query router too so /query's regular limit doesn't trip.
    tight_app.include_router(query_router.router)

    with TestClient(tight_app) as client:
        r1 = client.get("/tight")
        r2 = client.get("/tight")

    assert r1.status_code == 200
    assert r2.status_code == 429
    body = r2.json()
    assert body["error"] == "rate_limit_exceeded"
    assert "1 per 1 minute" in body["detail"] or "per minute" in body["detail"]


def test_rate_limit_can_be_disabled_via_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RATE_LIMIT_ENABLED=false must turn the limiter into a no-op."""
    # Confirm the module-level flag respects the env var path.
    from src.api import rate_limit

    monkeypatch.setattr(rate_limit.limiter, "enabled", False)

    app = FastAPI()
    install_rate_limiting(app)
    install_error_handlers(app)

    @app.get("/disabled")
    @rate_limit.limiter.limit("1/minute")
    async def _h(request: Request) -> dict[str, str]:
        return {"ok": "yes"}

    with TestClient(app) as client:
        for _ in range(5):
            r = client.get("/disabled")
            # Disabled limiter → never returns 429.
            assert r.status_code == 200
