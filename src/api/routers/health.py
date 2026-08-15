"""Health endpoint.

Probes Postgres + Qdrant + (best-effort) Phoenix. Returns 200 when all
required dependencies are healthy, 503 otherwise.

Phoenix is informational: tracing degrades gracefully (spans buffer) when
Phoenix is down, so a Phoenix outage does NOT flip the gate to unhealthy —
it surfaces as a status field instead.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import structlog
from fastapi import APIRouter, Depends, Response, status
from pydantic import BaseModel
from sqlalchemy import text

from src.auth.dependencies import require_observability_admin
from src.auth.principal import Principal
from src.config.settings import settings
from src.storage.database import get_session_factory

router = APIRouter(tags=["health"])
log = structlog.get_logger(__name__)


class DependencyStatus(BaseModel):
    name: str
    healthy: bool
    detail: str | None = None


class HealthResponse(BaseModel):
    status: str
    environment: str
    app_name: str
    dependencies: list[DependencyStatus]


async def _check_postgres() -> DependencyStatus:
    try:
        factory = get_session_factory()
        async with factory() as session:
            await session.execute(text("SELECT 1"))
        return DependencyStatus(name="postgres", healthy=True)
    except Exception as exc:  # noqa: BLE001
        return DependencyStatus(name="postgres", healthy=False, detail=str(exc)[:200])


async def _check_qdrant() -> DependencyStatus:
    url = f"http://{settings.qdrant.host}:{settings.qdrant.port}/readyz"
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            r = await client.get(url)
        if r.status_code == 200:
            return DependencyStatus(name="qdrant", healthy=True)
        return DependencyStatus(name="qdrant", healthy=False, detail=f"HTTP {r.status_code}")
    except Exception as exc:  # noqa: BLE001
        return DependencyStatus(name="qdrant", healthy=False, detail=str(exc)[:200])


async def _check_phoenix() -> DependencyStatus:
    # Phoenix endpoint is the OTLP gRPC port; UI is on a sibling HTTP port.
    # Pinging the UI health endpoint is the cheapest "is it up" probe.
    endpoint = settings.observability.phoenix_endpoint
    ui_url = endpoint.replace(":4317", ":6006").rstrip("/") + "/healthz"
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            r = await client.get(ui_url)
        if r.status_code in (200, 204):
            return DependencyStatus(name="phoenix", healthy=True)
        return DependencyStatus(name="phoenix", healthy=False, detail=f"HTTP {r.status_code}")
    except Exception as exc:  # noqa: BLE001
        return DependencyStatus(name="phoenix", healthy=False, detail=str(exc)[:200])


@router.get("/health/live")
@router.get("/health")
async def live() -> dict[str, str]:
    """Minimal process liveness; anonymous and dependency-free."""
    return {"status": "ok"}


@router.get("/health/ready", response_model=HealthResponse)
async def readiness(
    response: Response,
    principal: Principal = Depends(require_observability_admin),
) -> HealthResponse:
    pg, qd, px = await asyncio.gather(_check_postgres(), _check_qdrant(), _check_phoenix())
    deps = [pg, qd, px]
    # Required for "healthy": postgres + qdrant. Phoenix degradation is allowed.
    required_ok = pg.healthy and qd.healthy
    if not required_ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        overall: str = "unhealthy"
    elif not px.healthy:
        overall = "degraded"
    else:
        overall = "ok"
    return HealthResponse(
        status=overall,
        environment=settings.environment,
        app_name=settings.app_name,
        dependencies=deps,
    )


def _summary(deps: list[DependencyStatus]) -> dict[str, Any]:
    return {d.name: ("ok" if d.healthy else d.detail or "down") for d in deps}
