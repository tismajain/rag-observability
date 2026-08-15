"""CRUD operations for :class:`DefectEvent` rows."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import structlog
from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.auth.principal import Permission, Principal
from src.config.settings import settings
from src.storage.models import DefectEvent

log = structlog.get_logger(__name__)


async def create_defect(
    session: AsyncSession,
    *,
    query_id: str,
    trace_id: str,
    defect_type: str,
    severity: str,
    description: str,
    event_metadata: dict[str, Any] | None = None,
    defect_id: str | None = None,
) -> DefectEvent:
    if not settings.auth.enabled:
        kwargs_user_id = settings.auth.disabled_local_user_id
        kwargs_organization_id = settings.auth.disabled_local_organization_id
    else:
        kwargs_user_id = None
        kwargs_organization_id = None
    kwargs: dict[str, Any] = {
        "query_id": query_id,
        "trace_id": trace_id,
        "defect_type": defect_type,
        "severity": severity,
        "description": description,
        "event_metadata": event_metadata or {},
        "user_id": kwargs_user_id,
        "organization_id": kwargs_organization_id,
    }
    if defect_id is not None:
        kwargs["id"] = defect_id
    defect = DefectEvent(**kwargs)
    session.add(defect)
    await session.flush()
    return defect


async def bulk_create_defects(
    session: AsyncSession,
    defects: list[dict[str, Any]],
) -> int:
    """Insert many defects in one round-trip. Returns the count written."""
    if not defects:
        return 0
    if not settings.auth.enabled:
        for defect in defects:
            defect.setdefault("user_id", settings.auth.disabled_local_user_id)
            defect.setdefault("organization_id", settings.auth.disabled_local_organization_id)
    session.add_all([DefectEvent(**d) for d in defects])
    await session.flush()
    return len(defects)


async def list_defects(
    session: AsyncSession,
    *,
    limit: int = 50,
    offset: int = 0,
    severity: str | None = None,
    defect_type: str | None = None,
    since: datetime | None = None,
    principal: Principal | None = None,
) -> tuple[list[DefectEvent], int]:
    base = select(DefectEvent)
    count_base = select(func.count()).select_from(DefectEvent)

    conditions = []
    if principal is not None and not principal.has(Permission.OBSERVABILITY_ADMIN):
        conditions.append(DefectEvent.user_id == principal.subject)
    if severity is not None:
        conditions.append(DefectEvent.severity == severity)
    if defect_type is not None:
        conditions.append(DefectEvent.defect_type == defect_type)
    if since is not None:
        conditions.append(DefectEvent.detected_at >= since)

    if conditions:
        base = base.where(and_(*conditions))
        count_base = count_base.where(and_(*conditions))

    base = base.order_by(DefectEvent.detected_at.desc()).limit(limit).offset(offset)

    items_result = await session.execute(base)
    total_result = await session.execute(count_base)
    return list(items_result.scalars().all()), int(total_result.scalar_one())


async def defects_by_query(
    session: AsyncSession, query_id: str, principal: Principal | None = None
) -> list[DefectEvent]:
    stmt = (
        select(DefectEvent)
        .where(DefectEvent.query_id == query_id)
        .order_by(DefectEvent.detected_at.asc())
    )
    if principal is not None and not principal.has(Permission.OBSERVABILITY_ADMIN):
        stmt = stmt.where(DefectEvent.user_id == principal.subject)
    result = await session.execute(stmt)
    return list(result.scalars().all())
