"""CRUD operations for :class:`Trace` rows.

All methods take an :class:`AsyncSession` so the caller controls transaction
boundaries — typically via :func:`src.storage.database.session_scope`.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

import structlog
from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.auth.principal import Permission, Principal
from src.config.settings import settings
from src.storage.models import DefectEvent, EvalScore, Trace

log = structlog.get_logger(__name__)

QualityGateFilter = Literal["pass", "warn", "fail", "skip"]


async def create_trace(
    session: AsyncSession,
    *,
    query_id: str,
    trace_id: str,
    query_text: str,
    answer_text: str,
    context_truncated: bool,
    latency_ms: int,
    top_k: int,
    chunks_retrieved: int,
    model_used: str,
    user_id: str | None = None,
    organization_id: str | None = None,
    retrieval_strategy: str = "hybrid",
) -> Trace:
    if not settings.auth.enabled:
        user_id = user_id or settings.auth.disabled_local_user_id
        organization_id = organization_id or settings.auth.disabled_local_organization_id
    trace = Trace(
        query_id=query_id,
        trace_id=trace_id,
        query_text=query_text,
        answer_text=answer_text,
        context_truncated=context_truncated,
        latency_ms=latency_ms,
        retrieval_strategy=retrieval_strategy,
        top_k=top_k,
        chunks_retrieved=chunks_retrieved,
        model_used=model_used,
        user_id=user_id,
        organization_id=organization_id,
    )
    session.add(trace)
    await session.flush()
    return trace


async def get_by_trace_id(
    session: AsyncSession, trace_id: str, principal: Principal | None = None
) -> Trace | None:
    stmt = select(Trace).where(Trace.trace_id == trace_id)
    if principal is not None and not principal.has(Permission.OBSERVABILITY_ADMIN):
        stmt = stmt.where(Trace.user_id == principal.subject)
    stmt = stmt.limit(1)
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def get_by_query_id(session: AsyncSession, query_id: str) -> Trace | None:
    stmt = select(Trace).where(Trace.query_id == query_id).limit(1)
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def list_traces(
    session: AsyncSession,
    *,
    limit: int = 50,
    offset: int = 0,
    has_defects: bool | None = None,
    quality_gate: QualityGateFilter | None = None,
    since: datetime | None = None,
    principal: Principal | None = None,
) -> tuple[list[Trace], int]:
    """List traces with pagination + optional filters.

    Returns ``(items, total_count)``. Total honours the same filter set so
    pagination math in callers is straightforward.
    """
    base = select(Trace)
    count_base = select(func.count()).select_from(Trace)

    conditions = []
    if principal is not None and not principal.has(Permission.OBSERVABILITY_ADMIN):
        conditions.append(Trace.user_id == principal.subject)
    if since is not None:
        conditions.append(Trace.created_at >= since)

    if has_defects is True:
        subq = select(DefectEvent.query_id).distinct().subquery()
        conditions.append(Trace.query_id.in_(select(subq)))
    elif has_defects is False:
        subq = select(DefectEvent.query_id).distinct().subquery()
        conditions.append(~Trace.query_id.in_(select(subq)))

    if quality_gate is not None:
        subq = (
            select(EvalScore.query_id)
            .where(EvalScore.quality_gate_result == quality_gate)
            .distinct()
            .subquery()
        )
        conditions.append(Trace.query_id.in_(select(subq)))

    if conditions:
        base = base.where(and_(*conditions))
        count_base = count_base.where(and_(*conditions))

    base = base.order_by(Trace.created_at.desc()).limit(limit).offset(offset)

    items_result = await session.execute(base)
    total_result = await session.execute(count_base)
    items = list(items_result.scalars().all())
    total = int(total_result.scalar_one())
    return items, total
