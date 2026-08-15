"""CRUD operations for :class:`EvalScore` rows."""

from __future__ import annotations

from datetime import datetime, timedelta

import structlog
from sqlalchemy import case, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.auth.principal import Permission, Principal
from src.config.settings import settings
from src.storage.models import EvalScore

log = structlog.get_logger(__name__)


async def create_eval(
    session: AsyncSession,
    *,
    query_id: str,
    trace_id: str,
    faithfulness: float | None,
    context_recall: float | None,
    answer_relevancy: float | None,
    quality_gate_result: str,
    evaluation_latency_ms: int,
    judge_model: str,
    status: str = "ok",
    error: str | None = None,
    user_id: str | None = None,
    organization_id: str | None = None,
) -> EvalScore:
    if not settings.auth.enabled:
        user_id = user_id or settings.auth.disabled_local_user_id
        organization_id = organization_id or settings.auth.disabled_local_organization_id
    score = EvalScore(
        query_id=query_id,
        trace_id=trace_id,
        faithfulness=faithfulness,
        context_recall=context_recall,
        answer_relevancy=answer_relevancy,
        quality_gate_result=quality_gate_result,
        evaluation_latency_ms=evaluation_latency_ms,
        judge_model=judge_model,
        status=status,
        error=error,
        user_id=user_id,
        organization_id=organization_id,
    )
    session.add(score)
    await session.flush()
    return score


async def list_evals(
    session: AsyncSession,
    *,
    query_id: str | None = None,
    limit: int = 50,
    offset: int = 0,
    principal: Principal | None = None,
) -> tuple[list[EvalScore], int]:
    base = select(EvalScore)
    count_base = select(func.count()).select_from(EvalScore)

    if query_id is not None:
        base = base.where(EvalScore.query_id == query_id)
        count_base = count_base.where(EvalScore.query_id == query_id)
    if principal is not None and not principal.has(Permission.OBSERVABILITY_ADMIN):
        base = base.where(EvalScore.user_id == principal.subject)
        count_base = count_base.where(EvalScore.user_id == principal.subject)

    base = base.order_by(EvalScore.evaluated_at.desc()).limit(limit).offset(offset)
    items_result = await session.execute(base)
    total_result = await session.execute(count_base)
    return list(items_result.scalars().all()), int(total_result.scalar_one())


async def summary(
    session: AsyncSession,
    *,
    period_days: int = 7,
    principal: Principal | None = None,
) -> dict[str, object]:
    """Aggregate eval scores over the last ``period_days`` days."""
    since = datetime.utcnow() - timedelta(days=period_days)
    stmt = select(
        func.avg(EvalScore.faithfulness).label("mean_faithfulness"),
        func.avg(EvalScore.context_recall).label("mean_context_recall"),
        func.avg(EvalScore.answer_relevancy).label("mean_answer_relevancy"),
        func.count().label("total"),
        func.sum(case((EvalScore.quality_gate_result == "pass", 1), else_=0)).label("n_pass"),
        func.sum(case((EvalScore.quality_gate_result == "warn", 1), else_=0)).label("n_warn"),
        func.sum(case((EvalScore.quality_gate_result == "fail", 1), else_=0)).label("n_fail"),
        func.sum(case((EvalScore.quality_gate_result == "skip", 1), else_=0)).label("n_skip"),
    ).where(EvalScore.evaluated_at >= since)
    if principal is not None and not principal.has(Permission.OBSERVABILITY_ADMIN):
        stmt = stmt.where(EvalScore.user_id == principal.subject)
    row = (await session.execute(stmt)).one()
    return {
        "period_days": period_days,
        "since": since,
        "mean_faithfulness": float(row.mean_faithfulness)
        if row.mean_faithfulness is not None
        else None,
        "mean_context_recall": float(row.mean_context_recall)
        if row.mean_context_recall is not None
        else None,
        "mean_answer_relevancy": float(row.mean_answer_relevancy)
        if row.mean_answer_relevancy is not None
        else None,
        "total_evaluated": int(row.total or 0),
        "quality_gate_distribution": {
            "pass": int(row.n_pass or 0),
            "warn": int(row.n_warn or 0),
            "fail": int(row.n_fail or 0),
            "skip": int(row.n_skip or 0),
        },
    }
