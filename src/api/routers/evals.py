"""GET /evals + /evals/summary — eval scores and aggregates, backed by Postgres."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status

from src.auth.dependencies import require_observability_read
from src.auth.principal import Principal
from src.storage import eval_store
from src.storage.database import session_scope

router = APIRouter(tags=["evals"])
log = structlog.get_logger(__name__)

_PERIOD_RE = re.compile(r"^(\d+)([dh])$")


def _serialize(e: Any) -> dict[str, Any]:
    return {
        "id": e.id,
        "query_id": e.query_id,
        "trace_id": e.trace_id,
        "faithfulness": e.faithfulness,
        "context_recall": e.context_recall,
        "answer_relevancy": e.answer_relevancy,
        "quality_gate_result": e.quality_gate_result,
        "evaluation_latency_ms": e.evaluation_latency_ms,
        "judge_model": e.judge_model,
        "status": e.status,
        "error": e.error,
        "evaluated_at": e.evaluated_at.isoformat(),
    }


@router.get("/evals")
async def list_evals(
    query_id: str | None = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    principal: Principal = Depends(require_observability_read),
) -> dict[str, Any]:
    async with session_scope() as session:
        items, total = await eval_store.list_evals(
            session, query_id=query_id, limit=limit, offset=offset, principal=principal
        )
    return {
        "items": [_serialize(e) for e in items],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get("/evals/summary")
async def evals_summary(
    period: str = Query("7d", description="Period suffix: e.g. '7d', '24h'."),
    principal: Principal = Depends(require_observability_read),
) -> dict[str, Any]:
    days = _parse_period_to_days(period)
    async with session_scope() as session:
        summary = await eval_store.summary(session, period_days=days, principal=principal)
    since = summary["since"]
    since_iso = since.isoformat() if isinstance(since, datetime) else None
    return {
        "period": period,
        "since": since_iso,
        "mean_faithfulness": summary["mean_faithfulness"],
        "mean_context_recall": summary["mean_context_recall"],
        "mean_answer_relevancy": summary["mean_answer_relevancy"],
        "total_evaluated": summary["total_evaluated"],
        "quality_gate_distribution": summary["quality_gate_distribution"],
    }


def _parse_period_to_days(period: str) -> int:
    """Parse '7d' / '24h' → integer days (rounded up for hours)."""
    m = _PERIOD_RE.match(period.strip())
    if not m:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid period {period!r}; expected '<N>d' or '<N>h'",
        )
    value, unit = int(m.group(1)), m.group(2)
    if value <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid period {period!r}; must be positive",
        )
    return value if unit == "d" else max(1, (value + 23) // 24)
