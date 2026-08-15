"""GET /traces — historical trace listing, backed by the Postgres trace store."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status

from src.auth.dependencies import require_observability_read
from src.auth.principal import Principal
from src.storage import defect_store, eval_store, trace_store
from src.storage.database import session_scope

router = APIRouter(tags=["traces"])
log = structlog.get_logger(__name__)


def _serialize_trace(trace: Any, principal: Principal) -> dict[str, Any]:
    can_read_content = trace.user_id == principal.subject or principal.has("document:content:admin")
    return {
        "id": trace.id,
        "query_id": trace.query_id,
        "trace_id": trace.trace_id,
        "query_text": trace.query_text if can_read_content else "[REDACTED]",
        "answer_text": trace.answer_text if can_read_content else "[REDACTED]",
        "context_truncated": trace.context_truncated,
        "latency_ms": trace.latency_ms,
        "retrieval_strategy": trace.retrieval_strategy,
        "top_k": trace.top_k,
        "chunks_retrieved": trace.chunks_retrieved,
        "model_used": trace.model_used,
        "created_at": trace.created_at.isoformat(),
    }


def _serialize_defect(d: Any) -> dict[str, Any]:
    return {
        "id": d.id,
        "defect_type": d.defect_type,
        "severity": d.severity,
        "description": d.description,
        "metadata": d.event_metadata,
        "detected_at": d.detected_at.isoformat(),
    }


def _serialize_eval(e: Any) -> dict[str, Any]:
    return {
        "id": e.id,
        "faithfulness": e.faithfulness,
        "context_recall": e.context_recall,
        "answer_relevancy": e.answer_relevancy,
        "quality_gate_result": e.quality_gate_result,
        "evaluation_latency_ms": e.evaluation_latency_ms,
        "judge_model": e.judge_model,
        "status": e.status,
        "evaluated_at": e.evaluated_at.isoformat(),
    }


@router.get("/traces")
async def list_traces(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    has_defects: bool | None = Query(None),
    quality_gate: Literal["pass", "warn", "fail", "skip"] | None = Query(None),
    since: datetime | None = Query(None),
    principal: Principal = Depends(require_observability_read),
) -> dict[str, Any]:
    async with session_scope() as session:
        items, total = await trace_store.list_traces(
            session,
            limit=limit,
            offset=offset,
            has_defects=has_defects,
            quality_gate=quality_gate,
            since=since,
            principal=principal,
        )
    return {
        "items": [_serialize_trace(t, principal) for t in items],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get("/traces/{trace_id}")
async def get_trace(
    trace_id: str, principal: Principal = Depends(require_observability_read)
) -> dict[str, Any]:
    async with session_scope() as session:
        trace = await trace_store.get_by_trace_id(session, trace_id, principal)
        if trace is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No trace found with trace_id={trace_id}",
            )
        defects = await defect_store.defects_by_query(session, trace.query_id, principal)
        evals, _ = await eval_store.list_evals(
            session, query_id=trace.query_id, limit=10, principal=principal
        )

    return {
        "trace": _serialize_trace(trace, principal),
        "defects": [_serialize_defect(d) for d in defects],
        "evals": [_serialize_eval(e) for e in evals],
    }
