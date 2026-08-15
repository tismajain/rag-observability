"""GET /defects — historical defect listing, backed by the Postgres defect store."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import structlog
from fastapi import APIRouter, Depends, Query

from src.auth.dependencies import require_observability_read
from src.auth.principal import Principal
from src.storage import defect_store
from src.storage.database import session_scope

router = APIRouter(tags=["defects"])
log = structlog.get_logger(__name__)


def _serialize(d: Any) -> dict[str, Any]:
    return {
        "id": d.id,
        "query_id": d.query_id,
        "trace_id": d.trace_id,
        "defect_type": d.defect_type,
        "severity": d.severity,
        "description": d.description,
        "metadata": d.event_metadata,
        "detected_at": d.detected_at.isoformat(),
    }


@router.get("/defects")
async def list_defects(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    severity: str | None = Query(None),
    defect_type: str | None = Query(None),
    since: datetime | None = Query(None),
    principal: Principal = Depends(require_observability_read),
) -> dict[str, Any]:
    async with session_scope() as session:
        items, total = await defect_store.list_defects(
            session,
            limit=limit,
            offset=offset,
            severity=severity,
            defect_type=defect_type,
            since=since,
            principal=principal,
        )
    return {
        "items": [_serialize(d) for d in items],
        "total": total,
        "limit": limit,
        "offset": offset,
    }
