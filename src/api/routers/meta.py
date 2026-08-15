"""POST /meta/query — natural-language Q&A over the system's own trace history.

Wraps the meta-RAG pipeline (retrieve → assemble → generate) with the same
observability shape as POST /query: a root span, structured logging, and a
typed response. Defect detection + eval are intentionally absent — those
quality gates are calibrated for the production RAG path, not the
observability-analyst surface.
"""

from __future__ import annotations

from uuid import uuid4

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, status
from opentelemetry import trace

from src.api.resource_controls import enforce_endpoint_rate, request_capacity, reserve_llm_cost
from src.api.schemas import (
    ErrorResponse,
    MetaQueryRequest,
    MetaQueryResponse,
    RetrievedTraceSummary,
)
from src.auth.dependencies import require_observability_read
from src.auth.principal import Principal
from src.config.settings import settings
from src.observability.attributes import SpanAttributes
from src.observability.spans import root_span
from src.pipeline.pipeline_runner import pipeline_runner

router = APIRouter(tags=["meta"])
log = structlog.get_logger(__name__)

TEXT_PREVIEW_CHARS = 240


@router.post(
    "/meta/query",
    response_model=MetaQueryResponse,
    responses={
        429: {"model": ErrorResponse, "description": "Rate limit exceeded."},
        503: {
            "model": ErrorResponse,
            "description": "Pipeline not ready (missing API keys / Qdrant down).",
        },
        500: {"model": ErrorResponse, "description": "Unexpected meta pipeline error."},
    },
)
async def meta_query(
    request: Request,
    payload: MetaQueryRequest,
    principal: Principal = Depends(require_observability_read),
) -> MetaQueryResponse:
    await enforce_endpoint_rate(
        principal,
        endpoint="meta-query",
        client_ip=request.client.host if request.client else "unknown",
        limit=settings.rate_limit.meta_query_per_minute,
    )
    await reserve_llm_cost(principal, len(payload.query) + settings.llm.max_tokens)
    async with request_capacity(principal):
        return await _run_meta_query(request, payload, principal)


async def _run_meta_query(
    request: Request, payload: MetaQueryRequest, principal: Principal
) -> MetaQueryResponse:
    if not pipeline_runner.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Pipeline unavailable: {pipeline_runner.build_error}",
        )

    query_id = str(uuid4())
    meta_pipeline = pipeline_runner.meta_pipeline

    try:
        with root_span(query_id, payload.query) as span:
            result = await meta_pipeline.run(
                query_id=query_id,
                query=payload.query,
                top_k=payload.top_k,
                scope=principal.authorization_scope(),
            )
            span.set_attribute(SpanAttributes.RETRIEVAL_FUSED_HITS, len(result.retrieved_traces))
            span.set_attribute(
                SpanAttributes.CONTEXT_TRUNCATED, result.assembled_context.was_truncated
            )
            span.set_attribute(SpanAttributes.GENERATION_MODEL, result.llm.model)
            span.set_attribute(SpanAttributes.GENERATION_PROVIDER, result.llm.provider)

            trace_id = _trace_id_hex(span)
    except Exception as exc:
        log.exception("api.meta_query.failed", query_id=query_id, error_type=type(exc).__name__)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Meta query failed: {type(exc).__name__}",
        ) from exc

    return MetaQueryResponse(
        query_id=query_id,
        trace_id=trace_id,
        answer=result.answer,
        retrieved_traces=[
            RetrievedTraceSummary(
                query_id=_extract_query_id(c.metadata.source_file),
                rrf_score=c.rrf_score,
                text_preview=c.text[:TEXT_PREVIEW_CHARS],
            )
            for c in result.retrieved_traces
        ],
        context_truncated=result.assembled_context.was_truncated,
        latency_ms=result.total_latency_ms,
    )


def _extract_query_id(source_file: str) -> str:
    """Trace-doc source_file is encoded as ``trace:<query_id>``; strip the prefix."""
    if source_file.startswith("trace:"):
        return source_file[len("trace:") :]
    return source_file


def _trace_id_hex(span: trace.Span) -> str:
    ctx = span.get_span_context()
    if ctx and ctx.is_valid:
        return f"{ctx.trace_id:032x}"
    return "0" * 32
