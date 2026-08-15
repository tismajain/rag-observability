"""POST /query — the main RAG endpoint.

Wraps the full pipeline in a root OTel span, runs defect detection inline,
schedules async eval, and returns a typed response. Errors are translated
into a structured :class:`ErrorResponse` so internal stack traces never
reach the client.
"""

from __future__ import annotations

from uuid import uuid4

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, status
from opentelemetry import trace

from src.api.resource_controls import enforce_endpoint_rate, request_capacity, reserve_llm_cost
from src.api.schemas import ErrorResponse, QueryRequest, QueryResponse, RetrievedChunkSummary
from src.auth.dependencies import require_query
from src.auth.principal import Principal
from src.config.settings import settings
from src.observability.attributes import SpanAttributes
from src.observability.spans import root_span
from src.pipeline.pipeline_runner import pipeline_runner

router = APIRouter(tags=["query"])
log = structlog.get_logger(__name__)

TEXT_PREVIEW_CHARS = 240


@router.post(
    "/query",
    response_model=QueryResponse,
    responses={
        429: {"model": ErrorResponse, "description": "Rate limit exceeded."},
        503: {
            "model": ErrorResponse,
            "description": "Pipeline not ready (missing API keys / Qdrant down).",
        },
        500: {"model": ErrorResponse, "description": "Unexpected pipeline error."},
    },
)
async def query(
    request: Request,
    payload: QueryRequest,
    principal: Principal = Depends(require_query),
) -> QueryResponse:
    await enforce_endpoint_rate(
        principal,
        endpoint="query",
        client_ip=request.client.host if request.client else "unknown",
        limit=settings.rate_limit.query_per_minute,
    )
    await reserve_llm_cost(principal, len(payload.query) + settings.llm.max_tokens)
    try:
        async with request_capacity(principal):
            return await _run_query(request, payload, principal)
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("api.query.failed", query_id="unassigned", error_type=type(exc).__name__)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query failed: {type(exc).__name__}",
        ) from exc


async def _run_query(
    request: Request, payload: QueryRequest, principal: Principal
) -> QueryResponse:
    if not pipeline_runner.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Pipeline unavailable: {pipeline_runner.build_error}",
        )
    query_id = str(uuid4())
    pipeline = pipeline_runner.pipeline
    try:
        with root_span(query_id, payload.query) as span:
            result = await pipeline.run(
                query_id=query_id,
                query=payload.query,
                top_k=payload.top_k,
                enable_reranking=payload.enable_reranking,
                scope=principal.authorization_scope(),
            )
            # Roll up summary attributes onto the root span so Phoenix shows
            # the headline numbers without expanding children.
            span.set_attribute(SpanAttributes.RETRIEVAL_FUSED_HITS, len(result.retrieved_chunks))
            span.set_attribute(
                SpanAttributes.CONTEXT_TRUNCATED, result.assembled_context.was_truncated
            )
            span.set_attribute(SpanAttributes.GENERATION_MODEL, result.llm.model)
            span.set_attribute(SpanAttributes.GENERATION_PROVIDER, result.llm.provider)
            span.set_attribute(SpanAttributes.DEFECT_COUNT, len(result.defects))

            trace_id = _trace_id_hex(span)

    except Exception as exc:
        log.exception("api.query.failed", query_id=query_id, error_type=type(exc).__name__)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query failed: {type(exc).__name__}",
        ) from exc

    # Persist trace + defects synchronously so /traces sees the row immediately
    # on its next call. Best-effort: DB outages do not fail the request.
    await pipeline_runner.persist_result(
        result,
        query_text=payload.query,
        trace_id=trace_id,
        top_k=payload.top_k,
        principal=principal,
    )

    eval_scheduled = pipeline_runner.schedule_eval(result, payload.query, trace_id, principal)

    return QueryResponse(
        query_id=query_id,
        trace_id=trace_id,
        answer=result.answer,
        retrieved_chunks=[
            RetrievedChunkSummary(
                chunk_id=c.chunk_id,
                rrf_score=c.rrf_score,
                dense_score=c.dense_score,
                sparse_score=c.sparse_score,
                source_file=c.metadata.source_file,
                chunk_index=c.metadata.chunk_index,
                text_preview=c.text[:TEXT_PREVIEW_CHARS],
            )
            for c in result.retrieved_chunks
        ],
        context_truncated=result.assembled_context.was_truncated,
        defects_detected=[d.defect_type.value for d in result.defects],
        latency_ms=result.total_latency_ms,
        eval_scheduled=eval_scheduled,
    )


def _trace_id_hex(span: trace.Span) -> str:
    ctx = span.get_span_context()
    if ctx and ctx.is_valid:
        return f"{ctx.trace_id:032x}"
    return "0" * 32
