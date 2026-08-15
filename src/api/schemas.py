"""Public request/response schemas for the HTTP API.

These models are the contract between callers and the FastAPI handlers.
Kept in one module so the API surface is discoverable and easy to diff.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from src.evaluation.quality_gate import QualityGateResult


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    enable_reranking: bool | None = Field(
        default=None,
        description="When set, overrides the server's default reranker policy.",
    )
    collection_name: str | None = Field(
        default=None,
        description="Override the configured Qdrant collection. Defaults to settings value.",
    )


class RetrievedChunkSummary(BaseModel):
    chunk_id: str
    rrf_score: float
    dense_score: float | None = None
    sparse_score: float | None = None
    source_file: str
    chunk_index: int
    text_preview: str = Field(..., description="First 240 chars of the chunk text.")


class QueryResponse(BaseModel):
    query_id: str
    trace_id: str
    answer: str
    retrieved_chunks: list[RetrievedChunkSummary]
    context_truncated: bool
    defects_detected: list[str] = Field(
        default_factory=list,
        description="Defect type names (DEFECT_*). Empty list means no defects fired.",
    )
    latency_ms: int
    eval_scheduled: bool


class ErrorResponse(BaseModel):
    error: str
    detail: str
    request_id: str | None = None


class DocumentResponse(BaseModel):
    id: str
    title: str
    content_type: str
    byte_size: int
    visibility: Literal["private", "shared"]
    ingestion_status: Literal["queued", "processing", "completed", "failed"]
    owned_by_requester: bool
    created_at: datetime
    updated_at: datetime


class DocumentListResponse(BaseModel):
    items: list[DocumentResponse]
    limit: int
    offset: int


class DocumentUpdateRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=512)


class DocumentShareRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=255)


class IngestionJobResponse(BaseModel):
    id: str
    document_id: str
    status: Literal["queued", "processing", "completed", "failed"]
    error_code: str | None = None


# Placeholder schemas for the observability endpoints that land in Phase 6.
# Defined now so the API docs show the eventual shape even while the handlers
# return 501.


class PaginatedTraces(BaseModel):
    items: list[dict[str, object]]
    total: int
    limit: int
    offset: int


class PaginatedDefects(BaseModel):
    items: list[dict[str, object]]
    total: int
    limit: int
    offset: int


class EvalSummary(BaseModel):
    period: str
    mean_faithfulness: float | None
    mean_context_recall: float | None
    mean_answer_relevancy: float | None
    quality_gate_distribution: dict[QualityGateResult, int]
    total_evaluated: int


# ---------------------------------------------------------------- meta-RAG --


class MetaQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=8, ge=1, le=20)


class RetrievedTraceSummary(BaseModel):
    query_id: str
    rrf_score: float
    text_preview: str = Field(..., description="First 240 chars of the trace document.")


class MetaQueryResponse(BaseModel):
    query_id: str
    trace_id: str
    answer: str
    retrieved_traces: list[RetrievedTraceSummary]
    context_truncated: bool
    latency_ms: int
