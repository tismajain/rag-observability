"""Index trace history into the meta Qdrant collection.

Phase 7's meta-RAG layer needs a queryable corpus of *the system's own past
behavior*. This module is the producer: it reads recent rows from the
:class:`Trace` / :class:`DefectEvent` / :class:`EvalScore` tables, formats one
document per query interaction, embeds them with the same embedder the main
pipeline uses, and upserts to a separate Qdrant collection
(``settings.qdrant.meta_collection_name``).

Idempotent: the point ID is a deterministic UUIDv5 derived from ``query_id``,
so re-indexing the same trace overwrites instead of duplicating. Safe to run
on a schedule alongside on-demand CLI invocations.

The producer here is independent of the meta retriever / pipeline; those land
in sibling modules and consume whatever this writes.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timedelta

import structlog
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)
from sqlalchemy import select

from src.config.settings import settings
from src.ingestion.embedder import EmbedderProtocol
from src.observability.tracer import get_tracer
from src.storage.database import session_scope
from src.storage.models import DefectEvent, EvalScore, Trace

log = structlog.get_logger(__name__)

DEFAULT_BATCH_SIZE = 64
DEFAULT_FETCH_LIMIT = 1000

# Namespace UUID for meta-collection point IDs. Independent from the primary
# collection's namespace so the two ID spaces never collide.
_META_POINT_NAMESPACE = uuid.UUID("d1a3f6c2-9b4a-4f0e-8c91-5b2e7d3a9c4f")


def _build_client() -> AsyncQdrantClient:
    return AsyncQdrantClient(
        host=settings.qdrant.host,
        port=settings.qdrant.port,
        prefer_grpc=False,
    )


def _meta_point_id(query_id: str) -> str:
    return str(uuid.uuid5(_META_POINT_NAMESPACE, query_id))


async def init_meta_collection() -> None:
    """Ensure the meta Qdrant collection exists with the expected schema.

    Mirrors :func:`src.ingestion.indexer.init_qdrant` but for the trace
    collection. Safe to call at every startup.
    """
    client = _build_client()
    try:
        existing = {c.name for c in (await client.get_collections()).collections}
        name = settings.qdrant.meta_collection_name
        if name in existing:
            log.info("meta_rag.indexer.collection_exists", collection=name)
            return

        await client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(
                size=settings.qdrant.vector_size,
                distance=Distance.COSINE,
            ),
            hnsw_config=HnswConfigDiff(m=16, ef_construct=200),
        )
        await client.create_payload_index(name, "query_id", PayloadSchemaType.KEYWORD)
        await client.create_payload_index(name, "model_used", PayloadSchemaType.KEYWORD)
        await client.create_payload_index(name, "quality_gate", PayloadSchemaType.KEYWORD)
        await client.create_payload_index(name, "created_at", PayloadSchemaType.DATETIME)
        await client.create_payload_index(name, "authorization_ready", PayloadSchemaType.BOOL)
        await client.create_payload_index(name, "access_subjects", PayloadSchemaType.KEYWORD)
        log.info(
            "meta_rag.indexer.collection_created",
            collection=name,
            vector_size=settings.qdrant.vector_size,
            distance="cosine",
        )
    finally:
        await client.close()


def format_trace_document(
    trace: Trace,
    defects: list[DefectEvent],
    evals: list[EvalScore],
) -> str:
    """Render one trace as a single embedding-ready document string.

    Format is human-readable and dense in named fields so the embedding picks
    up signal from defect types, models, gate results, and source files —
    not just the question + answer text.
    """
    parts: list[str] = []
    parts.append(f"Query ID: {trace.query_id}")
    parts.append(f"Trace ID: {trace.trace_id}")
    parts.append(f"Timestamp: {trace.created_at.isoformat()}")
    parts.append(f"Model: {trace.model_used}")
    parts.append(f"Latency: {trace.latency_ms}ms")
    parts.append(f"Top-K: {trace.top_k}")
    parts.append(f"Chunks retrieved: {trace.chunks_retrieved}")
    parts.append(f"Retrieval strategy: {trace.retrieval_strategy}")
    parts.append(f"Context truncated: {trace.context_truncated}")
    parts.append("")
    parts.append(f"Question: {trace.query_text}")
    parts.append("")
    parts.append(f"Answer: {trace.answer_text}")

    if defects:
        parts.append("")
        parts.append("Defects detected:")
        for d in defects:
            parts.append(f"- {d.defect_type} ({d.severity}): {d.description}")
    else:
        parts.append("")
        parts.append("Defects detected: none")

    if evals:
        parts.append("")
        parts.append("Evaluation scores:")
        for ev in evals:
            faith = _fmt_score(ev.faithfulness)
            recall = _fmt_score(ev.context_recall)
            relev = _fmt_score(ev.answer_relevancy)
            parts.append(
                f"- Faithfulness: {faith}, Context recall: {recall}, "
                f"Answer relevancy: {relev}, Quality gate: {ev.quality_gate_result}, "
                f"Status: {ev.status}, Judge: {ev.judge_model}"
            )
    else:
        parts.append("")
        parts.append("Evaluation scores: not evaluated")

    return "\n".join(parts)


def _build_payload(
    trace: Trace,
    defects: list[DefectEvent],
    evals: list[EvalScore],
    document: str,
) -> dict[str, object]:
    quality_gate = evals[0].quality_gate_result if evals else "skip"
    faithfulness = evals[0].faithfulness if evals else None
    context_recall = evals[0].context_recall if evals else None
    answer_relevancy = evals[0].answer_relevancy if evals else None
    return {
        "text": document,
        # source_file/chunk_index/total_chunks/etc. are reused for the meta
        # collection so the shared ChunkMetadata adapter works unchanged.
        # ``trace:<query_id>`` is a stable handle the meta API surfaces back
        # to clients.
        "source_file": f"trace:{trace.query_id}",
        "chunk_index": 0,
        "total_chunks": 1,
        # Reuses the existing ChunkMetadata Literal — meta docs aren't really
        # "chunked"; the field exists only to satisfy the shared schema.
        "chunking_strategy": "recursive",
        "char_start": 0,
        "char_end": len(document),
        "token_count": 0,
        "ingested_at": trace.created_at.isoformat(),
        "query_id": trace.query_id,
        "trace_id": trace.trace_id,
        "query_text": trace.query_text,
        "answer_text": trace.answer_text,
        "model_used": trace.model_used,
        "latency_ms": trace.latency_ms,
        "top_k": trace.top_k,
        "chunks_retrieved": trace.chunks_retrieved,
        "context_truncated": trace.context_truncated,
        "created_at": trace.created_at.isoformat(),
        "defect_count": len(defects),
        "defect_types": [d.defect_type for d in defects],
        "quality_gate": quality_gate,
        "faithfulness": faithfulness,
        "context_recall": context_recall,
        "answer_relevancy": answer_relevancy,
        "owner_user_id": trace.user_id,
        "organization_id": trace.organization_id,
        "visibility": "private",
        "access_subjects": [f"user:{trace.user_id}"] if trace.user_id else [],
        "authorization_ready": bool(trace.user_id and trace.organization_id),
    }


class TraceIndexer:
    """Pulls traces from Postgres, embeds them, upserts to the meta collection."""

    def __init__(
        self,
        embedder: EmbedderProtocol,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self._embedder = embedder
        self._batch_size = batch_size
        self._client = _build_client()
        self._collection = settings.qdrant.meta_collection_name
        self._tracer = get_tracer(__name__)

    async def ensure_ready(self) -> None:
        try:
            await self._client.get_collection(self._collection)
        except (UnexpectedResponse, ValueError):
            await init_meta_collection()

    async def reindex(
        self,
        since: datetime | None = None,
        limit: int = DEFAULT_FETCH_LIMIT,
    ) -> int:
        """Fetch recent traces and upsert them as meta-collection points.

        ``since`` defaults to "everything"; the periodic scheduler passes a
        rolling window. Returns the number of points written.
        """
        await self.ensure_ready()

        rows = await self._fetch_traces(since=since, limit=limit)
        if not rows:
            log.info("meta_rag.indexer.no_traces", since=since.isoformat() if since else None)
            return 0

        documents = [format_trace_document(t, defects, evals) for t, defects, evals in rows]
        payloads = [
            _build_payload(t, defects, evals, doc)
            for (t, defects, evals), doc in zip(rows, documents, strict=True)
        ]
        ids = [_meta_point_id(t.query_id) for t, _, _ in rows]

        with self._tracer.start_as_current_span("meta_rag.indexer.reindex") as span:
            span.set_attribute("meta_rag.indexer.collection", self._collection)
            span.set_attribute("meta_rag.indexer.candidate_count", len(rows))

            t0 = time.perf_counter()
            written = 0
            for start in range(0, len(documents), self._batch_size):
                end = start + self._batch_size
                batch_texts = documents[start:end]
                batch_payloads = payloads[start:end]
                batch_ids = ids[start:end]

                vectors = await self._embedder.embed_texts(batch_texts)
                points = [
                    PointStruct(id=pid, vector=vec, payload=payload)
                    for pid, vec, payload in zip(batch_ids, vectors, batch_payloads, strict=True)
                ]
                await self._client.upsert(
                    collection_name=self._collection,
                    points=points,
                    wait=True,
                )
                written += len(points)

            latency_ms = int((time.perf_counter() - t0) * 1000)
            span.set_attribute("meta_rag.indexer.written", written)
            span.set_attribute("meta_rag.indexer.latency_ms", latency_ms)

        log.info(
            "meta_rag.indexer.completed",
            written=written,
            collection=self._collection,
            latency_ms=latency_ms,
            since=since.isoformat() if since else None,
        )
        return written

    async def aclose(self) -> None:
        await self._client.close()

    # ----------------------------------------------------------------- fetch

    async def _fetch_traces(
        self,
        since: datetime | None,
        limit: int,
    ) -> list[tuple[Trace, list[DefectEvent], list[EvalScore]]]:
        async with session_scope() as session:
            stmt = select(Trace).order_by(Trace.created_at.desc()).limit(limit)
            if since is not None:
                stmt = stmt.where(Trace.created_at >= since)
            result = await session.execute(stmt)
            traces = list(result.scalars().all())
            # SQLAlchemy's lazy="selectin" on the relationships means the
            # defects/evals collections are eagerly loaded in the same flush.
            return [(t, list(t.defects), list(t.evals)) for t in traces]


async def reindex_recent(
    embedder: EmbedderProtocol,
    window_minutes: int | None = None,
    limit: int = DEFAULT_FETCH_LIMIT,
) -> int:
    """Convenience wrapper for callers that just want "do the indexing now".

    ``window_minutes=None`` means index everything (bounded by ``limit``).
    The periodic task passes a rolling window so we re-embed only what's
    changed; the CLI uses ``None`` for a full reindex.
    """
    since: datetime | None = None
    if window_minutes is not None:
        since = datetime.utcnow() - timedelta(minutes=window_minutes)

    indexer = TraceIndexer(embedder=embedder)
    try:
        return await indexer.reindex(since=since, limit=limit)
    finally:
        await indexer.aclose()


def _fmt_score(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"
