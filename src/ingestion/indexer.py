"""Qdrant indexing layer.

Creates and writes to the primary chunk collection. Idempotent: re-ingesting
the same source file produces the same deterministic point IDs (UUIDv5 derived
from ``source_file:chunk_index:content_hash``), so upserts overwrite instead of
duplicating.

Phase 2 ships:

* :func:`init_qdrant` — startup-time collection bootstrap (called by the API
  lifespan and by the ingest CLI).
* :class:`QdrantIndexer` — the upsert pipeline used during ingestion.
"""

from __future__ import annotations

import hashlib
import time
import uuid
from collections.abc import Iterable

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

from src.auth.principal import AuthorizationScope
from src.config.settings import settings
from src.ingestion.chunker import Chunk
from src.observability.tracer import get_tracer

log = structlog.get_logger(__name__)

DEFAULT_UPSERT_BATCH = 100
PROGRESS_LOG_EVERY = 1000

# Namespace UUID for deterministic point IDs. Random one-time value — never
# change it or every ingested doc gets new IDs on next ingest.
_POINT_NAMESPACE = uuid.UUID("8e3b8b1a-6f3a-4d6a-9e2d-2f4a3a6b3c01")


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _point_id(source_file: str, chunk_index: int, text: str) -> str:
    name = f"{source_file}:{chunk_index}:{_content_hash(text)}"
    return str(uuid.uuid5(_POINT_NAMESPACE, name))


def _build_client() -> AsyncQdrantClient:
    return AsyncQdrantClient(
        host=settings.qdrant.host,
        port=settings.qdrant.port,
        prefer_grpc=False,
    )


async def init_qdrant() -> None:
    """Ensure the primary collection exists with the expected schema.

    Safe to call at every process startup — no-op if the collection is already
    present and correctly configured.
    """
    client = _build_client()
    try:
        existing = {c.name for c in (await client.get_collections()).collections}
        name = settings.qdrant.collection_name
        if name in existing:
            log.info("ingestion.indexer.collection_exists", collection=name)
            return

        await client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(
                size=settings.qdrant.vector_size,
                distance=Distance.COSINE,
            ),
            hnsw_config=HnswConfigDiff(m=16, ef_construct=200),
        )
        # Payload indexes for filtered retrieval (by source + ingestion date).
        await client.create_payload_index(name, "source_file", PayloadSchemaType.KEYWORD)
        await client.create_payload_index(name, "ingested_at", PayloadSchemaType.DATETIME)
        await client.create_payload_index(name, "document_id", PayloadSchemaType.KEYWORD)
        await client.create_payload_index(name, "owner_user_id", PayloadSchemaType.KEYWORD)
        await client.create_payload_index(name, "organization_id", PayloadSchemaType.KEYWORD)
        await client.create_payload_index(name, "access_subjects", PayloadSchemaType.KEYWORD)
        await client.create_payload_index(name, "authorization_ready", PayloadSchemaType.BOOL)
        log.info(
            "ingestion.indexer.collection_created",
            collection=name,
            vector_size=settings.qdrant.vector_size,
            distance="cosine",
            hnsw_m=16,
            hnsw_ef_construct=200,
        )
    finally:
        await client.close()


class QdrantIndexer:
    """Writes chunks + embeddings into the configured Qdrant collection."""

    def __init__(self, batch_size: int = DEFAULT_UPSERT_BATCH) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self._batch_size = batch_size
        self._client = _build_client()
        self._collection = settings.qdrant.collection_name
        self._tracer = get_tracer(__name__)

    async def ensure_ready(self) -> None:
        """Bootstrap the collection if a caller forgot to run :func:`init_qdrant`."""
        try:
            await self._client.get_collection(self._collection)
        except (UnexpectedResponse, ValueError):
            await init_qdrant()

    async def index(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        *,
        scope: AuthorizationScope,
        document_id: str,
        visibility: str = "private",
        shared_user_ids: frozenset[str] = frozenset(),
    ) -> int:
        """Upsert ``(chunk, embedding)`` pairs. Returns the number of points written."""
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks/embeddings length mismatch: {len(chunks)} vs {len(embeddings)}"
            )
        if not chunks:
            return 0

        await self.ensure_ready()
        if visibility not in {"private", "organization", "shared"}:
            raise ValueError("Invalid document visibility")
        points = list(
            self._build_points(
                chunks,
                embeddings,
                scope=scope,
                document_id=document_id,
                visibility=visibility,
                shared_user_ids=shared_user_ids,
            )
        )
        written = 0

        with self._tracer.start_as_current_span("rag.indexing.upsert") as span:
            span.set_attribute("rag.indexing.collection", self._collection)
            span.set_attribute("rag.indexing.total_points", len(points))

            t0 = time.perf_counter()
            for start in range(0, len(points), self._batch_size):
                batch = points[start : start + self._batch_size]
                await self._client.upsert(
                    collection_name=self._collection,
                    points=batch,
                    wait=True,
                )
                written += len(batch)
                if written % PROGRESS_LOG_EVERY == 0:
                    log.info(
                        "ingestion.indexer.progress",
                        written=written,
                        total=len(points),
                    )
            latency_ms = int((time.perf_counter() - t0) * 1000)
            span.set_attribute("rag.indexing.latency_ms", latency_ms)

        log.info(
            "ingestion.indexer.upserted",
            collection=self._collection,
            written=written,
            batches=(len(points) + self._batch_size - 1) // self._batch_size,
            latency_ms=latency_ms,
        )
        return written

    def _build_points(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        *,
        scope: AuthorizationScope,
        document_id: str,
        visibility: str,
        shared_user_ids: frozenset[str],
    ) -> Iterable[PointStruct]:
        for chunk, vector in zip(chunks, embeddings, strict=True):
            md = chunk.metadata
            access_subjects = [f"user:{scope.user_id}"]
            if visibility == "organization":
                access_subjects.append(f"org:{scope.organization_id}")
            if visibility == "shared":
                access_subjects.extend(f"user:{user_id}" for user_id in sorted(shared_user_ids))
            yield PointStruct(
                id=_point_id(document_id, md.chunk_index, chunk.text),
                vector=vector,
                payload={
                    "text": chunk.text,
                    "source_file": md.source_file,
                    "chunk_index": md.chunk_index,
                    "total_chunks": md.total_chunks,
                    "chunking_strategy": md.chunking_strategy,
                    "char_start": md.char_start,
                    "char_end": md.char_end,
                    "token_count": md.token_count,
                    "ingested_at": md.ingested_at.isoformat(),
                    "document_id": document_id,
                    "owner_user_id": scope.user_id,
                    "organization_id": scope.organization_id,
                    "visibility": visibility,
                    "access_subjects": access_subjects,
                    "authorization_ready": True,
                },
            )

    async def aclose(self) -> None:
        await self._client.close()
