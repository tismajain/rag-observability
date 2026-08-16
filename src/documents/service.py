"""Authorized document ingestion, sharing, and deletion orchestration."""

from __future__ import annotations

from datetime import UTC, datetime

import structlog
from sqlalchemy import delete, select

from src.auth.principal import AuthorizationScope, Principal
from src.config.settings import settings
from src.ingestion.chunker import RecursiveChunker
from src.ingestion.embedder import Embedder, MockEmbedder
from src.ingestion.indexer import QdrantIndexer
from src.ingestion.loader import load_bytes
from src.pipeline.pipeline_runner import pipeline_runner
from src.storage.database import session_scope
from src.storage.document_store import apply_rls_identity
from src.storage.models import DocumentChunk, DocumentRecord, DocumentShare, IngestionJob
from src.storage.object_store import get_object_store

log = structlog.get_logger(__name__)


class DocumentLifecycleService:
    async def process_ingestion(self, job_id: str) -> None:
        """Process one durable queued row; partial vectors remain non-searchable."""
        indexer = QdrantIndexer()
        embedder = MockEmbedder() if settings.llm.provider == "mock" else Embedder()
        document_id: str | None = None
        try:
            async with session_scope() as session:
                job = await session.get(IngestionJob, job_id)
                if job is None or job.status != "queued":
                    return
                document = await session.get(DocumentRecord, job.document_id)
                if document is None or document.lifecycle_state != "active":
                    job.status = "failed"
                    job.error_code = "document_unavailable"
                    return
                job.status = "processing"
                document_id = document.id
                storage_key = document.storage_key
                source_filename = document.source_filename
                scope = AuthorizationScope(document.owner_user_id, document.organization_id)
                shares = frozenset(
                    (
                        await session.execute(
                            select(DocumentShare.user_id).where(
                                DocumentShare.document_id == document.id
                            )
                        )
                    ).scalars()
                )
                visibility = document.visibility

            body = await get_object_store().get(storage_key)
            loaded = load_bytes(source_filename, body)
            chunks = await RecursiveChunker().chunk(loaded)
            embeddings = await embedder.embed_texts([chunk.text for chunk in chunks])
            await indexer.index(
                chunks,
                embeddings,
                scope=scope,
                document_id=document_id,
                visibility=visibility,
                shared_user_ids=shares,
            )
            async with session_scope() as session:
                job = await session.get(IngestionJob, job_id)
                document = await session.get(DocumentRecord, document_id)
                if job is None or document is None or document.lifecycle_state != "active":
                    raise RuntimeError("Document was removed during ingestion")
                session.add_all(
                    DocumentChunk(
                        document_id=document_id,
                        organization_id=scope.organization_id,
                        owner_user_id=scope.user_id,
                        qdrant_point_id=str(point_id),
                        chunk_index=chunk.metadata.chunk_index,
                    )
                    for point_id, chunk in zip(
                        indexer.point_ids(chunks, document_id), chunks, strict=True
                    )
                )
                job.status = "completed"
                job.error_code = None
            await indexer.set_searchable(document_id, True)
            pipeline_runner.invalidate_document_indexes()
        except Exception as exc:  # noqa: BLE001 - job boundary must record failure
            if document_id is not None:
                try:
                    await indexer.delete_document(document_id)
                except Exception as cleanup_exc:  # noqa: BLE001
                    log.error(
                        "document.ingestion.cleanup_failed",
                        document_id=document_id,
                        error_type=type(cleanup_exc).__name__,
                    )
            async with session_scope() as session:
                job = await session.get(IngestionJob, job_id)
                if job is not None:
                    job.status = "failed"
                    job.error_code = type(exc).__name__[:64]
                if document_id is not None:
                    await session.execute(
                        delete(DocumentChunk).where(DocumentChunk.document_id == document_id)
                    )
            log.warning(
                "document.ingestion.failed",
                job_id=job_id,
                error_type=type(exc).__name__,
            )
        finally:
            await indexer.aclose()
            await embedder.aclose()

    async def delete_document(self, document: DocumentRecord, principal: Principal) -> None:
        """Make content non-searchable before removing any durable copy."""
        indexer = QdrantIndexer()
        try:
            await indexer.set_searchable(document.id, False)
            async with session_scope() as session:
                await apply_rls_identity(session, principal)
                current = await session.get(DocumentRecord, document.id)
                if current is None:
                    return
                current.lifecycle_state = "deleting"
                current.deleted_at = datetime.now(UTC)
            await indexer.delete_document(document.id)
            await get_object_store().delete(document.storage_key)
            async with session_scope() as session:
                await apply_rls_identity(session, principal)
                await session.execute(
                    delete(DocumentChunk).where(DocumentChunk.document_id == document.id)
                )
                current = await session.get(DocumentRecord, document.id)
                if current is not None:
                    current.lifecycle_state = "deleted"
            pipeline_runner.invalidate_document_indexes()
        finally:
            await indexer.aclose()

    async def update_access(
        self, document_id: str, owner_user_id: str, shared_user_ids: frozenset[str]
    ) -> None:
        indexer = QdrantIndexer()
        try:
            await indexer.update_access(document_id, owner_user_id, shared_user_ids)
            pipeline_runner.invalidate_document_indexes()
        finally:
            await indexer.aclose()


document_lifecycle = DocumentLifecycleService()
