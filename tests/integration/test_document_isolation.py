"""Real-service document isolation tests, enabled with RUN_INTEGRATION=1."""

from __future__ import annotations

import os
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct
from redis.asyncio import Redis

from src.api.routers import documents
from src.auth.principal import AuthorizationScope, Principal
from src.config.settings import settings
from src.ingestion.embedder import MockEmbedder
from src.retrieval.retriever import HybridRetriever
from src.storage.database import init_engine, shutdown_engine

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.getenv("RUN_INTEGRATION") != "1", reason="real services not enabled"),
]


@pytest.mark.asyncio
async def test_real_service_isolation_share_delete_and_partial_failure() -> None:
    init_engine()
    redis = Redis.from_url(settings.rate_limit.redis_url.get_secret_value())  # type: ignore[union-attr]
    assert await redis.ping()
    await redis.aclose()

    suffix = uuid4().hex
    principals = {
        "a": Principal(f"integration-a-{suffix}", f"org-a-{suffix}", frozenset({"user"})),
        "b": Principal(f"integration-b-{suffix}", f"org-b-{suffix}", frozenset({"user"})),
    }
    selected = {"value": principals["a"]}

    async def principal() -> Principal:
        return selected["value"]

    app = FastAPI()
    for dependency in (
        documents.require_document_create,
        documents.require_document_read,
        documents.require_document_update,
        documents.require_document_delete,
        documents.require_document_share,
    ):
        app.dependency_overrides[dependency] = principal
    app.include_router(documents.router)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        created = await client.post(
            "/documents",
            files={"file": ("isolation.md", b"integration isolation sentinel", "text/markdown")},
        )
        assert created.status_code == 202, created.text
        document_id = created.json()["id"]

        selected["value"] = principals["b"]
        assert (await client.get("/documents")).json()["items"] == []
        assert (await client.get(f"/documents/{document_id}")).status_code == 404
        assert (await client.get(f"/documents/{document_id}/download")).status_code == 404
        assert (
            await client.patch(f"/documents/{document_id}", json={"title": "forbidden"})
        ).status_code == 404
        assert (await client.delete(f"/documents/{document_id}")).status_code == 404

        denied_retriever = HybridRetriever(MockEmbedder())
        try:
            denied_scope = AuthorizationScope(
                principals["b"].subject, principals["b"].organization_id
            )
            assert all(
                hit.metadata.document_id != document_id
                for hit in await denied_retriever.retrieve(
                    "integration isolation sentinel", scope=denied_scope
                )
            )
        finally:
            await denied_retriever.aclose()

        selected["value"] = principals["a"]
        assert (
            await client.post(
                f"/documents/{document_id}/shares", json={"user_id": principals["b"].subject}
            )
        ).status_code == 204
        selected["value"] = principals["b"]
        assert (await client.get(f"/documents/{document_id}")).status_code == 200
        assert (await client.get(f"/documents/{document_id}/download")).status_code == 200

        retriever = HybridRetriever(MockEmbedder())
        try:
            shared_scope = AuthorizationScope(
                principals["b"].subject,
                principals["b"].organization_id,
                frozenset({document_id}),
            )
            assert any(
                hit.metadata.document_id == document_id
                for hit in await retriever.retrieve(
                    "integration isolation sentinel", scope=shared_scope
                )
            )

            qdrant = AsyncQdrantClient(host=settings.qdrant.host, port=settings.qdrant.port)
            legacy_id = str(uuid4())
            await qdrant.upsert(
                settings.qdrant.collection_name,
                [
                    PointStruct(
                        id=legacy_id,
                        vector=[0.0] * settings.qdrant.vector_size,
                        payload={"text": "legacy secret without authorization scope"},
                    )
                ],
                wait=True,
            )
            failed_id = str(uuid4())
            await qdrant.upsert(
                settings.qdrant.collection_name,
                [
                    PointStruct(
                        id=failed_id,
                        vector=[0.0] * settings.qdrant.vector_size,
                        payload={
                            "text": "failed partial sentinel",
                            "document_id": failed_id,
                            "authorization_ready": True,
                            "searchable": False,
                            "access_subjects": [f"user:{principals['b'].subject}"],
                        },
                    )
                ],
                wait=True,
            )
            retriever.invalidate_sparse_indexes()
            hits = await retriever.retrieve("legacy failed partial sentinel", scope=shared_scope)
            assert all(hit.chunk_id not in {legacy_id, failed_id} for hit in hits)
            await qdrant.delete(settings.qdrant.collection_name, [legacy_id, failed_id], wait=True)
            await qdrant.close()
        finally:
            await retriever.aclose()

        failed = await client.post(
            "/documents",
            files={"file": ("broken.pdf", b"not a valid pdf", "application/pdf")},
        )
        assert failed.status_code == 202
        failed_id = failed.json()["id"]
        failed_metadata = await client.get(f"/documents/{failed_id}")
        assert failed_metadata.json()["ingestion_status"] == "failed"
        failed_retriever = HybridRetriever(MockEmbedder())
        try:
            assert all(
                hit.metadata.document_id != failed_id
                for hit in await failed_retriever.retrieve(
                    "not a valid pdf",
                    scope=AuthorizationScope(
                        principals["b"].subject, principals["b"].organization_id
                    ),
                )
            )
        finally:
            await failed_retriever.aclose()

        selected["value"] = principals["a"]
        assert (await client.delete(f"/documents/{document_id}")).status_code == 204
        assert (await client.get(f"/documents/{document_id}")).status_code == 404
        retriever = HybridRetriever(MockEmbedder())
        try:
            owner_scope = AuthorizationScope(
                principals["a"].subject, principals["a"].organization_id
            )
            assert all(
                hit.metadata.document_id != document_id
                for hit in await retriever.retrieve(
                    "integration isolation sentinel", scope=owner_scope
                )
            )
        finally:
            await retriever.aclose()
    await shutdown_engine()
