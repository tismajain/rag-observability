from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routers import documents
from src.auth.principal import Principal
from src.storage.database import get_session_factory, init_engine, session_scope, shutdown_engine
from src.storage.models import Base, DocumentRecord, IngestionJob
from src.storage.object_store import ObjectNotFoundError, set_object_store_for_tests


class MemoryObjectStore:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    async def ensure_bucket(self) -> None:
        return None

    async def put(self, key: str, body: bytes, content_type: str) -> None:
        self.objects[key] = body

    async def get(self, key: str) -> bytes:
        try:
            return self.objects[key]
        except KeyError:
            raise ObjectNotFoundError from None

    async def delete(self, key: str) -> None:
        self.objects.pop(key, None)


class FakeLifecycle:
    def __init__(self, store: MemoryObjectStore, *, fail_ingestion: bool = False) -> None:
        self.store = store
        self.fail_ingestion = fail_ingestion

    async def process_ingestion(self, job_id: str) -> None:
        async with session_scope() as session:
            job = await session.get(IngestionJob, job_id)
            assert job is not None
            job.status = "failed" if self.fail_ingestion else "completed"
            job.error_code = "test_failure" if self.fail_ingestion else None

    async def delete_document(self, document: DocumentRecord, principal: Principal) -> None:
        await self.store.delete(document.storage_key)
        async with session_scope() as session:
            current = await session.get(DocumentRecord, document.id)
            assert current is not None
            current.lifecycle_state = "deleted"

    async def update_access(
        self, document_id: str, owner_user_id: str, shared_user_ids: frozenset[str]
    ) -> None:
        return None


@pytest.fixture
def document_client(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[TestClient, dict[str, Principal], MemoryObjectStore]]:
    import asyncio

    async def setup() -> None:
        await shutdown_engine()
        init_engine("sqlite+aiosqlite:///:memory:")
        engine = get_session_factory().kw["bind"]
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

    asyncio.run(setup())
    store = MemoryObjectStore()
    set_object_store_for_tests(store)
    monkeypatch.setattr(documents, "document_lifecycle", FakeLifecycle(store))
    current = {
        "principal": Principal("user-a", "org-a", frozenset({"user"})),
    }

    async def principal() -> Principal:
        return current["principal"]

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
    with TestClient(app) as client:
        yield client, current, store
    set_object_store_for_tests(None)
    asyncio.run(shutdown_engine())


def _upload(client: TestClient, name: str = "private.md") -> str:
    response = client.post(
        "/documents",
        files={"file": (name, b"private test content", "text/markdown")},
    )
    assert response.status_code == 202, response.text
    return str(response.json()["id"])


def test_cross_user_document_lifecycle_is_non_enumerable(
    document_client: tuple[TestClient, dict[str, Principal], MemoryObjectStore],
) -> None:
    client, current, _ = document_client
    document_id = _upload(client)
    assert client.get("/documents").json()["items"][0]["id"] == document_id

    current["principal"] = Principal("user-b", "org-b", frozenset({"user"}))
    assert client.get("/documents").json()["items"] == []
    assert client.get(f"/documents/{document_id}").status_code == 404
    assert client.get(f"/documents/{document_id}/download").status_code == 404
    assert client.patch(f"/documents/{document_id}", json={"title": "stolen"}).status_code == 404
    assert client.delete(f"/documents/{document_id}").status_code == 404


def test_explicit_share_and_unshare(
    document_client: tuple[TestClient, dict[str, Principal], MemoryObjectStore],
) -> None:
    client, current, _ = document_client
    document_id = _upload(client)
    current["principal"] = Principal("user-b", "org-b", frozenset({"user"}))
    assert client.get("/documents").status_code == 200  # materializes trusted identity

    current["principal"] = Principal("user-a", "org-a", frozenset({"user"}))
    assert (
        client.post(f"/documents/{document_id}/shares", json={"user_id": "user-b"}).status_code
        == 204
    )
    current["principal"] = Principal("user-b", "org-b", frozenset({"user"}))
    assert client.get(f"/documents/{document_id}").status_code == 200
    assert client.get(f"/documents/{document_id}/download").content == b"private test content"

    current["principal"] = Principal("user-a", "org-a", frozenset({"user"}))
    assert client.delete(f"/documents/{document_id}/shares/user-b").status_code == 204
    current["principal"] = Principal("user-b", "org-b", frozenset({"user"}))
    assert client.get(f"/documents/{document_id}").status_code == 404


def test_deletion_removes_object_and_access(
    document_client: tuple[TestClient, dict[str, Principal], MemoryObjectStore],
) -> None:
    client, _, store = document_client
    document_id = _upload(client)
    assert store.objects
    assert client.delete(f"/documents/{document_id}").status_code == 204
    assert not store.objects
    assert client.get(f"/documents/{document_id}").status_code == 404


def test_failed_ingestion_is_reported(
    document_client: tuple[TestClient, dict[str, Principal], MemoryObjectStore],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _, store = document_client
    monkeypatch.setattr(documents, "document_lifecycle", FakeLifecycle(store, fail_ingestion=True))
    document_id = _upload(client, "failure.md")
    response = client.get(f"/documents/{document_id}")
    assert response.status_code == 200
    assert response.json()["ingestion_status"] == "failed"
