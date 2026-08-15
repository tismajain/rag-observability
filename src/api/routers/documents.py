"""Authenticated document lifecycle endpoints with fail-closed authorization."""

from __future__ import annotations

import hashlib
from typing import Literal, cast
from uuid import uuid4

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Response,
    UploadFile,
    status,
)
from sqlalchemy import delete, select

from src.api.schemas import (
    DocumentListResponse,
    DocumentResponse,
    DocumentShareRequest,
    DocumentUpdateRequest,
    ErrorResponse,
    IngestionJobResponse,
)
from src.auth.dependencies import (
    require_document_create,
    require_document_delete,
    require_document_read,
    require_document_share,
    require_document_update,
)
from src.auth.principal import Principal
from src.config.settings import settings
from src.documents.service import document_lifecycle
from src.ingestion.loader import SUPPORTED_EXTENSIONS
from src.storage.database import session_scope
from src.storage.document_store import (
    apply_rls_identity,
    ensure_identity,
    get_accessible,
    list_accessible,
)
from src.storage.models import AppUser, DocumentRecord, DocumentShare, IngestionJob
from src.storage.object_store import ObjectNotFoundError, get_object_store

router = APIRouter(prefix="/documents", tags=["documents"])


def _not_found() -> HTTPException:
    return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")


async def _owned(document_id: str, principal: Principal) -> DocumentRecord:
    async with session_scope() as session:
        await apply_rls_identity(session, principal)
        document = (
            await session.execute(
                select(DocumentRecord).where(
                    DocumentRecord.id == document_id,
                    DocumentRecord.owner_user_id == principal.subject,
                    DocumentRecord.lifecycle_state == "active",
                )
            )
        ).scalar_one_or_none()
        if document is None:
            raise _not_found()
        return document


async def _response(document: DocumentRecord, principal: Principal) -> DocumentResponse:
    async with session_scope() as session:
        await apply_rls_identity(session, principal)
        job_status = (
            await session.execute(
                select(IngestionJob.status)
                .where(IngestionJob.document_id == document.id)
                .order_by(IngestionJob.created_at.desc())
                .limit(1)
            )
        ).scalar_one_or_none() or "failed"
    visibility: Literal["private", "shared"] = (
        "shared" if document.visibility == "shared" else "private"
    )
    ingestion_status = cast(Literal["queued", "processing", "completed", "failed"], job_status)
    return DocumentResponse(
        id=document.id,
        title=document.title,
        content_type=document.content_type,
        byte_size=document.byte_size,
        visibility=visibility,
        ingestion_status=ingestion_status,
        owned_by_requester=document.owner_user_id == principal.subject,
        created_at=document.created_at,
        updated_at=document.updated_at,
    )


@router.post(
    "",
    response_model=DocumentResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses={413: {"model": ErrorResponse}},
)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: str | None = Form(default=None, max_length=512),
    principal: Principal = Depends(require_document_create),
) -> DocumentResponse:
    filename = file.filename or "document"
    if not any(filename.lower().endswith(ext) for ext in SUPPORTED_EXTENSIONS):
        raise HTTPException(status_code=415, detail="Unsupported document type")
    body = await file.read(settings.object_storage.max_upload_bytes + 1)
    if len(body) > settings.object_storage.max_upload_bytes:
        raise HTTPException(status_code=413, detail="Document exceeds the upload limit")
    if not body:
        raise HTTPException(status_code=422, detail="Document is empty")
    document_id = str(uuid4())
    storage_key = f"documents/{document_id}"
    content_type = file.content_type or "application/octet-stream"
    # Validate the trusted database identity before touching object storage.
    async with session_scope() as session:
        await apply_rls_identity(session, principal)
        await ensure_identity(session, principal)
    store = get_object_store()
    await store.ensure_bucket()
    await store.put(storage_key, body, content_type)
    try:
        async with session_scope() as session:
            await apply_rls_identity(session, principal)
            document = DocumentRecord(
                id=document_id,
                organization_id=principal.organization_id,
                owner_user_id=principal.subject,
                visibility="private",
                title=(title or filename).strip(),
                source_filename=filename,
                storage_key=storage_key,
                content_sha256=hashlib.sha256(body).hexdigest(),
                content_type=content_type,
                byte_size=len(body),
            )
            session.add(document)
            job = IngestionJob(
                document_id=document_id,
                organization_id=principal.organization_id,
                requested_by_user_id=principal.subject,
                status="queued",
            )
            session.add(job)
            await session.flush()
            response = DocumentResponse(
                id=document.id,
                title=document.title,
                content_type=document.content_type,
                byte_size=document.byte_size,
                visibility="private",
                ingestion_status="queued",
                owned_by_requester=True,
                created_at=document.created_at,
                updated_at=document.updated_at,
            )
            job_id = job.id
    except Exception:
        await store.delete(storage_key)
        raise
    background_tasks.add_task(document_lifecycle.process_ingestion, job_id)
    return response


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    principal: Principal = Depends(require_document_read),
) -> DocumentListResponse:
    async with session_scope() as session:
        await apply_rls_identity(session, principal)
        await ensure_identity(session, principal)
        documents = await list_accessible(
            session, principal.authorization_scope(), limit=limit, offset=offset
        )
    return DocumentListResponse(
        items=[await _response(document, principal) for document in documents],
        limit=limit,
        offset=offset,
    )


@router.get("/{document_id}", response_model=DocumentResponse)
async def document_metadata(
    document_id: str, principal: Principal = Depends(require_document_read)
) -> DocumentResponse:
    async with session_scope() as session:
        await apply_rls_identity(session, principal)
        document = await get_accessible(session, document_id, principal.authorization_scope())
        if document is None:
            raise _not_found()
    return await _response(document, principal)


@router.get("/{document_id}/download")
async def download_document(
    document_id: str, principal: Principal = Depends(require_document_read)
) -> Response:
    async with session_scope() as session:
        await apply_rls_identity(session, principal)
        document = await get_accessible(session, document_id, principal.authorization_scope())
        if document is None:
            raise _not_found()
    try:
        body = await get_object_store().get(document.storage_key)
    except ObjectNotFoundError:
        raise _not_found() from None
    return Response(
        content=body,
        media_type=document.content_type,
        headers={"Content-Disposition": f'attachment; filename="{document.id}"'},
    )


@router.patch("/{document_id}", response_model=DocumentResponse)
async def update_document(
    document_id: str,
    payload: DocumentUpdateRequest,
    principal: Principal = Depends(require_document_update),
) -> DocumentResponse:
    await _owned(document_id, principal)
    async with session_scope() as session:
        await apply_rls_identity(session, principal)
        document = await session.get(DocumentRecord, document_id)
        if document is None:
            raise _not_found()
        document.title = payload.title.strip()
    return await _response(document, principal)


@router.post("/{document_id}/shares", status_code=status.HTTP_204_NO_CONTENT)
async def share_document(
    document_id: str,
    payload: DocumentShareRequest,
    principal: Principal = Depends(require_document_share),
) -> Response:
    document = await _owned(document_id, principal)
    async with session_scope() as session:
        await apply_rls_identity(session, principal)
        target = await session.get(AppUser, payload.user_id)
        if target is None or not target.active or target.subject == principal.subject:
            raise HTTPException(status_code=404, detail="Share target not found")
        existing = (
            await session.execute(
                select(DocumentShare).where(
                    DocumentShare.document_id == document_id,
                    DocumentShare.user_id == target.subject,
                )
            )
        ).scalar_one_or_none()
        if existing is None:
            session.add(
                DocumentShare(
                    document_id=document_id,
                    user_id=target.subject,
                    created_by_user_id=principal.subject,
                )
            )
        current = await session.get(DocumentRecord, document_id)
        if current is not None:
            current.visibility = "shared"
        await session.flush()
        share_ids = frozenset(
            (
                await session.execute(
                    select(DocumentShare.user_id).where(DocumentShare.document_id == document_id)
                )
            ).scalars()
        )
    await document_lifecycle.update_access(document_id, document.owner_user_id, share_ids)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.delete("/{document_id}/shares/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def unshare_document(
    document_id: str,
    user_id: str,
    principal: Principal = Depends(require_document_share),
) -> Response:
    document = await _owned(document_id, principal)
    async with session_scope() as session:
        await apply_rls_identity(session, principal)
        share_ids = frozenset(
            value
            for value in (
                await session.execute(
                    select(DocumentShare.user_id).where(DocumentShare.document_id == document_id)
                )
            ).scalars()
            if value != user_id
        )
    await document_lifecycle.update_access(document_id, document.owner_user_id, share_ids)
    async with session_scope() as session:
        await apply_rls_identity(session, principal)
        await session.execute(
            delete(DocumentShare).where(
                DocumentShare.document_id == document_id, DocumentShare.user_id == user_id
            )
        )
        current = await session.get(DocumentRecord, document_id)
        if current is not None and not share_ids:
            current.visibility = "private"
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/{document_id}/jobs/{job_id}", response_model=IngestionJobResponse)
async def ingestion_status(
    document_id: str,
    job_id: str,
    principal: Principal = Depends(require_document_read),
) -> IngestionJobResponse:
    async with session_scope() as session:
        await apply_rls_identity(session, principal)
        document = await get_accessible(session, document_id, principal.authorization_scope())
        if document is None:
            raise _not_found()
        job = await session.get(IngestionJob, job_id)
        if job is None or job.document_id != document.id:
            raise _not_found()
        return IngestionJobResponse(
            id=job.id,
            document_id=document.id,
            status=cast(Literal["queued", "processing", "completed", "failed"], job.status),
            error_code=job.error_code,
        )


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: str, principal: Principal = Depends(require_document_delete)
) -> Response:
    document = await _owned(document_id, principal)
    await document_lifecycle.delete_document(document, principal)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
