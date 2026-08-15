"""Document access predicates; never accepts client-supplied ownership scope."""

from __future__ import annotations

from sqlalchemy import and_, exists, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.elements import ColumnElement

from src.auth.principal import AuthorizationScope
from src.storage.models import DocumentRecord, DocumentShare


def access_predicate(scope: AuthorizationScope) -> ColumnElement[bool]:
    if scope.can_access_all_documents:
        return DocumentRecord.id.is_not(None)
    shared = exists(
        select(DocumentShare.id).where(
            and_(
                DocumentShare.document_id == DocumentRecord.id,
                DocumentShare.user_id == scope.user_id,
            )
        )
    )
    return or_(
        DocumentRecord.owner_user_id == scope.user_id,
        and_(
            DocumentRecord.organization_id == scope.organization_id,
            DocumentRecord.visibility == "organization",
        ),
        and_(DocumentRecord.visibility == "shared", shared),
    )


async def get_accessible(
    session: AsyncSession, document_id: str, scope: AuthorizationScope
) -> DocumentRecord | None:
    """Return None for both nonexistent and unauthorized IDs (non-enumerable)."""
    stmt = select(DocumentRecord).where(
        DocumentRecord.id == document_id,
        access_predicate(scope),
    )
    return (await session.execute(stmt)).scalar_one_or_none()


async def list_accessible(
    session: AsyncSession,
    scope: AuthorizationScope,
    *,
    limit: int = 50,
    offset: int = 0,
) -> list[DocumentRecord]:
    stmt = (
        select(DocumentRecord)
        .where(access_predicate(scope))
        .order_by(DocumentRecord.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    return list((await session.execute(stmt)).scalars().all())
