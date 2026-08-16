"""Document access predicates; never accepts client-supplied ownership scope."""

from __future__ import annotations

from sqlalchemy import and_, exists, or_, select, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.elements import ColumnElement

from src.auth.principal import AuthorizationScope, Principal
from src.storage.models import AppUser, DocumentRecord, DocumentShare, Organization


def access_predicate(scope: AuthorizationScope) -> ColumnElement[bool]:
    if scope.can_access_all_documents:
        return (DocumentRecord.id.is_not(None)) & (DocumentRecord.lifecycle_state == "active")
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
        and_(DocumentRecord.visibility == "shared", shared),
    ) & (DocumentRecord.lifecycle_state == "active")


async def ensure_identity(session: AsyncSession, principal: Principal) -> None:
    """Materialize trusted JWT identity without accepting client ownership fields."""
    organization = await session.get(Organization, principal.organization_id)
    if organization is None:
        session.add(
            Organization(id=principal.organization_id, display_name=principal.organization_id)
        )
        await session.flush()
    user = await session.get(AppUser, principal.subject)
    if user is None:
        session.add(AppUser(subject=principal.subject, organization_id=principal.organization_id))
    elif user.organization_id != principal.organization_id or not user.active:
        raise PermissionError("Trusted identity is not active in this organization")
    await session.flush()


async def authorization_scope(session: AsyncSession, principal: Principal) -> AuthorizationScope:
    """Build retrieval scope from verified identity and trusted share rows."""
    shared_ids = frozenset(
        (
            await session.execute(
                select(DocumentShare.document_id).where(DocumentShare.user_id == principal.subject)
            )
        ).scalars()
    )
    return principal.authorization_scope(shared_ids)


async def apply_rls_identity(session: AsyncSession, principal: Principal) -> None:
    """Set transaction-local PostgreSQL RLS identity from a verified principal."""
    bind = session.get_bind()
    if bind.dialect.name != "postgresql":
        return
    await session.execute(
        text("SELECT set_config('app.user_id', :user_id, true)"),
        {"user_id": principal.subject},
    )
    await session.execute(
        text("SELECT set_config('app.organization_id', :organization_id, true)"),
        {"organization_id": principal.organization_id},
    )
    await session.execute(
        text("SELECT set_config('app.document_content_admin', :allowed, true)"),
        {
            "allowed": "true"
            if principal.authorization_scope().can_access_all_documents
            else "false"
        },
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
