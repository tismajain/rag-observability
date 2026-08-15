"""Typed identities and application permissions derived from trusted tokens."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import StrEnum


class Permission(StrEnum):
    QUERY = "query:execute"
    OBSERVABILITY_READ = "observability:read"
    OBSERVABILITY_ADMIN = "observability:admin"
    DOCUMENT_CREATE = "document:create"
    DOCUMENT_READ = "document:read"
    DOCUMENT_UPDATE = "document:update"
    DOCUMENT_DELETE = "document:delete"
    DOCUMENT_SHARE = "document:share"
    DOCUMENT_CONTENT_ADMIN = "document:content:admin"


ROLE_PERMISSIONS: dict[str, frozenset[Permission]] = {
    "user": frozenset(
        {
            Permission.QUERY,
            Permission.DOCUMENT_CREATE,
            Permission.DOCUMENT_READ,
            Permission.DOCUMENT_UPDATE,
            Permission.DOCUMENT_DELETE,
            Permission.DOCUMENT_SHARE,
        }
    ),
    "observer": frozenset({Permission.QUERY, Permission.OBSERVABILITY_READ}),
    "observability_admin": frozenset(
        {Permission.QUERY, Permission.OBSERVABILITY_READ, Permission.OBSERVABILITY_ADMIN}
    ),
    "document_admin": frozenset(
        {
            Permission.QUERY,
            Permission.DOCUMENT_READ,
            Permission.DOCUMENT_UPDATE,
            Permission.DOCUMENT_DELETE,
            Permission.DOCUMENT_SHARE,
            Permission.DOCUMENT_CONTENT_ADMIN,
        }
    ),
}


@dataclass(frozen=True, slots=True)
class Principal:
    subject: str
    organization_id: str
    roles: frozenset[str]
    explicit_permissions: frozenset[str] = frozenset()
    authenticated: bool = True

    @property
    def permissions(self) -> frozenset[str]:
        inherited = {
            permission.value
            for role in self.roles
            for permission in ROLE_PERMISSIONS.get(role, frozenset())
        }
        return frozenset(inherited | set(self.explicit_permissions))

    def has(self, permission: Permission | str) -> bool:
        value = permission.value if isinstance(permission, Permission) else permission
        return value in self.permissions

    def authorization_scope(
        self, shared_document_ids: frozenset[str] = frozenset()
    ) -> AuthorizationScope:
        return AuthorizationScope(
            user_id=self.subject,
            organization_id=self.organization_id,
            shared_document_ids=shared_document_ids,
            can_access_all_documents=self.has(Permission.DOCUMENT_CONTENT_ADMIN),
        )


@dataclass(frozen=True, slots=True)
class AuthorizationScope:
    user_id: str
    organization_id: str
    shared_document_ids: frozenset[str] = frozenset()
    can_access_all_documents: bool = False

    @property
    def fingerprint(self) -> str:
        raw = "\x1f".join(
            [
                self.user_id,
                self.organization_id,
                "1" if self.can_access_all_documents else "0",
                *sorted(self.shared_document_ids),
            ]
        )
        return hashlib.sha256(raw.encode()).hexdigest()[:24]
