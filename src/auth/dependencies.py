"""FastAPI authentication and permission dependencies."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

import jwt
import structlog
from fastapi import Depends, HTTPException, Request, status

from src.auth.jwks import JWKSUnavailable, SigningKeyNotFound, get_jwks_client
from src.auth.principal import Permission, Principal
from src.config.settings import settings

log = structlog.get_logger(__name__)


def _claim_strings(claims: dict[str, Any], name: str) -> frozenset[str]:
    value = claims.get(name, [])
    if isinstance(value, str):
        return frozenset({value})
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return frozenset(value)
    return frozenset()


async def get_principal(request: Request) -> Principal:
    config = settings.auth
    if not config.enabled:
        # Local mode is deliberately a regular user, never an admin.
        principal = Principal(
            subject=config.disabled_local_user_id,
            organization_id=config.disabled_local_organization_id,
            roles=frozenset({"user", "observer"}),
            authenticated=False,
        )
        request.state.principal = principal
        return principal

    header = request.headers.get("Authorization", "")
    scheme, separator, token = header.partition(" ")
    if not separator or scheme.lower() != "bearer" or not token or " " in token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Valid bearer authentication is required",
        )

    try:
        # Header fields are untrusted. `alg` is checked only for early rejection;
        # decode below receives the independently configured allow-list.
        unverified_header = jwt.get_unverified_header(token)
        kid = unverified_header.get("kid")
        if not isinstance(kid, str) or unverified_header.get("alg") not in config.algorithms:
            raise jwt.InvalidTokenError("Untrusted signing parameters")
        key = await get_jwks_client(config).get_key(kid)
        claims = jwt.decode(
            token,
            key=key,
            algorithms=list(config.algorithms),
            audience=config.audience,
            issuer=config.issuer,
            leeway=config.clock_skew_seconds,
            options={"require": ["exp", "iat", "iss", "aud", "sub"]},
        )
        subject = claims.get("sub")
        organization = claims.get(config.organization_claim)
        if (
            not isinstance(subject, str)
            or not subject
            or not isinstance(organization, str)
            or not organization
        ):
            raise jwt.InvalidTokenError("Required identity claims are missing")
        principal = Principal(
            subject=subject,
            organization_id=organization,
            roles=_claim_strings(claims, config.roles_claim),
            explicit_permissions=_claim_strings(claims, config.permissions_claim),
        )
    except (jwt.PyJWTError, SigningKeyNotFound):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Bearer token is invalid"
        ) from None
    except JWKSUnavailable:
        log.warning("auth.jwks.unavailable")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service is unavailable",
        ) from None
    request.state.principal = principal
    return principal


def require_permission(permission: Permission) -> Callable[..., Awaitable[Principal]]:
    async def dependency(
        principal: Principal = Depends(get_principal),  # noqa: B008
    ) -> Principal:
        if not principal.has(permission):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Permission denied")
        return principal

    return dependency


require_query = require_permission(Permission.QUERY)
require_observability_read = require_permission(Permission.OBSERVABILITY_READ)
require_observability_admin = require_permission(Permission.OBSERVABILITY_ADMIN)
require_document_read = require_permission(Permission.DOCUMENT_READ)
