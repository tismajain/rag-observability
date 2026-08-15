"""OIDC, RBAC, rotation, redaction, and pre-ranking isolation tests."""

from __future__ import annotations

import base64
import json
from datetime import UTC, datetime, timedelta

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import HTTPException
from starlette.requests import Request

from src.auth.dependencies import get_principal
from src.auth.jwks import JWKSClient, JWKSUnavailable
from src.auth.principal import AuthorizationScope, Permission, Principal
from src.config.settings import AuthSettings, Settings
from src.security.redaction import redact, redact_text


def _b64int(value: int) -> str:
    width = (value.bit_length() + 7) // 8
    return base64.urlsafe_b64encode(value.to_bytes(width, "big")).rstrip(b"=").decode()


def _keypair(kid: str) -> tuple[object, dict[str, str]]:
    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    numbers = private.public_key().public_numbers()
    return private, {
        "kty": "RSA",
        "use": "sig",
        "alg": "RS256",
        "kid": kid,
        "n": _b64int(numbers.n),
        "e": _b64int(numbers.e),
    }


def _token(private: object, kid: str, **overrides: object) -> str:
    now = datetime.now(UTC)
    claims: dict[str, object] = {
        "iss": "https://issuer.example/",
        "aud": "rag-api",
        "sub": "user-a",
        "org_id": "org-a",
        "roles": ["user"],
        "iat": now,
        "exp": now + timedelta(minutes=5),
    }
    claims.update(overrides)
    return jwt.encode(claims, private, algorithm="RS256", headers={"kid": kid})


@pytest.fixture
def auth_config() -> AuthSettings:
    return AuthSettings(
        enabled=True,
        issuer="https://issuer.example/",
        audience="rag-api",
        jwks_uri="https://issuer.example/jwks",
        algorithms=("RS256",),
        jwks_cache_ttl_seconds=30,
    )


def test_disabled_auth_is_not_admin() -> None:
    principal = Principal("local", "local", frozenset({"user", "observer"}), authenticated=False)
    assert not principal.has(Permission.OBSERVABILITY_ADMIN)
    assert not principal.has(Permission.DOCUMENT_CONTENT_ADMIN)


def test_observability_admin_has_no_document_content_access() -> None:
    principal = Principal("admin", "org-a", frozenset({"observability_admin"}))
    assert principal.has(Permission.OBSERVABILITY_ADMIN)
    assert not principal.authorization_scope().can_access_all_documents


def test_scope_fingerprint_is_tenant_and_share_aware() -> None:
    a = AuthorizationScope("u1", "o1")
    assert a.fingerprint != AuthorizationScope("u2", "o1").fingerprint
    assert a.fingerprint != AuthorizationScope("u1", "o2").fingerprint
    assert a.fingerprint != AuthorizationScope("u1", "o1", frozenset({"doc"})).fingerprint


async def test_jwks_unknown_kid_refreshes_for_rotation(auth_config: AuthSettings) -> None:
    _, jwk1 = _keypair("old")
    _, jwk2 = _keypair("new")
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"keys": [jwk1] if calls == 1 else [jwk1, jwk2]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        cache = JWKSClient(auth_config, client)
        await cache.get_key("old")
        await cache.get_key("new")
    assert calls >= 2


async def test_jwks_failure_has_safe_message(auth_config: AuthSettings) -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(503, text="secret upstream body"))
    ) as client:
        with pytest.raises(JWKSUnavailable, match="Trusted signing keys are unavailable"):
            await JWKSClient(auth_config, client).get_key("unknown")


async def test_verified_token_builds_principal(
    monkeypatch: pytest.MonkeyPatch, auth_config: AuthSettings
) -> None:
    private, public = _keypair("key-1")
    verified_key = jwt.PyJWK.from_dict(public).key

    class StubJWKSClient:
        async def get_key(self, kid: str) -> object:
            assert kid == "key-1"
            return verified_key

    monkeypatch.setattr("src.auth.dependencies.get_jwks_client", lambda _config: StubJWKSClient())
    from src.config.settings import settings

    monkeypatch.setattr(settings, "auth", auth_config)
    token = _token(private, "key-1")
    request = Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/",
            "headers": [(b"authorization", f"Bearer {token}".encode())],
        }
    )
    principal = await get_principal(request)
    assert principal.subject == "user-a"
    assert principal.organization_id == "org-a"


async def test_unconfigured_algorithm_is_rejected(
    monkeypatch: pytest.MonkeyPatch, auth_config: AuthSettings
) -> None:
    from src.config.settings import settings

    monkeypatch.setattr(settings, "auth", auth_config)
    header = (
        base64.urlsafe_b64encode(json.dumps({"alg": "none", "kid": "x"}).encode())
        .rstrip(b"=")
        .decode()
    )
    token = f"{header}.e30."
    request = Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/",
            "headers": [(b"authorization", f"Bearer {token}".encode())],
        }
    )
    with pytest.raises(HTTPException) as caught:
        await get_principal(request)
    assert caught.value.status_code == 401
    assert token not in str(caught.value.detail)


def test_redaction_removes_tokens_and_secret_fields() -> None:
    token = "eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiJ1In0.signature"
    assert token not in redact_text(f"Bearer {token}")
    assert redact({"access_token": token, "nested": token}) == {
        "access_token": "[REDACTED]",
        "nested": "[REDACTED_TOKEN]",
    }


def test_production_requires_auth_and_redis(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    with pytest.raises(ValueError, match="Authentication must be enabled"):
        Settings(environment="production")
