"""Bounded, rotation-aware JWKS cache."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx
import jwt

from src.config.settings import AuthSettings


class JWKSUnavailable(RuntimeError):
    """The trusted signing-key set could not be obtained."""


class SigningKeyNotFound(RuntimeError):
    """No trusted key matches the token's key identifier."""


class JWKSClient:
    def __init__(self, config: AuthSettings, client: httpx.AsyncClient | None = None) -> None:
        self._config = config
        self._client = client
        self._keys: dict[str, Any] = {}
        self._expires_at = 0.0
        self._lock = asyncio.Lock()

    async def get_key(self, kid: str) -> Any:
        if not kid:
            raise SigningKeyNotFound("Token key identifier is required")
        now = time.monotonic()
        key = self._keys.get(kid)
        if key is not None and now < self._expires_at:
            return key
        await self._refresh(force=key is None)
        key = self._keys.get(kid)
        if key is None:
            # Unknown kid may mean a just-completed rotation. One forced refresh
            # is intentional; repeated attacker-controlled kids remain serialized.
            await self._refresh(force=True)
            key = self._keys.get(kid)
        if key is None:
            raise SigningKeyNotFound("No trusted signing key matches the token")
        return key

    async def _refresh(self, *, force: bool) -> None:
        async with self._lock:
            if not force and self._keys and time.monotonic() < self._expires_at:
                return
            try:
                if self._client is None:
                    async with httpx.AsyncClient(
                        timeout=self._config.jwks_timeout_seconds
                    ) as client:
                        response = await client.get(str(self._config.jwks_uri))
                else:
                    response = await self._client.get(str(self._config.jwks_uri))
                response.raise_for_status()
                document = response.json()
                raw_keys = document.get("keys") if isinstance(document, dict) else None
                if not isinstance(raw_keys, list) or not raw_keys:
                    raise ValueError("JWKS contains no keys")
                parsed: dict[str, Any] = {}
                for raw in raw_keys:
                    if not isinstance(raw, dict) or not isinstance(raw.get("kid"), str):
                        continue
                    if raw.get("alg") and raw["alg"] not in self._config.algorithms:
                        continue
                    parsed[raw["kid"]] = jwt.PyJWK.from_dict(raw).key
                if not parsed:
                    raise ValueError("JWKS contains no configured signing keys")
            except (httpx.HTTPError, ValueError, KeyError, jwt.PyJWTError) as exc:
                # Never continue indefinitely with expired keys during an outage.
                if self._keys and time.monotonic() < self._expires_at:
                    return
                raise JWKSUnavailable("Trusted signing keys are unavailable") from exc
            self._keys = parsed
            self._expires_at = time.monotonic() + self._config.jwks_cache_ttl_seconds


_client: JWKSClient | None = None


def get_jwks_client(config: AuthSettings) -> JWKSClient:
    global _client
    if _client is None or _client._config != config:
        _client = JWKSClient(config)
    return _client


def reset_jwks_client() -> None:
    global _client
    _client = None
