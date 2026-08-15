"""Conservative secret redaction for log and error boundaries."""

from __future__ import annotations

import re
from typing import Any

_BEARER = re.compile(r"(?i)bearer\s+[A-Za-z0-9._~+/=-]+")
_JWT = re.compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b")
_SECRET_KEYS = {
    "authorization",
    "access_token",
    "id_token",
    "cookie",
    "set-cookie",
    "secret",
    "password",
}


def redact_text(value: object) -> str:
    text = str(value)
    return _JWT.sub("[REDACTED_TOKEN]", _BEARER.sub("Bearer [REDACTED]", text))


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(k): "[REDACTED]" if str(k).lower() in _SECRET_KEYS else redact(v)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [redact(v) for v in value]
    if isinstance(value, tuple):
        return tuple(redact(v) for v in value)
    if isinstance(value, str):
        return redact_text(value)
    return value
