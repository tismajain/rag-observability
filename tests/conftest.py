"""Shared pytest fixtures and bootstrap.

Sets dummy required env vars before any ``src.config.settings`` import so the
module-level singleton can be constructed in CI without a populated ``.env``.
A real ``.env.test`` (if present) wins over these defaults.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _bootstrap_env() -> None:
    env_test = ROOT / ".env.test"
    if env_test.exists():
        for line in env_test.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())
    # Fallbacks so settings.py can construct.
    os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
    os.environ.setdefault("ENVIRONMENT", "development")


_bootstrap_env()
