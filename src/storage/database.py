"""Async SQLAlchemy engine + session factory.

Lifecycle:

* :func:`init_engine` is called once at application startup (FastAPI lifespan
  and tests both call this). It builds a process-wide :class:`AsyncEngine`
  with a sensible pool config.
* :func:`shutdown_engine` disposes the engine cleanly on shutdown.
* :func:`session_scope` is an async context manager that yields a session
  with automatic commit/rollback. Pipeline code uses it; routers use a
  FastAPI dependency that wraps the same context manager.

Failure tolerance: ``init_engine`` itself does not connect — SQLAlchemy
connects lazily on first query. A Postgres outage at boot will not crash
the API; individual queries will fail and be caught in their stores.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src.config.settings import settings

log = structlog.get_logger(__name__)

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def init_engine(url: str | None = None) -> AsyncEngine:
    """Build the process-wide async engine. Idempotent."""
    global _engine, _session_factory
    if _engine is not None:
        return _engine

    db_url = url or settings.database_url
    # SQLite's aiosqlite driver uses StaticPool and rejects pool_size /
    # max_overflow. Detect sqlite so the same init_engine call works for
    # both Postgres (prod) and SQLite (tests).
    is_sqlite = db_url.startswith("sqlite")
    kwargs: dict[str, object] = {"echo": False}
    if not is_sqlite:
        kwargs.update(pool_size=10, max_overflow=20, pool_pre_ping=True)
    _engine = create_async_engine(db_url, **kwargs)
    _session_factory = async_sessionmaker(
        bind=_engine,
        expire_on_commit=False,
        autoflush=False,
        class_=AsyncSession,
    )
    log.info("storage.engine.initialized", url=_safe_url(db_url))
    return _engine


async def shutdown_engine() -> None:
    """Dispose the engine. Idempotent."""
    global _engine, _session_factory
    if _engine is None:
        return
    try:
        await _engine.dispose()
    except Exception as exc:  # noqa: BLE001 — never let shutdown crash the app
        log.warning("storage.engine.shutdown_failed", error=str(exc))
    finally:
        _engine = None
        _session_factory = None


def get_engine() -> AsyncEngine:
    if _engine is None:
        raise RuntimeError("Database engine not initialized — call init_engine() first")
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    if _session_factory is None:
        raise RuntimeError("Session factory not initialized — call init_engine() first")
    return _session_factory


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    """Yield a session, commit on success, rollback on exception."""
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


def _safe_url(url: str) -> str:
    """Redact the password component so it never lands in logs."""
    if "@" not in url or "://" not in url:
        return url
    scheme, rest = url.split("://", 1)
    if "@" not in rest:
        return url
    creds, host_part = rest.rsplit("@", 1)
    if ":" in creds:
        user, _ = creds.split(":", 1)
        return f"{scheme}://{user}:***@{host_part}"
    return url
