"""Integration-flavored tests for the meta-RAG trace indexer.

These run end-to-end through the *real* SQLAlchemy code paths against an
in-memory SQLite database, with the Qdrant client + embedder mocked. That
gets us confidence in:

* The DB → document → Qdrant payload shape pipeline.
* Deterministic point IDs (re-running reindex produces the same point per
  trace, so writes overwrite cleanly).
* The window filter on ``since``.
* The eager-loading of defects + evals via the ``selectin`` relationship.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src.meta_rag.trace_indexer import TraceIndexer, _meta_point_id, reindex_recent
from src.storage import defect_store, eval_store, trace_store
from src.storage.models import Base


class _StubEmbedder:
    """Returns one-hot-ish vectors keyed by hash — deterministic, no API call."""

    @property
    def model(self) -> str:
        return "stub"

    @property
    def dimensions(self) -> int:
        return 4

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[float(hash(t) % 100) / 100.0] * 4 for t in texts]

    async def aclose(self) -> None:
        return None


@pytest_asyncio.fixture
async def db_session() -> AsyncIterator[AsyncSession]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    async with factory() as s:
        yield s
    await engine.dispose()


async def _seed_trace(
    session: AsyncSession,
    query_id: str,
    *,
    with_defect: bool = False,
    with_eval: bool = False,
) -> None:
    await trace_store.create_trace(
        session,
        query_id=query_id,
        trace_id=f"trace-{query_id}",
        query_text=f"q for {query_id}",
        answer_text=f"a for {query_id}",
        context_truncated=False,
        latency_ms=120,
        top_k=5,
        chunks_retrieved=2,
        model_used="mock-1",
    )
    if with_defect:
        await defect_store.create_defect(
            session,
            query_id=query_id,
            trace_id=f"trace-{query_id}",
            defect_type="DEFECT_LOW_RETRIEVAL_QUALITY",
            severity="HIGH",
            description="low score",
        )
    if with_eval:
        await eval_store.create_eval(
            session,
            query_id=query_id,
            trace_id=f"trace-{query_id}",
            faithfulness=0.91,
            context_recall=0.84,
            answer_relevancy=0.88,
            quality_gate_result="pass",
            evaluation_latency_ms=200,
            judge_model="judge-1",
        )
    await session.commit()


def _make_indexer_with_mocked_qdrant() -> tuple[TraceIndexer, MagicMock]:
    """Build a TraceIndexer whose Qdrant client is fully mocked."""
    indexer = TraceIndexer(embedder=_StubEmbedder(), batch_size=10)
    mock_client = MagicMock()
    mock_client.get_collection = AsyncMock(return_value=MagicMock())
    mock_client.upsert = AsyncMock()
    mock_client.close = AsyncMock()
    indexer._client = mock_client  # type: ignore[assignment]
    return indexer, mock_client


# ----------------------------------------------------------------- reindex


async def test_reindex_writes_points_for_each_trace(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    for qid in ["q-a", "q-b", "q-c"]:
        await _seed_trace(db_session, qid, with_defect=(qid == "q-a"), with_eval=(qid == "q-b"))

    # session_scope() in the indexer pulls from the real engine, which we override:
    monkeypatch.setattr(
        "src.meta_rag.trace_indexer.session_scope",
        _fake_session_scope_factory(db_session),
    )

    indexer, mock_client = _make_indexer_with_mocked_qdrant()
    written = await indexer.reindex()

    assert written == 3
    mock_client.upsert.assert_called_once()
    points = mock_client.upsert.call_args.kwargs["points"]
    assert len(points) == 3
    # Point IDs are deterministic from query_id.
    expected_ids = {_meta_point_id(qid) for qid in ["q-a", "q-b", "q-c"]}
    assert {p.id for p in points} == expected_ids


async def test_reindex_payload_contains_defect_and_eval_fields(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _seed_trace(db_session, "q-with", with_defect=True, with_eval=True)

    monkeypatch.setattr(
        "src.meta_rag.trace_indexer.session_scope",
        _fake_session_scope_factory(db_session),
    )

    indexer, mock_client = _make_indexer_with_mocked_qdrant()
    await indexer.reindex()

    points = mock_client.upsert.call_args.kwargs["points"]
    payload = points[0].payload
    assert payload["query_id"] == "q-with"
    assert payload["source_file"] == "trace:q-with"
    assert payload["defect_count"] == 1
    assert payload["defect_types"] == ["DEFECT_LOW_RETRIEVAL_QUALITY"]
    assert payload["quality_gate"] == "pass"
    assert payload["faithfulness"] == pytest.approx(0.91)
    # Document text carries the human-readable rendering.
    assert "DEFECT_LOW_RETRIEVAL_QUALITY" in payload["text"]
    assert "Faithfulness: 0.910" in payload["text"]


async def test_reindex_respects_since_window(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Seed two recent traces, then a third "old" one (60 days ago).
    await _seed_trace(db_session, "q-new1")
    await _seed_trace(db_session, "q-new2")
    await _seed_trace(db_session, "q-old")
    old = await trace_store.get_by_query_id(db_session, "q-old")
    assert old is not None
    old.created_at = datetime.now(UTC) - timedelta(days=60)
    await db_session.commit()

    monkeypatch.setattr(
        "src.meta_rag.trace_indexer.session_scope",
        _fake_session_scope_factory(db_session),
    )

    indexer, mock_client = _make_indexer_with_mocked_qdrant()
    written = await indexer.reindex(since=datetime.now(UTC) - timedelta(days=7))

    assert written == 2
    points = mock_client.upsert.call_args.kwargs["points"]
    written_qids = {p.payload["query_id"] for p in points}
    assert "q-old" not in written_qids


async def test_reindex_no_rows_is_a_noop(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "src.meta_rag.trace_indexer.session_scope",
        _fake_session_scope_factory(db_session),
    )
    indexer, mock_client = _make_indexer_with_mocked_qdrant()
    written = await indexer.reindex()
    assert written == 0
    mock_client.upsert.assert_not_called()


async def test_reindex_recent_helper_drives_indexer(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _seed_trace(db_session, "q1")

    monkeypatch.setattr(
        "src.meta_rag.trace_indexer.session_scope",
        _fake_session_scope_factory(db_session),
    )

    captured: dict[str, Any] = {}

    class _CaptureIndexer:
        def __init__(self, *_: Any, **__: Any) -> None:
            captured["instantiated"] = True

        async def reindex(self, since: datetime | None, limit: int) -> int:
            captured["since"] = since
            captured["limit"] = limit
            return 1

        async def aclose(self) -> None:
            captured["closed"] = True

    monkeypatch.setattr("src.meta_rag.trace_indexer.TraceIndexer", _CaptureIndexer)

    n = await reindex_recent(_StubEmbedder(), window_minutes=30, limit=42)
    assert n == 1
    assert captured["instantiated"]
    assert captured["closed"]
    assert captured["limit"] == 42
    assert captured["since"] is not None


# --------------------------------------------------------------- helper


def _fake_session_scope_factory(session: AsyncSession) -> Any:
    """Returns an async-context-manager factory that always yields ``session``."""
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _scope() -> AsyncIterator[AsyncSession]:
        yield session

    return _scope
