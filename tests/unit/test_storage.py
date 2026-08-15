"""Store CRUD round-trip tests against in-memory SQLite.

Each test runs in isolation against a fresh database created by the fixture.
SQLite is structurally close enough to Postgres for the store logic we want
to verify (basic CRUD + filters + the eval summary aggregation).

These complement the integration test of /query → DB → /traces, which uses
the real Postgres in Docker.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src.storage import defect_store, eval_store, trace_store
from src.storage.models import Base


@pytest_asyncio.fixture
async def session() -> AsyncIterator[AsyncSession]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    async with factory() as s:
        yield s
        await s.rollback()
    await engine.dispose()


# ---------------------------------------------------------------- trace_store


async def test_create_and_get_trace(session: AsyncSession) -> None:
    t = await trace_store.create_trace(
        session,
        query_id="q-1",
        trace_id="t-1",
        query_text="hello",
        answer_text="world",
        context_truncated=False,
        latency_ms=42,
        top_k=5,
        chunks_retrieved=3,
        model_used="mock-1",
    )
    await session.commit()

    fetched = await trace_store.get_by_query_id(session, "q-1")
    assert fetched is not None
    assert fetched.id == t.id
    assert fetched.latency_ms == 42

    by_trace = await trace_store.get_by_trace_id(session, "t-1")
    assert by_trace is not None and by_trace.id == t.id


async def test_list_traces_pagination_and_total(session: AsyncSession) -> None:
    for i in range(5):
        await trace_store.create_trace(
            session,
            query_id=f"q-{i}",
            trace_id=f"t-{i}",
            query_text=f"q{i}",
            answer_text="a",
            context_truncated=False,
            latency_ms=i,
            top_k=5,
            chunks_retrieved=1,
            model_used="m",
        )
    await session.commit()

    items, total = await trace_store.list_traces(session, limit=2, offset=0)
    assert total == 5
    assert len(items) == 2

    items, _ = await trace_store.list_traces(session, limit=10, offset=3)
    assert len(items) == 2


async def test_list_traces_filters_by_defects(session: AsyncSession) -> None:
    await trace_store.create_trace(
        session,
        query_id="qA",
        trace_id="tA",
        query_text="x",
        answer_text="y",
        context_truncated=False,
        latency_ms=1,
        top_k=5,
        chunks_retrieved=1,
        model_used="m",
    )
    await trace_store.create_trace(
        session,
        query_id="qB",
        trace_id="tB",
        query_text="x",
        answer_text="y",
        context_truncated=False,
        latency_ms=1,
        top_k=5,
        chunks_retrieved=1,
        model_used="m",
    )
    await defect_store.create_defect(
        session,
        query_id="qA",
        trace_id="tA",
        defect_type="DEFECT_EMPTY_RETRIEVAL",
        severity="CRITICAL",
        description="-",
    )
    await session.commit()

    with_def, _ = await trace_store.list_traces(session, has_defects=True)
    without_def, _ = await trace_store.list_traces(session, has_defects=False)
    assert {t.query_id for t in with_def} == {"qA"}
    assert {t.query_id for t in without_def} == {"qB"}


# ---------------------------------------------------------------- defect_store


async def test_bulk_create_defects_round_trip(session: AsyncSession) -> None:
    await trace_store.create_trace(
        session,
        query_id="qD",
        trace_id="tD",
        query_text="x",
        answer_text="y",
        context_truncated=False,
        latency_ms=1,
        top_k=5,
        chunks_retrieved=1,
        model_used="m",
    )
    count = await defect_store.bulk_create_defects(
        session,
        [
            {
                "query_id": "qD",
                "trace_id": "tD",
                "defect_type": "DEFECT_CONTEXT_TRUNCATED",
                "severity": "MEDIUM",
                "description": "-",
                "event_metadata": {"dropped": 2},
            },
            {
                "query_id": "qD",
                "trace_id": "tD",
                "defect_type": "DEFECT_HALLUCINATION_SIGNAL",
                "severity": "HIGH",
                "description": "-",
                "event_metadata": {"overlap_ratio": 0.01},
            },
        ],
    )
    await session.commit()
    assert count == 2

    items, total = await defect_store.list_defects(session)
    assert total == 2
    severities = {d.severity for d in items}
    assert severities == {"MEDIUM", "HIGH"}


async def test_list_defects_filters(session: AsyncSession) -> None:
    await trace_store.create_trace(
        session,
        query_id="qE",
        trace_id="tE",
        query_text="x",
        answer_text="y",
        context_truncated=False,
        latency_ms=1,
        top_k=5,
        chunks_retrieved=1,
        model_used="m",
    )
    for i, sev in enumerate(["LOW", "HIGH", "HIGH"]):
        await defect_store.create_defect(
            session,
            query_id="qE",
            trace_id="tE",
            defect_type="DEFECT_LOW_CHUNK_DIVERSITY"
            if sev == "LOW"
            else "DEFECT_LOW_RETRIEVAL_QUALITY",
            severity=sev,
            description=f"d{i}",
        )
    await session.commit()

    items, total = await defect_store.list_defects(session, severity="HIGH")
    assert total == 2
    items, total = await defect_store.list_defects(
        session, defect_type="DEFECT_LOW_CHUNK_DIVERSITY"
    )
    assert total == 1


# ---------------------------------------------------------------- eval_store


async def test_create_and_summary_eval(session: AsyncSession) -> None:
    for i in range(3):
        await trace_store.create_trace(
            session,
            query_id=f"q-{i}",
            trace_id=f"t-{i}",
            query_text="x",
            answer_text="y",
            context_truncated=False,
            latency_ms=1,
            top_k=5,
            chunks_retrieved=1,
            model_used="m",
        )
        await eval_store.create_eval(
            session,
            query_id=f"q-{i}",
            trace_id=f"t-{i}",
            faithfulness=0.9,
            context_recall=0.8,
            answer_relevancy=0.85,
            quality_gate_result="pass" if i < 2 else "fail",
            evaluation_latency_ms=100,
            judge_model="judge-1",
        )
    await session.commit()

    summary = await eval_store.summary(session, period_days=7)
    assert summary["total_evaluated"] == 3
    assert summary["mean_faithfulness"] == pytest.approx(0.9)
    dist = summary["quality_gate_distribution"]
    assert dist == {"pass": 2, "warn": 0, "fail": 1, "skip": 0}


async def test_eval_summary_period_filter(session: AsyncSession) -> None:
    """Rows older than the period must be excluded."""
    await trace_store.create_trace(
        session,
        query_id="qO",
        trace_id="tO",
        query_text="x",
        answer_text="y",
        context_truncated=False,
        latency_ms=1,
        top_k=5,
        chunks_retrieved=1,
        model_used="m",
    )
    e = await eval_store.create_eval(
        session,
        query_id="qO",
        trace_id="tO",
        faithfulness=0.5,
        context_recall=0.5,
        answer_relevancy=0.5,
        quality_gate_result="warn",
        evaluation_latency_ms=10,
        judge_model="j",
    )
    # Force evaluated_at backwards
    e.evaluated_at = datetime.now(UTC) - timedelta(days=30)
    await session.commit()

    summary = await eval_store.summary(session, period_days=7)
    assert summary["total_evaluated"] == 0
