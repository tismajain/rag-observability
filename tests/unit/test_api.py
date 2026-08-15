"""FastAPI surface tests via TestClient.

We construct a *fresh* FastAPI app (rather than reusing ``src.api.main.app``)
so the production lifespan never runs — no real Phoenix exporter, no Qdrant
init, no background indexer task. State setup happens in fixtures:

* The database engine points at an in-memory SQLite. The Phase-6 stores
  + models are compatible with SQLite (we already exercise them in
  ``test_storage.py``).
* ``pipeline_runner`` is monkey-patched per test to expose a stubbed
  pipeline so /query and /meta/query exercise the *router* logic without
  needing a real retriever or LLM.

This lets us validate the HTTP contract — request shapes, status codes,
response envelopes — in fast, in-process tests.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routers import defects, evals, health, meta, query, traces
from src.auth.principal import Principal
from src.ingestion.chunker import ChunkMetadata
from src.observability.defect_detector import DefectEvent as RuntimeDefectEvent
from src.observability.defect_types import DefectType, Severity
from src.pipeline.pipeline_runner import pipeline_runner as runner_singleton
from src.pipeline.rag_pipeline import PipelineResult
from src.retrieval.context_assembler import AssembledContext
from src.retrieval.retriever import RetrievedChunk
from src.storage import defect_store, eval_store, trace_store
from src.storage.database import (
    get_session_factory,
    init_engine,
    shutdown_engine,
)
from src.storage.models import Base

# ----------------------------------------------------------- app + db setup


def _build_app() -> FastAPI:
    """Construct a router-only FastAPI app without the production lifespan."""
    app = FastAPI(title="rag-observability-test")
    app.dependency_overrides[health.require_observability_admin] = lambda: Principal(
        "admin", "test-org", frozenset({"observability_admin"})
    )
    app.include_router(health.router)
    app.include_router(query.router)
    app.include_router(traces.router)
    app.include_router(defects.router)
    app.include_router(evals.router)
    app.include_router(meta.router)
    return app


@pytest.fixture
def app() -> Iterator[FastAPI]:
    """One fresh app per test, with a fresh in-memory SQLite database."""
    import asyncio

    asyncio.get_event_loop_policy().new_event_loop()

    async def _setup() -> None:
        await shutdown_engine()  # in case a prior test left one
        init_engine("sqlite+aiosqlite:///:memory:")
        engine = get_session_factory().kw["bind"]
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def _teardown() -> None:
        await shutdown_engine()

    asyncio.run(_setup())
    yield _build_app()
    asyncio.run(_teardown())


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app) as c:
        yield c


# --------------------------------------------------------- pipeline stubs


def _stub_chunk() -> RetrievedChunk:
    text = "stub context body for the test pipeline"
    return RetrievedChunk(
        chunk_id="stub-1",
        text=text,
        metadata=ChunkMetadata(
            source_file="stub.md",
            chunk_index=0,
            total_chunks=1,
            chunking_strategy="recursive",
            char_start=0,
            char_end=len(text),
            token_count=10,
            ingested_at=datetime.now(UTC),
        ),
        dense_score=0.85,
        sparse_score=0.4,
        rrf_score=0.6,
        rank=0,
    )


def _stub_pipeline_result(query_id: str, with_defect: bool = False) -> PipelineResult:
    from src.generation.llm_client import LLMResponse

    chunk = _stub_chunk()
    ctx = AssembledContext(
        text="[1] source=stub.md chunk=0\nstub context body for the test pipeline",
        included_chunks=[chunk],
        dropped_chunks=[],
        total_tokens=10,
        was_truncated=False,
    )
    defects: list[RuntimeDefectEvent] = []
    if with_defect:
        defects.append(
            RuntimeDefectEvent(
                query_id=query_id,
                trace_id="trace-xyz",
                defect_type=DefectType.LOW_RETRIEVAL_QUALITY,
                severity=Severity.HIGH,
                description="below threshold",
                metadata={"max_score": 0.1},
            )
        )
    return PipelineResult(
        query_id=query_id,
        answer="The stubbed answer.",
        llm=LLMResponse(
            text="The stubbed answer.",
            model="claude-sonnet-4-6",
            provider="anthropic",
            prompt_tokens=120,
            completion_tokens=8,
            finish_reason="end_turn",
            latency_ms=42,
        ),
        retrieved_chunks=[chunk],
        assembled_context=ctx,
        defects=defects,
        total_latency_ms=42,
    )


# ------------------------------------------------------------------ /health


def test_liveness_is_minimal_and_anonymous(client: TestClient) -> None:
    response = client.get("/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_ok_when_all_deps_healthy(client: TestClient) -> None:
    with (
        patch(
            "src.api.routers.health._check_postgres",
            new=AsyncMock(return_value=health.DependencyStatus(name="postgres", healthy=True)),
        ),
        patch(
            "src.api.routers.health._check_qdrant",
            new=AsyncMock(return_value=health.DependencyStatus(name="qdrant", healthy=True)),
        ),
        patch(
            "src.api.routers.health._check_phoenix",
            new=AsyncMock(return_value=health.DependencyStatus(name="phoenix", healthy=True)),
        ),
    ):
        r = client.get("/health/ready")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert {d["name"] for d in body["dependencies"]} == {"postgres", "qdrant", "phoenix"}


def test_health_degraded_when_phoenix_down(client: TestClient) -> None:
    with (
        patch(
            "src.api.routers.health._check_postgres",
            new=AsyncMock(return_value=health.DependencyStatus(name="postgres", healthy=True)),
        ),
        patch(
            "src.api.routers.health._check_qdrant",
            new=AsyncMock(return_value=health.DependencyStatus(name="qdrant", healthy=True)),
        ),
        patch(
            "src.api.routers.health._check_phoenix",
            new=AsyncMock(
                return_value=health.DependencyStatus(
                    name="phoenix", healthy=False, detail="conn refused"
                )
            ),
        ),
    ):
        r = client.get("/health/ready")
    assert r.status_code == 200  # degraded is still 200
    assert r.json()["status"] == "degraded"


def test_health_503_when_postgres_down(client: TestClient) -> None:
    with (
        patch(
            "src.api.routers.health._check_postgres",
            new=AsyncMock(
                return_value=health.DependencyStatus(name="postgres", healthy=False, detail="oops")
            ),
        ),
        patch(
            "src.api.routers.health._check_qdrant",
            new=AsyncMock(return_value=health.DependencyStatus(name="qdrant", healthy=True)),
        ),
        patch(
            "src.api.routers.health._check_phoenix",
            new=AsyncMock(return_value=health.DependencyStatus(name="phoenix", healthy=True)),
        ),
    ):
        r = client.get("/health/ready")
    assert r.status_code == 503
    assert r.json()["status"] == "unhealthy"


# ------------------------------------------------------------------ /query


def test_query_503_when_pipeline_not_ready(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner_singleton, "_pipeline", None)
    monkeypatch.setattr(runner_singleton, "_build_error", "missing-key", raising=False)
    r = client.post("/query", json={"query": "anything"})
    assert r.status_code == 503
    assert "missing-key" in r.json()["detail"]


def test_query_happy_path_returns_typed_envelope(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stub the pipeline; exercise the router's response composition."""
    stub_pipeline = MagicMock()
    stub_pipeline.run = AsyncMock(side_effect=lambda **kw: _stub_pipeline_result(kw["query_id"]))
    monkeypatch.setattr(runner_singleton, "_pipeline", stub_pipeline)
    monkeypatch.setattr(runner_singleton, "_evaluator", None)  # disable async eval scheduling

    r = client.post("/query", json={"query": "what is alpha?", "top_k": 3})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["query_id"]
    assert body["trace_id"]
    assert body["answer"] == "The stubbed answer."
    assert len(body["retrieved_chunks"]) == 1
    chunk = body["retrieved_chunks"][0]
    assert chunk["source_file"] == "stub.md"
    assert chunk["text_preview"].startswith("stub context")
    assert body["context_truncated"] is False
    assert body["defects_detected"] == []
    assert body["latency_ms"] == 42


def test_query_surfaces_defect_types_in_response(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub_pipeline = MagicMock()
    stub_pipeline.run = AsyncMock(
        side_effect=lambda **kw: _stub_pipeline_result(kw["query_id"], with_defect=True)
    )
    monkeypatch.setattr(runner_singleton, "_pipeline", stub_pipeline)
    monkeypatch.setattr(runner_singleton, "_evaluator", None)

    r = client.post("/query", json={"query": "x"})
    assert r.status_code == 200
    assert r.json()["defects_detected"] == ["DEFECT_LOW_RETRIEVAL_QUALITY"]


def test_query_validation_rejects_empty_string(client: TestClient) -> None:
    r = client.post("/query", json={"query": ""})
    assert r.status_code == 422


# ----------------------------------------------------------- /meta/query


def test_meta_query_happy_path(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.meta_rag.meta_pipeline import MetaPipelineResult

    chunk = _stub_chunk()
    chunk = chunk.model_copy(
        update={"metadata": chunk.metadata.model_copy(update={"source_file": "trace:abc-123"})}
    )
    ctx = AssembledContext(
        text="Query ID: abc-123\n…",
        included_chunks=[chunk],
        dropped_chunks=[],
        total_tokens=8,
        was_truncated=False,
    )
    from src.generation.llm_client import LLMResponse

    stub_meta = MagicMock()
    stub_meta.run = AsyncMock(
        return_value=MetaPipelineResult(
            query_id="meta-q1",
            answer="Three queries failed yesterday.",
            llm=LLMResponse(
                text="x",
                model="claude-sonnet-4-6",
                provider="anthropic",
                prompt_tokens=10,
                completion_tokens=5,
                finish_reason="end_turn",
                latency_ms=12,
            ),
            retrieved_traces=[chunk],
            assembled_context=ctx,
            total_latency_ms=120,
        )
    )
    monkeypatch.setattr(runner_singleton, "_meta_pipeline", stub_meta)
    # is_ready depends on the prod pipeline; stub it too.
    monkeypatch.setattr(runner_singleton, "_pipeline", MagicMock())

    r = client.post("/meta/query", json={"query": "what happened yesterday?"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["answer"].startswith("Three queries failed")
    assert body["retrieved_traces"][0]["query_id"] == "abc-123"  # stripped from trace:abc-123
    assert body["context_truncated"] is False
    assert body["latency_ms"] == 120


# ------------------------------------------------------------------ /traces


async def _seed(session_factory: Any) -> None:
    async with session_factory() as session:
        await trace_store.create_trace(
            session,
            query_id="q-1",
            trace_id="t-1",
            query_text="seed query",
            answer_text="seed answer",
            context_truncated=False,
            latency_ms=50,
            top_k=5,
            chunks_retrieved=2,
            model_used="mock-1",
        )
        await defect_store.create_defect(
            session,
            query_id="q-1",
            trace_id="t-1",
            defect_type="DEFECT_LOW_RETRIEVAL_QUALITY",
            severity="HIGH",
            description="seed defect",
        )
        await eval_store.create_eval(
            session,
            query_id="q-1",
            trace_id="t-1",
            faithfulness=0.9,
            context_recall=0.8,
            answer_relevancy=0.85,
            quality_gate_result="pass",
            evaluation_latency_ms=100,
            judge_model="judge-1",
        )
        await session.commit()


def _run_seed(client_fixture: TestClient) -> None:
    """Synchronous helper to seed via the same SQLAlchemy engine the routes use."""
    import asyncio

    asyncio.run(_seed(get_session_factory()))


def test_traces_list_returns_seeded_rows(client: TestClient) -> None:
    _run_seed(client)
    r = client.get("/traces", params={"limit": 10})
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 1
    assert body["items"][0]["query_id"] == "q-1"


def test_get_trace_returns_defects_and_evals_envelope(client: TestClient) -> None:
    _run_seed(client)
    r = client.get("/traces/t-1")
    assert r.status_code == 200
    body = r.json()
    assert body["trace"]["query_id"] == "q-1"
    assert len(body["defects"]) == 1
    assert body["defects"][0]["defect_type"] == "DEFECT_LOW_RETRIEVAL_QUALITY"
    assert len(body["evals"]) == 1
    assert body["evals"][0]["quality_gate_result"] == "pass"


def test_get_trace_404_for_unknown(client: TestClient) -> None:
    r = client.get("/traces/missing-id")
    assert r.status_code == 404


# ------------------------------------------------------------------ /defects


def test_defects_list(client: TestClient) -> None:
    _run_seed(client)
    r = client.get("/defects")
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 1
    assert body["items"][0]["severity"] == "HIGH"


def test_defects_filter_by_severity(client: TestClient) -> None:
    _run_seed(client)
    r = client.get("/defects", params={"severity": "LOW"})
    assert r.status_code == 200
    assert r.json()["total"] == 0


# ------------------------------------------------------------------ /evals


def test_evals_list(client: TestClient) -> None:
    _run_seed(client)
    r = client.get("/evals")
    assert r.status_code == 200
    assert r.json()["total"] == 1


def test_evals_summary_returns_aggregates(client: TestClient) -> None:
    _run_seed(client)
    r = client.get("/evals/summary", params={"period": "7d"})
    assert r.status_code == 200
    body = r.json()
    assert body["period"] == "7d"
    assert body["total_evaluated"] == 1
    assert body["mean_faithfulness"] == pytest.approx(0.9)
    assert body["quality_gate_distribution"]["pass"] == 1


def test_evals_summary_rejects_invalid_period(client: TestClient) -> None:
    r = client.get("/evals/summary", params={"period": "garbage"})
    assert r.status_code == 400
