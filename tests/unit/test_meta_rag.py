"""Unit tests for the Phase 7 meta-RAG layer.

Two concerns covered:

* :func:`format_trace_document` produces a stable, field-dense string with all
  the named fields the meta retriever needs to embed signal.
* :class:`MetaRAGPipeline` composes retrieve → assemble → generate, surfaces
  the meta prompt, and produces a :class:`MetaPipelineResult` with the
  expected shape.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.generation.llm_client import LLMResponse, MockClient
from src.ingestion.chunker import ChunkMetadata
from src.meta_rag.meta_pipeline import MetaRAGPipeline
from src.meta_rag.trace_indexer import format_trace_document
from src.retrieval.context_assembler import AssembledContext
from src.retrieval.retriever import RetrievedChunk
from src.storage.models import DefectEvent, EvalScore, Trace

# ---------------------------------------------------------------- helpers


def _trace_chunk(query_id: str, text: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"point-{query_id}",
        text=text,
        metadata=ChunkMetadata(
            source_file=f"trace:{query_id}",
            chunk_index=0,
            total_chunks=1,
            chunking_strategy="recursive",
            char_start=0,
            char_end=len(text),
            token_count=max(1, len(text) // 4),
            ingested_at=datetime.now(UTC),
        ),
        dense_score=0.7,
        sparse_score=0.2,
        rrf_score=0.5,
        rank=0,
    )


def _make_trace(query_id: str = "q-1") -> Trace:
    t = Trace(
        query_id=query_id,
        trace_id="trace-abc",
        query_text="What is alpha?",
        answer_text="Alpha is a Greek letter.",
        context_truncated=False,
        latency_ms=350,
        retrieval_strategy="hybrid",
        top_k=5,
        chunks_retrieved=3,
        model_used="claude-sonnet-4-6",
    )
    t.created_at = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    return t


# ---------------------------------------------------- format_trace_document --


def test_format_trace_document_includes_all_named_fields() -> None:
    trace = _make_trace()
    doc = format_trace_document(trace, defects=[], evals=[])
    # Spot-check the field-dense lines — these are the embedding's signal.
    assert "Query ID: q-1" in doc
    assert "Trace ID: trace-abc" in doc
    assert "Model: claude-sonnet-4-6" in doc
    assert "Latency: 350ms" in doc
    assert "Question: What is alpha?" in doc
    assert "Answer: Alpha is a Greek letter." in doc
    assert "Defects detected: none" in doc
    assert "Evaluation scores: not evaluated" in doc


def test_format_trace_document_renders_defects_and_evals() -> None:
    trace = _make_trace()
    defects = [
        DefectEvent(
            query_id=trace.query_id,
            trace_id=trace.trace_id,
            defect_type="DEFECT_LOW_RETRIEVAL_QUALITY",
            severity="HIGH",
            description="Top chunk below threshold.",
            event_metadata={},
        )
    ]
    evals = [
        EvalScore(
            query_id=trace.query_id,
            trace_id=trace.trace_id,
            faithfulness=0.9,
            context_recall=0.75,
            answer_relevancy=0.81,
            quality_gate_result="warn",
            evaluation_latency_ms=2100,
            judge_model="judge-x",
            status="ok",
        )
    ]
    doc = format_trace_document(trace, defects=defects, evals=evals)
    assert "DEFECT_LOW_RETRIEVAL_QUALITY (HIGH)" in doc
    assert "Faithfulness: 0.900" in doc
    assert "Context recall: 0.750" in doc
    assert "Answer relevancy: 0.810" in doc
    assert "Quality gate: warn" in doc


# ----------------------------------------------------- MetaRAGPipeline -------


def _make_meta_pipeline(hits: list[RetrievedChunk]) -> MetaRAGPipeline:
    retriever = MagicMock()
    retriever.retrieve = AsyncMock(return_value=hits)
    return MetaRAGPipeline(retriever=retriever, llm=MockClient())


async def test_meta_pipeline_runs_end_to_end() -> None:
    hits = [
        _trace_chunk("q-1", "Query ID: q-1\nAnswer: alpha is a letter"),
        _trace_chunk("q-2", "Query ID: q-2\nAnswer: beta is a letter"),
    ]
    pipeline = _make_meta_pipeline(hits)
    result = await pipeline.run(
        query_id="meta-1", query="What queries asked about letters?", top_k=2
    )

    assert result.query_id == "meta-1"
    assert result.answer.startswith("[mock answer]")
    assert len(result.retrieved_traces) == 2
    assert isinstance(result.llm, LLMResponse)
    assert isinstance(result.assembled_context, AssembledContext)


async def test_meta_pipeline_calls_retriever_with_requested_top_k() -> None:
    """Meta pipeline retrieves exactly top_k — no extra candidates for a missing reranker."""
    retriever = MagicMock()
    retriever.retrieve = AsyncMock(return_value=[])
    pipeline = MetaRAGPipeline(retriever=retriever, llm=MockClient())
    await pipeline.run(query_id="m", query="x", top_k=7)

    retriever.retrieve.assert_awaited_once()
    assert retriever.retrieve.await_args.kwargs["top_k"] == 7


async def test_meta_pipeline_handles_empty_retrieval() -> None:
    pipeline = _make_meta_pipeline([])
    result = await pipeline.run(query_id="m", query="anything", top_k=3)
    # Empty retrieval is allowed — the meta layer is a debugging surface, not
    # a quality-gated one. The pipeline still completes and returns a result.
    assert result.retrieved_traces == []
    assert result.assembled_context.text == ""
    assert result.answer.startswith("[mock answer]")


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
