"""Pipeline composition tests with all collaborators mocked.

Verifies that:
* The pipeline calls retrieve → (rerank) → assemble → generate in order.
* Defect detection runs on the final answer + context.
* The reranking toggle is honored.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.generation.llm_client import LLMResponse, MockClient
from src.ingestion.chunker import ChunkMetadata
from src.pipeline.rag_pipeline import RAGPipeline
from src.retrieval.context_assembler import AssembledContext
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.retriever import RetrievedChunk

# ---------------------------------------------------------------- helpers


def _chunk(cid: str, text: str = "alpha beta gamma") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid,
        text=text,
        metadata=ChunkMetadata(
            source_file="doc.txt",
            chunk_index=0,
            total_chunks=1,
            chunking_strategy="recursive",
            char_start=0,
            char_end=len(text),
            token_count=max(1, len(text) // 4),
            ingested_at=datetime.now(UTC),
        ),
        dense_score=0.9,
        sparse_score=0.5,
        rrf_score=0.8,
        rank=0,
    )


def _make_pipeline(hits: list[RetrievedChunk]) -> RAGPipeline:
    retriever = MagicMock()
    retriever.retrieve = AsyncMock(return_value=hits)
    reranker = CrossEncoderReranker(enabled=False)  # bypass cross-encoder model
    llm = MockClient()
    return RAGPipeline(retriever=retriever, reranker=reranker, llm=llm)


# ----------------------------------------------------------------- tests


async def test_pipeline_runs_end_to_end_with_mock_llm() -> None:
    pipeline = _make_pipeline([_chunk("c1"), _chunk("c2")])
    result = await pipeline.run(query_id="q1", query="What is alpha?", top_k=2)

    assert result.query_id == "q1"
    assert result.answer.startswith("[mock answer]")
    assert len(result.retrieved_chunks) == 2
    assert isinstance(result.llm, LLMResponse)
    assert isinstance(result.assembled_context, AssembledContext)
    assert result.total_latency_ms >= 0


async def test_pipeline_emits_empty_retrieval_defect_when_no_hits() -> None:
    pipeline = _make_pipeline([])
    result = await pipeline.run(query_id="q-empty", query="anything", top_k=5)
    defect_types = [d.defect_type.value for d in result.defects]
    assert "DEFECT_EMPTY_RETRIEVAL" in defect_types


async def test_pipeline_honors_enable_reranking_false() -> None:
    """When reranking is disabled, raw_hits[:top_k] is returned untouched."""
    hits = [_chunk(f"c{i}") for i in range(5)]
    pipeline = _make_pipeline(hits)
    result = await pipeline.run(query_id="q-no-rerank", query="x", top_k=3, enable_reranking=False)
    assert [c.chunk_id for c in result.retrieved_chunks] == ["c0", "c1", "c2"]


async def test_pipeline_calls_retriever_with_more_candidates_than_top_k() -> None:
    """We request top_k*4 candidates so the reranker has room to reorder."""
    hits = [_chunk(f"c{i}") for i in range(20)]
    retriever = MagicMock()
    retriever.retrieve = AsyncMock(return_value=hits)
    pipeline = RAGPipeline(
        retriever=retriever,
        reranker=CrossEncoderReranker(enabled=False),
        llm=MockClient(),
    )
    await pipeline.run(query_id="q", query="x", top_k=5)

    retriever.retrieve.assert_awaited_once()
    call_args = retriever.retrieve.await_args
    assert call_args is not None
    # candidates = max(top_k*4, 20) = 20
    assert call_args.kwargs["top_k"] == 20


async def test_pipeline_assembled_context_truncated_triggers_defect() -> None:
    """Long chunks → context truncated → CONTEXT_TRUNCATED defect fires."""
    big_text = "word " * 2000  # ~2000 tokens each
    hits = [_chunk(f"c{i}", text=big_text) for i in range(5)]
    retriever = MagicMock()
    retriever.retrieve = AsyncMock(return_value=hits)
    pipeline = RAGPipeline(
        retriever=retriever,
        reranker=CrossEncoderReranker(enabled=False),
        llm=MockClient(),
        max_context_tokens=500,
    )
    result = await pipeline.run(query_id="q", query="x", top_k=5)
    defect_types = [d.defect_type.value for d in result.defects]
    assert "DEFECT_CONTEXT_TRUNCATED" in defect_types
    assert result.assembled_context.was_truncated is True


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
