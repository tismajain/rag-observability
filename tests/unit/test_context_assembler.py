"""Tests for the token-budget context assembler.

Covers three things:
* Greedy packing under a token budget with truncation tracking.
* The "always include the first chunk even if it overflows" rule — partial
  relevance is better than nothing for downstream generation.
* Provenance headers + separators contribute to the running total.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.ingestion.chunker import ChunkMetadata, count_tokens
from src.retrieval.context_assembler import (
    CHUNK_SEPARATOR,
    assemble_context,
)
from src.retrieval.retriever import RetrievedChunk


def _chunk(idx: int, text: str = "alpha beta gamma delta") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"c{idx}",
        text=text,
        metadata=ChunkMetadata(
            source_file=f"doc{idx}.txt",
            chunk_index=idx,
            total_chunks=10,
            chunking_strategy="recursive",
            char_start=0,
            char_end=len(text),
            token_count=count_tokens(text),
            ingested_at=datetime.now(UTC),
        ),
        dense_score=0.5,
        sparse_score=0.3,
        rrf_score=0.4,
        rank=idx,
    )


def test_empty_input_returns_empty_context() -> None:
    out = assemble_context([], max_context_tokens=1000)
    assert out.text == ""
    assert out.total_tokens == 0
    assert out.was_truncated is False
    assert out.included_chunks == []
    assert out.dropped_chunks == []


def test_fits_within_budget_includes_all() -> None:
    chunks = [_chunk(i) for i in range(3)]
    out = assemble_context(chunks, max_context_tokens=500)
    assert len(out.included_chunks) == 3
    assert out.dropped_chunks == []
    assert out.was_truncated is False
    assert out.total_tokens > 0


def test_first_chunk_always_included_even_if_oversized() -> None:
    """Partial relevance beats nothing — the spec says always keep the top chunk."""
    big = "word " * 2000
    chunks = [_chunk(0, text=big), _chunk(1)]
    out = assemble_context(chunks, max_context_tokens=50)
    assert len(out.included_chunks) == 1
    assert out.included_chunks[0].chunk_id == "c0"
    assert out.was_truncated is True
    assert len(out.dropped_chunks) == 1


def test_truncation_drops_later_chunks_when_budget_exhausted() -> None:
    chunks = [_chunk(i, text="word " * 200) for i in range(5)]
    out = assemble_context(chunks, max_context_tokens=400)
    assert 1 <= len(out.included_chunks) < 5
    assert len(out.dropped_chunks) > 0
    assert out.was_truncated is True
    # Dropped chunks preserve original ordering relative to inputs.
    dropped_ids = [c.chunk_id for c in out.dropped_chunks]
    assert dropped_ids == sorted(dropped_ids, key=lambda x: int(x[1:]))


def test_separator_appears_between_chunks() -> None:
    chunks = [_chunk(0, "alpha"), _chunk(1, "beta")]
    out = assemble_context(chunks, max_context_tokens=500)
    # Two chunks → exactly one separator in the rendered text.
    assert out.text.count(CHUNK_SEPARATOR) == 1


def test_provenance_header_embedded_per_chunk() -> None:
    chunks = [_chunk(7, "alpha")]
    out = assemble_context(chunks, max_context_tokens=500)
    assert "[1]" in out.text  # rendered index
    assert "source=doc7.txt" in out.text
    assert "chunk=7" in out.text


def test_zero_budget_raises() -> None:
    with pytest.raises(ValueError, match="max_context_tokens"):
        assemble_context([_chunk(0)], max_context_tokens=0)
