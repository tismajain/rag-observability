"""Per-defect parametrized tests for the Phase 4 detector.

Spec calls for one test per defect type plus boundary conditions. We cover:

* All five defect types fire in their expected conditions.
* Each detector is silent under the right conditions (no false positives).
* Boundary conditions: score exactly at threshold, single-chunk corpora,
  empty / missing inputs.
* Severity ordering of the returned list.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.ingestion.chunker import ChunkMetadata
from src.observability.defect_detector import (
    HALLUCINATION_OVERLAP_THRESHOLD,
    DefectDetector,
)
from src.observability.defect_types import DefectType, Severity
from src.retrieval.context_assembler import AssembledContext
from src.retrieval.retriever import RetrievedChunk

# --------------------------------------------------------------------- helpers


def _chunk(
    chunk_id: str = "c0",
    source: str = "doc.txt",
    rrf_score: float = 0.8,
    text: str = "alpha beta gamma delta",
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        metadata=ChunkMetadata(
            source_file=source,
            chunk_index=0,
            total_chunks=1,
            chunking_strategy="recursive",
            char_start=0,
            char_end=len(text),
            token_count=max(1, len(text) // 4),
            ingested_at=datetime.now(UTC),
        ),
        dense_score=None,
        sparse_score=None,
        rrf_score=rrf_score,
        rank=0,
    )


def _context(text: str = "alpha beta gamma delta", truncated: bool = False) -> AssembledContext:
    return AssembledContext(
        text=text,
        included_chunks=[],
        dropped_chunks=[_chunk("dropped")] if truncated else [],
        total_tokens=len(text) // 4,
        was_truncated=truncated,
    )


def _detector(similarity_threshold: float = 0.65) -> DefectDetector:
    return DefectDetector(similarity_threshold=similarity_threshold)


def _types(events) -> set[DefectType]:
    return {e.defect_type for e in events}


# ---------------------------------------------------------- EMPTY_RETRIEVAL


def test_empty_retrieval_fires_when_no_chunks() -> None:
    events = _detector().detect("q1", [], _context(), "an answer")
    assert DefectType.EMPTY_RETRIEVAL in _types(events)
    ev = next(e for e in events if e.defect_type == DefectType.EMPTY_RETRIEVAL)
    assert ev.severity == Severity.CRITICAL


def test_empty_retrieval_silent_when_chunks_present() -> None:
    events = _detector().detect("q2", [_chunk()], _context(), "alpha beta gamma")
    assert DefectType.EMPTY_RETRIEVAL not in _types(events)


# ------------------------------------------------------- LOW_RETRIEVAL_QUALITY


def test_low_retrieval_quality_fires_when_all_scores_below_threshold() -> None:
    chunks = [_chunk(f"c{i}", rrf_score=0.3) for i in range(3)]
    events = _detector(similarity_threshold=0.5).detect("q", chunks, _context(), "ans")
    assert DefectType.LOW_RETRIEVAL_QUALITY in _types(events)
    ev = next(e for e in events if e.defect_type == DefectType.LOW_RETRIEVAL_QUALITY)
    assert ev.severity == Severity.HIGH
    assert ev.metadata["max_score"] == pytest.approx(0.3)


def test_low_retrieval_quality_silent_when_max_score_at_threshold() -> None:
    # >= threshold should pass (boundary check)
    chunks = [_chunk("c0", rrf_score=0.5), _chunk("c1", rrf_score=0.4)]
    events = _detector(similarity_threshold=0.5).detect("q", chunks, _context(), "ans")
    assert DefectType.LOW_RETRIEVAL_QUALITY not in _types(events)


def test_low_retrieval_quality_silent_when_one_chunk_above_threshold() -> None:
    chunks = [_chunk("c0", rrf_score=0.3), _chunk("c1", rrf_score=0.9)]
    events = _detector(similarity_threshold=0.5).detect("q", chunks, _context(), "ans")
    assert DefectType.LOW_RETRIEVAL_QUALITY not in _types(events)


# ------------------------------------------------------------ CONTEXT_TRUNCATED


def test_context_truncated_fires_when_assembler_truncated() -> None:
    events = _detector().detect(
        "q",
        [_chunk()],
        _context(truncated=True),
        "alpha gamma",
    )
    assert DefectType.CONTEXT_TRUNCATED in _types(events)
    ev = next(e for e in events if e.defect_type == DefectType.CONTEXT_TRUNCATED)
    assert ev.severity == Severity.MEDIUM
    assert ev.metadata["dropped"] == 1


def test_context_truncated_silent_when_no_drops() -> None:
    events = _detector().detect("q", [_chunk()], _context(truncated=False), "alpha")
    assert DefectType.CONTEXT_TRUNCATED not in _types(events)


def test_context_truncated_silent_when_context_missing() -> None:
    events = _detector().detect("q", [_chunk()], None, "alpha")
    assert DefectType.CONTEXT_TRUNCATED not in _types(events)


# --------------------------------------------------------- LOW_CHUNK_DIVERSITY


def test_low_chunk_diversity_fires_when_all_chunks_share_source() -> None:
    chunks = [_chunk("c0", source="x.txt"), _chunk("c1", source="x.txt")]
    events = _detector().detect("q", chunks, _context(), "alpha gamma")
    assert DefectType.LOW_CHUNK_DIVERSITY in _types(events)
    ev = next(e for e in events if e.defect_type == DefectType.LOW_CHUNK_DIVERSITY)
    assert ev.severity == Severity.LOW
    assert ev.metadata["unique_sources"] == 1


def test_low_chunk_diversity_silent_with_mixed_sources() -> None:
    chunks = [_chunk("c0", source="a.txt"), _chunk("c1", source="b.txt")]
    events = _detector().detect("q", chunks, _context(), "alpha gamma")
    assert DefectType.LOW_CHUNK_DIVERSITY not in _types(events)


def test_low_chunk_diversity_silent_with_single_chunk() -> None:
    # Single chunk → trivially same source, but it's not really "low diversity"
    events = _detector().detect("q", [_chunk()], _context(), "alpha")
    assert DefectType.LOW_CHUNK_DIVERSITY not in _types(events)


# ---------------------------------------------------------- HALLUCINATION_SIGNAL


def test_hallucination_signal_fires_when_overlap_below_threshold() -> None:
    # Answer shares no content words with context → overlap = 0
    ctx = _context(text="apples bananas cherries")
    answer = "submarine elephant volcano oxygen"
    events = _detector().detect("q", [_chunk()], ctx, answer)
    assert DefectType.HALLUCINATION_SIGNAL in _types(events)
    ev = next(e for e in events if e.defect_type == DefectType.HALLUCINATION_SIGNAL)
    assert ev.severity == Severity.HIGH
    assert ev.metadata["overlap_ratio"] == 0.0


def test_hallucination_signal_silent_when_overlap_above_threshold() -> None:
    # Answer copies most content words from context → high overlap
    ctx = _context(text="alpha beta gamma delta epsilon zeta")
    answer = "The alpha gamma epsilon are key — beta also delta zeta"
    events = _detector().detect("q", [_chunk()], ctx, answer)
    assert DefectType.HALLUCINATION_SIGNAL not in _types(events)


def test_hallucination_signal_silent_when_answer_empty() -> None:
    events = _detector().detect("q", [_chunk()], _context("alpha beta"), "")
    assert DefectType.HALLUCINATION_SIGNAL not in _types(events)


def test_hallucination_signal_silent_when_only_stopwords() -> None:
    # Answer reduces to zero content words after stopword stripping
    events = _detector().detect("q", [_chunk()], _context("alpha beta"), "the and or it")
    assert DefectType.HALLUCINATION_SIGNAL not in _types(events)


def test_hallucination_threshold_boundary() -> None:
    # Build an answer with ~exactly the threshold ratio.
    context_words = [f"word{i}" for i in range(20)]
    # 4 of 20 answer-words overlap → 20% overlap, above 15% threshold
    answer_words = context_words[:4] + [f"new{i}" for i in range(16)]
    ctx = _context(text=" ".join(context_words))
    events = _detector().detect("q", [_chunk()], ctx, " ".join(answer_words))
    assert DefectType.HALLUCINATION_SIGNAL not in _types(events)
    # Compute actual overlap for sanity
    assert HALLUCINATION_OVERLAP_THRESHOLD < 4 / 20


# ----------------------------------------------------- ordering & multi-defect


def test_multiple_defects_sorted_by_severity_desc() -> None:
    # Empty retrieval (CRITICAL) + context truncated (MEDIUM) can't coexist
    # (truncation requires included chunks). Use LOW_RETRIEVAL_QUALITY (HIGH)
    # + CONTEXT_TRUNCATED (MEDIUM) + LOW_CHUNK_DIVERSITY (LOW).
    chunks = [
        _chunk("a", source="x.txt", rrf_score=0.1),
        _chunk("b", source="x.txt", rrf_score=0.1),
    ]
    ctx = _context(text="apples", truncated=True)
    answer = "apples are red"  # decent overlap to avoid hallucination defect
    events = _detector(similarity_threshold=0.5).detect("q", chunks, ctx, answer)
    severities = [e.severity for e in events]
    assert len(events) >= 3
    # CRITICAL < HIGH < MEDIUM < LOW in our order
    assert severities == sorted(
        severities,
        key=lambda s: {Severity.CRITICAL: 0, Severity.HIGH: 1, Severity.MEDIUM: 2, Severity.LOW: 3}[
            s
        ],
    )


def test_clean_pipeline_emits_no_defects() -> None:
    chunks = [
        _chunk("a", source="x.txt", rrf_score=0.8),
        _chunk("b", source="y.txt", rrf_score=0.7),
    ]
    ctx = AssembledContext(
        text="alpha beta gamma delta epsilon zeta",
        included_chunks=chunks,
        dropped_chunks=[],
        total_tokens=20,
        was_truncated=False,
    )
    answer = "alpha gamma epsilon zeta"  # high overlap
    events = _detector().detect("q", chunks, ctx, answer)
    assert events == []
