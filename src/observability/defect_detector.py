"""Runtime defect detection.

Run :meth:`DefectDetector.detect` synchronously inside the /query path after
generation completes. It walks five quality checks against the request's
artifacts and returns a list of :class:`DefectEvent`. Each event is also:

* attached to the current OTel span (so Phoenix shows it on the trace)
* logged structurally (level = severity-driven)
* (Phase 6) persisted to PostgreSQL via the defect store

Persistence is intentionally out of scope here — :mod:`storage.defect_store`
will land in Phase 6 and consume the same :class:`DefectEvent` objects.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import uuid4

import structlog
from opentelemetry import trace
from pydantic import BaseModel, Field

from src.config.settings import settings
from src.observability.attributes import SpanAttributes
from src.observability.defect_types import (
    DEFAULT_SEVERITY,
    DESCRIPTIONS,
    DefectType,
    Severity,
)

if TYPE_CHECKING:
    from src.retrieval.context_assembler import AssembledContext
    from src.retrieval.retriever import RetrievedChunk

log = structlog.get_logger(__name__)

HALLUCINATION_OVERLAP_THRESHOLD = 0.15

# Compact English stopword set. Avoids the nltk dependency; "simple but
# effective" per the spec. If Phase 4+ wants richer NLP, swap this for nltk's
# stopword list or a regex-based noun-phrase extractor.
_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "for",
        "from",
        "had",
        "has",
        "have",
        "he",
        "her",
        "his",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "our",
        "she",
        "such",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "to",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
        "do",
        "does",
        "did",
        "can",
        "could",
        "should",
        "may",
        "might",
        "about",
        "above",
        "after",
        "again",
        "all",
        "am",
        "any",
        "because",
        "been",
        "before",
        "being",
        "below",
        "between",
        "both",
        "down",
        "during",
        "each",
        "few",
        "further",
        "here",
        "how",
        "just",
        "more",
        "most",
        "no",
        "not",
        "now",
        "off",
        "once",
        "only",
        "other",
        "out",
        "over",
        "own",
        "same",
        "so",
        "some",
        "too",
        "under",
        "until",
        "up",
        "very",
    }
)

_WORD_RE = re.compile(r"\b[\w\-']{2,}\b")


def _content_words(text: str) -> set[str]:
    """Lowercase content-word set (stopwords removed)."""
    tokens = (m.group(0).lower() for m in _WORD_RE.finditer(text))
    return {t for t in tokens if t not in _STOPWORDS}


class DefectEvent(BaseModel):
    """One detected defect for a single query."""

    defect_id: str = Field(default_factory=lambda: str(uuid4()))
    query_id: str
    trace_id: str
    defect_type: DefectType
    severity: Severity
    description: str
    metadata: dict[str, object] = Field(default_factory=dict)
    detected_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class DefectDetector:
    """Computes :class:`DefectEvent` objects for a single query.

    Stateless beyond config — safe to share across requests. The five checks
    are pure functions of the request's artifacts plus the global similarity
    threshold from settings.
    """

    def __init__(
        self,
        similarity_threshold: float | None = None,
        hallucination_overlap_threshold: float = HALLUCINATION_OVERLAP_THRESHOLD,
    ) -> None:
        self._sim_threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else settings.observability.defect_similarity_threshold
        )
        self._overlap_threshold = hallucination_overlap_threshold

    def detect(
        self,
        query_id: str,
        retrieved_chunks: list[RetrievedChunk],
        assembled_context: AssembledContext | None,
        generated_answer: str | None,
    ) -> list[DefectEvent]:
        """Run all five checks. Returns possibly-empty list ordered by severity."""
        trace_id = _current_trace_id()
        events: list[DefectEvent] = []

        # Order matters only in tie-breaking; severity sort at the end is the
        # source of truth for display ordering.
        ev = self._check_empty_retrieval(query_id, trace_id, retrieved_chunks)
        if ev:
            events.append(ev)

        ev = self._check_low_retrieval_quality(query_id, trace_id, retrieved_chunks)
        if ev:
            events.append(ev)

        ev = self._check_context_truncated(query_id, trace_id, assembled_context)
        if ev:
            events.append(ev)

        ev = self._check_low_chunk_diversity(query_id, trace_id, retrieved_chunks)
        if ev:
            events.append(ev)

        ev = self._check_hallucination_signal(
            query_id, trace_id, assembled_context, generated_answer
        )
        if ev:
            events.append(ev)

        events.sort(key=lambda e: _SEVERITY_ORDER[e.severity])
        if events:
            self._record(events)
        return events

    # ----------------------------------------------------------------- checks

    def _check_empty_retrieval(
        self,
        query_id: str,
        trace_id: str,
        retrieved: list[RetrievedChunk],
    ) -> DefectEvent | None:
        if retrieved:
            return None
        return _make_event(
            query_id,
            trace_id,
            DefectType.EMPTY_RETRIEVAL,
            metadata={"chunks_retrieved": 0},
        )

    def _check_low_retrieval_quality(
        self,
        query_id: str,
        trace_id: str,
        retrieved: list[RetrievedChunk],
    ) -> DefectEvent | None:
        if not retrieved:
            return None  # EMPTY_RETRIEVAL covers this case

        # The threshold is calibrated for cosine similarity (0..1). RRF scores
        # have a natural ceiling around 0.03 and would always trip a 0.65 gate.
        # Prefer dense_score when present; fall back to sparse, then RRF.
        def _quality(c: RetrievedChunk) -> float:
            if c.dense_score is not None:
                return c.dense_score
            if c.sparse_score is not None:
                return c.sparse_score
            return c.rrf_score or 0.0

        scores = [_quality(c) for c in retrieved]
        max_score = max(scores)
        if max_score >= self._sim_threshold:
            return None
        return _make_event(
            query_id,
            trace_id,
            DefectType.LOW_RETRIEVAL_QUALITY,
            metadata={
                "max_score": max_score,
                "threshold": self._sim_threshold,
                "chunks_retrieved": len(retrieved),
                "signal": "dense" if retrieved[0].dense_score is not None else "rrf",
            },
        )

    def _check_context_truncated(
        self,
        query_id: str,
        trace_id: str,
        ctx: AssembledContext | None,
    ) -> DefectEvent | None:
        if ctx is None or not ctx.was_truncated:
            return None
        return _make_event(
            query_id,
            trace_id,
            DefectType.CONTEXT_TRUNCATED,
            metadata={
                "included": len(ctx.included_chunks),
                "dropped": len(ctx.dropped_chunks),
                "total_tokens": ctx.total_tokens,
            },
        )

    def _check_low_chunk_diversity(
        self,
        query_id: str,
        trace_id: str,
        retrieved: list[RetrievedChunk],
    ) -> DefectEvent | None:
        # Need at least two chunks to call it "low diversity".
        if len(retrieved) < 2:
            return None
        sources = {c.metadata.source_file for c in retrieved}
        if len(sources) > 1:
            return None
        return _make_event(
            query_id,
            trace_id,
            DefectType.LOW_CHUNK_DIVERSITY,
            metadata={
                "unique_sources": 1,
                "chunks_retrieved": len(retrieved),
                "source_file": next(iter(sources)),
            },
        )

    def _check_hallucination_signal(
        self,
        query_id: str,
        trace_id: str,
        ctx: AssembledContext | None,
        answer: str | None,
    ) -> DefectEvent | None:
        # Need both an assembled context and a non-trivial answer to compute
        # the overlap signal.
        if ctx is None or not ctx.text or not answer:
            return None
        answer_words = _content_words(answer)
        if not answer_words:
            return None
        context_words = _content_words(ctx.text)
        overlap = len(answer_words & context_words) / len(answer_words)
        if overlap >= self._overlap_threshold:
            return None
        return _make_event(
            query_id,
            trace_id,
            DefectType.HALLUCINATION_SIGNAL,
            metadata={
                "overlap_ratio": round(overlap, 4),
                "threshold": self._overlap_threshold,
                "answer_unique_words": len(answer_words),
                "context_unique_words": len(context_words),
            },
        )

    # ----------------------------------------------------------------- record

    def _record(self, events: list[DefectEvent]) -> None:
        """Attach to the current span + emit a log line per event."""
        span = trace.get_current_span()
        if span.is_recording():
            span.set_attribute(SpanAttributes.DEFECT_COUNT, len(events))
            span.set_attribute(
                SpanAttributes.DEFECT_TYPES,
                json.dumps([e.defect_type.value for e in events]),
            )
            span.set_attribute(
                SpanAttributes.DEFECT_SEVERITIES,
                json.dumps([e.severity.value for e in events]),
            )

        for ev in events:
            level = _LEVEL_FOR_SEVERITY[ev.severity]
            log.log(
                level,
                "observability.defect.emitted",
                defect_id=ev.defect_id,
                query_id=ev.query_id,
                defect_type=ev.defect_type.value,
                severity=ev.severity.value,
                description=ev.description,
                **{f"meta.{k}": v for k, v in ev.metadata.items()},
            )


# --------------------------------------------------------------------- helpers


_SEVERITY_ORDER: dict[Severity, int] = {
    Severity.CRITICAL: 0,
    Severity.HIGH: 1,
    Severity.MEDIUM: 2,
    Severity.LOW: 3,
}

_LEVEL_FOR_SEVERITY: dict[Severity, int] = {
    Severity.CRITICAL: logging.ERROR,
    Severity.HIGH: logging.ERROR,
    Severity.MEDIUM: logging.WARNING,
    Severity.LOW: logging.INFO,
}


def _current_trace_id() -> str:
    span = trace.get_current_span()
    ctx = span.get_span_context() if span else None
    if ctx and ctx.is_valid:
        return f"{ctx.trace_id:032x}"
    return "0" * 32


def _make_event(
    query_id: str,
    trace_id: str,
    defect_type: DefectType,
    metadata: dict[str, object],
) -> DefectEvent:
    return DefectEvent(
        query_id=query_id,
        trace_id=trace_id,
        defect_type=defect_type,
        severity=DEFAULT_SEVERITY[defect_type],
        description=DESCRIPTIONS[defect_type],
        metadata=metadata,
    )
