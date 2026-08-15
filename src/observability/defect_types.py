"""Defect taxonomy.

A *defect* is a runtime quality signal — something about a single query that
suggests retrieval or generation went wrong in a recognizable way. Defects
are detected synchronously in the query path, attached to the request's
trace, and (in Phase 6) persisted for trend analysis.

Severity drives both the log level and downstream alerting policy:

* ``CRITICAL`` — pipeline produced no usable output. Log at ERROR.
* ``HIGH``     — pipeline returned something but it is likely wrong. Log at ERROR.
* ``MEDIUM``   — pipeline returned something but a quality hint was missed. WARNING.
* ``LOW``      — informational; suggests a tuning opportunity. INFO.
"""

from __future__ import annotations

from enum import StrEnum


class DefectType(StrEnum):
    """All defect kinds the :mod:`defect_detector` can emit."""

    LOW_RETRIEVAL_QUALITY = "DEFECT_LOW_RETRIEVAL_QUALITY"
    CONTEXT_TRUNCATED = "DEFECT_CONTEXT_TRUNCATED"
    HALLUCINATION_SIGNAL = "DEFECT_HALLUCINATION_SIGNAL"
    EMPTY_RETRIEVAL = "DEFECT_EMPTY_RETRIEVAL"
    LOW_CHUNK_DIVERSITY = "DEFECT_LOW_CHUNK_DIVERSITY"


class Severity(StrEnum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


DEFAULT_SEVERITY: dict[DefectType, Severity] = {
    DefectType.EMPTY_RETRIEVAL: Severity.CRITICAL,
    DefectType.LOW_RETRIEVAL_QUALITY: Severity.HIGH,
    DefectType.HALLUCINATION_SIGNAL: Severity.HIGH,
    DefectType.CONTEXT_TRUNCATED: Severity.MEDIUM,
    DefectType.LOW_CHUNK_DIVERSITY: Severity.LOW,
}


DESCRIPTIONS: dict[DefectType, str] = {
    DefectType.EMPTY_RETRIEVAL: (
        "Retrieval returned zero chunks. The vector store is empty, the query "
        "produced no matches, or the retriever errored out silently."
    ),
    DefectType.LOW_RETRIEVAL_QUALITY: (
        "Every retrieved chunk scored below the configured similarity "
        "threshold. The vector store likely does not contain content relevant "
        "to this query."
    ),
    DefectType.HALLUCINATION_SIGNAL: (
        "The generated answer shares very little vocabulary with the assembled "
        "context, suggesting the model answered from parametric memory rather "
        "than the retrieved passages."
    ),
    DefectType.CONTEXT_TRUNCATED: (
        "The assembled context dropped one or more retrieved chunks because "
        "the token budget was exceeded. Relevant evidence may have been omitted."
    ),
    DefectType.LOW_CHUNK_DIVERSITY: (
        "All retrieved chunks came from the same source document. May indicate "
        "a corpus skew or a query that only one document covers."
    ),
}
