"""Stage-level span helpers.

These are thin context managers that:

* open a named span with a consistent name (``rag.<stage>``)
* set the canonical attributes for the stage
* catch any exception, record it on the span, set status to ERROR, and re-raise

Modules **may** still call ``tracer.start_as_current_span(...)`` directly when
they need fine-grained sub-spans (the ingestion modules already do). These
helpers exist so the *top-level* pipeline stages — query → retrieve → assemble
→ generate → defect-detect → eval — have a uniform shape that's easy to scan
in Phoenix.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from opentelemetry.trace import Span, Status, StatusCode

from src.observability.attributes import SpanAttributes
from src.observability.tracer import get_tracer

_tracer = get_tracer(__name__)


@contextmanager
def _stage_span(name: str, **attrs: object) -> Iterator[Span]:
    """Internal helper: open a span, set attributes, ERROR + re-raise on failure."""
    with _tracer.start_as_current_span(name) as span:
        for key, value in attrs.items():
            if value is not None:
                span.set_attribute(key, value)  # type: ignore[arg-type]
        try:
            yield span
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise


@contextmanager
def root_span(query_id: str, query_text: str) -> Iterator[Span]:
    """Top-level span for one user query. The /query endpoint opens this."""
    with _stage_span(
        "rag.query",
        **{
            SpanAttributes.QUERY_ID: query_id,
            # Query text can contain secrets or private document content. Keep
            # only its length in telemetry; content stays on the request path.
            "rag.query.length": len(query_text),
        },
    ) as span:
        yield span


@contextmanager
def retrieval_span(query_id: str, query_text: str, top_k: int) -> Iterator[Span]:
    with _stage_span(
        "rag.retrieval",
        **{
            SpanAttributes.QUERY_ID: query_id,
            "rag.query.length": len(query_text),
            SpanAttributes.RETRIEVAL_TOP_K: top_k,
        },
    ) as span:
        yield span


@contextmanager
def context_assembly_span(query_id: str, max_tokens: int, candidate_count: int) -> Iterator[Span]:
    with _stage_span(
        "rag.context.assemble",
        **{
            SpanAttributes.QUERY_ID: query_id,
            SpanAttributes.CONTEXT_MAX_TOKENS: max_tokens,
            SpanAttributes.CONTEXT_CANDIDATE_COUNT: candidate_count,
        },
    ) as span:
        yield span


@contextmanager
def generation_span(query_id: str, model: str, provider: str) -> Iterator[Span]:
    with _stage_span(
        "rag.generation",
        **{
            SpanAttributes.QUERY_ID: query_id,
            SpanAttributes.GENERATION_MODEL: model,
            SpanAttributes.GENERATION_PROVIDER: provider,
        },
    ) as span:
        yield span


@contextmanager
def defect_span(query_id: str) -> Iterator[Span]:
    with _stage_span(
        "rag.defect.detect",
        **{SpanAttributes.QUERY_ID: query_id},
    ) as span:
        yield span


@contextmanager
def eval_span(query_id: str, judge_model: str) -> Iterator[Span]:
    with _stage_span(
        "rag.eval",
        **{
            SpanAttributes.QUERY_ID: query_id,
            SpanAttributes.EVAL_JUDGE_MODEL: judge_model,
        },
    ) as span:
        yield span
