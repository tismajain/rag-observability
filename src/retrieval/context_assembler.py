"""Assemble retrieved chunks into a single prompt-ready context string.

Reranked chunks come in score-ordered (most relevant first). We greedily add
chunks while a running token tally stays under ``max_context_tokens``. The
moment the next chunk would push us over, we stop and record the rest as
``dropped_chunks`` — a signal the defect detector uses to raise
``DEFECT_CONTEXT_TRUNCATED`` in Phase 4.

The output is shaped so that downstream prompts can either use ``text``
directly or iterate ``included_chunks`` for citation-friendly formatting.
"""

from __future__ import annotations

import structlog
from pydantic import BaseModel, Field

from src.ingestion.chunker import count_tokens
from src.observability.tracer import get_tracer
from src.retrieval.retriever import RetrievedChunk

log = structlog.get_logger(__name__)

DEFAULT_MAX_CONTEXT_TOKENS = 3000
CHUNK_SEPARATOR = "\n\n---\n\n"


class AssembledContext(BaseModel):
    text: str
    included_chunks: list[RetrievedChunk] = Field(default_factory=list)
    dropped_chunks: list[RetrievedChunk] = Field(default_factory=list)
    total_tokens: int = Field(0, ge=0)
    was_truncated: bool = False


def _format_chunk(chunk: RetrievedChunk, idx: int) -> str:
    """Render a chunk with a lightweight provenance header.

    The header is informational for the LLM; it does not contribute to
    retrieval quality. Keeping it short keeps the token overhead small.
    """
    md = chunk.metadata
    header = f"[{idx + 1}] source={md.source_file} chunk={md.chunk_index}"
    return f"{header}\n{chunk.text}"


def assemble_context(
    chunks: list[RetrievedChunk],
    max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
    separator: str = CHUNK_SEPARATOR,
) -> AssembledContext:
    """Build a :class:`AssembledContext` honoring the token budget.

    Chunks must already be ordered by descending relevance (the reranker or
    the RRF fuser does this). The first chunk is *always* included even if it
    alone exceeds the budget — partial-relevance is better than nothing.
    """
    if max_context_tokens < 1:
        raise ValueError("max_context_tokens must be >= 1")

    tracer = get_tracer(__name__)
    with tracer.start_as_current_span("rag.context.assemble") as span:
        span.set_attribute("rag.context.candidate_count", len(chunks))
        span.set_attribute("rag.context.max_tokens", max_context_tokens)

        if not chunks:
            return AssembledContext(text="", total_tokens=0)

        included: list[RetrievedChunk] = []
        dropped: list[RetrievedChunk] = []
        running_tokens = 0
        sep_tokens = count_tokens(separator)

        for i, chunk in enumerate(chunks):
            rendered = _format_chunk(chunk, len(included))
            chunk_tokens = count_tokens(rendered)
            projected = running_tokens + chunk_tokens + (sep_tokens if included else 0)

            if i == 0 or projected <= max_context_tokens:
                included.append(chunk)
                running_tokens = projected
            else:
                dropped.append(chunk)

        text = separator.join(_format_chunk(c, idx) for idx, c in enumerate(included))
        was_truncated = bool(dropped)

        span.set_attribute("rag.context.included", len(included))
        span.set_attribute("rag.context.dropped", len(dropped))
        span.set_attribute("rag.context.total_tokens", running_tokens)
        span.set_attribute("rag.context.was_truncated", was_truncated)

    log.info(
        "retrieval.assembler.completed",
        included=len(included),
        dropped=len(dropped),
        total_tokens=running_tokens,
        was_truncated=was_truncated,
    )
    return AssembledContext(
        text=text,
        included_chunks=included,
        dropped_chunks=dropped,
        total_tokens=running_tokens,
        was_truncated=was_truncated,
    )
