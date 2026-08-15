"""End-to-end RAG pipeline composition.

This module is the *only* place that knows the full retrieve → rerank →
assemble → generate ordering for the main RAG flow. The /query router calls
:meth:`RAGPipeline.run`; the CLI scripts can use the same class. Anything
else that wants to run the pipeline should go through here so the
instrumentation, defect detection, and eval hand-off stay consistent.
"""

from __future__ import annotations

import time

import structlog
from pydantic import BaseModel, Field

from src.auth.principal import AuthorizationScope
from src.config.settings import settings
from src.generation.llm_client import BaseLLMClient, LLMResponse
from src.generation.prompt_templates import RAG_ANSWER_PROMPT
from src.observability.attributes import SpanAttributes
from src.observability.defect_detector import DefectDetector, DefectEvent
from src.observability.spans import (
    context_assembly_span,
    defect_span,
    retrieval_span,
)
from src.retrieval.context_assembler import (
    DEFAULT_MAX_CONTEXT_TOKENS,
    AssembledContext,
    assemble_context,
)
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.retriever import HybridRetriever, RetrievedChunk

log = structlog.get_logger(__name__)


class PipelineResult(BaseModel):
    """Everything the /query endpoint and downstream consumers need from one run."""

    query_id: str
    answer: str
    llm: LLMResponse
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    assembled_context: AssembledContext
    defects: list[DefectEvent] = Field(default_factory=list)
    total_latency_ms: int = 0


class RAGPipeline:
    """Stateless orchestrator. Inject collaborators; share across requests."""

    def __init__(
        self,
        retriever: HybridRetriever,
        reranker: CrossEncoderReranker,
        llm: BaseLLMClient,
        defect_detector: DefectDetector | None = None,
        max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
    ) -> None:
        self._retriever = retriever
        self._reranker = reranker
        self._llm = llm
        self._defect_detector = defect_detector or DefectDetector()
        self._max_context_tokens = max_context_tokens

    async def run(
        self,
        query_id: str,
        query: str,
        top_k: int = 5,
        enable_reranking: bool | None = None,
        scope: AuthorizationScope | None = None,
    ) -> PipelineResult:
        """Execute the full pipeline for one query.

        ``enable_reranking`` overrides the reranker's own enabled flag when
        not None — lets per-request callers (the /query body) toggle it.
        """
        t0 = time.perf_counter()

        # 1. Retrieve
        with retrieval_span(query_id, query, top_k) as span:
            # Pull more candidates than top_k so the reranker has room to work.
            candidates = max(top_k * 4, 20)
            if scope is None:
                if settings.auth.enabled:
                    raise ValueError("A verified authorization scope is required")
                scope = AuthorizationScope(
                    settings.auth.disabled_local_user_id,
                    settings.auth.disabled_local_organization_id,
                )
            raw_hits = await self._retriever.retrieve(query, top_k=candidates, scope=scope)
            span.set_attribute(SpanAttributes.RETRIEVAL_FUSED_HITS, len(raw_hits))

        # 2. Rerank (optional)
        if enable_reranking is False:
            hits = raw_hits[:top_k]
        elif enable_reranking is True or self._reranker.enabled:
            hits = await self._reranker.rerank(query, raw_hits, top_k=top_k)
        else:
            hits = raw_hits[:top_k]

        # 3. Assemble context
        with context_assembly_span(query_id, self._max_context_tokens, len(hits)):
            ctx = assemble_context(hits, max_context_tokens=self._max_context_tokens)

        # 4. Generate
        system, user = RAG_ANSWER_PROMPT.render(context=ctx.text, question=query)
        llm_response = await self._llm.generate(system, user)

        # 5. Defect detection (sync — must finish before response)
        with defect_span(query_id):
            defects = self._defect_detector.detect(
                query_id=query_id,
                retrieved_chunks=hits,
                assembled_context=ctx,
                generated_answer=llm_response.text,
            )

        total_latency_ms = int((time.perf_counter() - t0) * 1000)
        log.info(
            "pipeline.completed",
            query_id=query_id,
            top_k=top_k,
            retrieved=len(hits),
            defects=len(defects),
            total_latency_ms=total_latency_ms,
        )
        return PipelineResult(
            query_id=query_id,
            answer=llm_response.text,
            llm=llm_response,
            retrieved_chunks=hits,
            assembled_context=ctx,
            defects=defects,
            total_latency_ms=total_latency_ms,
        )
