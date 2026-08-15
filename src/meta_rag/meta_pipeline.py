"""End-to-end meta-RAG pipeline.

Mirrors :class:`src.pipeline.rag_pipeline.RAGPipeline` but for queries over
*trace history* rather than ingested documents:

* uses the meta :class:`HybridRetriever` (collection = meta_collection_name)
* renders the observability-analyst :data:`META_RAG_PROMPT`
* skips defect detection — those checks (LOW_CHUNK_DIVERSITY, etc.) are
  calibrated for the production RAG path and would fire spuriously here
  (every trace doc is sourced "trace://"); the meta layer is a debugging
  surface, not a quality-gated user surface.

Reuses the production :class:`AnthropicClient` / :class:`OpenAIClient` /
:class:`MockClient` — same generation infrastructure, different prompt and
context.
"""

from __future__ import annotations

import time

import structlog
from pydantic import BaseModel, Field

from src.auth.principal import AuthorizationScope
from src.config.settings import settings
from src.generation.llm_client import BaseLLMClient, LLMResponse
from src.generation.prompt_templates import META_RAG_PROMPT
from src.observability.attributes import SpanAttributes
from src.observability.spans import (
    context_assembly_span,
    retrieval_span,
)
from src.retrieval.context_assembler import (
    DEFAULT_MAX_CONTEXT_TOKENS,
    AssembledContext,
    assemble_context,
)
from src.retrieval.retriever import HybridRetriever, RetrievedChunk

log = structlog.get_logger(__name__)


class MetaPipelineResult(BaseModel):
    query_id: str
    answer: str
    llm: LLMResponse
    retrieved_traces: list[RetrievedChunk] = Field(default_factory=list)
    assembled_context: AssembledContext
    total_latency_ms: int = 0


class MetaRAGPipeline:
    """Stateless meta-RAG orchestrator. Inject collaborators, share across requests."""

    def __init__(
        self,
        retriever: HybridRetriever,
        llm: BaseLLMClient,
        max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._max_context_tokens = max_context_tokens

    async def run(
        self,
        query_id: str,
        query: str,
        top_k: int = 8,
        scope: AuthorizationScope | None = None,
    ) -> MetaPipelineResult:
        """Execute the meta pipeline: retrieve traces → assemble → generate.

        Default ``top_k=8`` (vs 5 for production) because trace docs are
        denser in named fields and the analyst answer typically benefits
        from seeing more of the comparable history.
        """
        t0 = time.perf_counter()

        with retrieval_span(query_id, query, top_k) as span:
            if scope is None:
                if settings.auth.enabled:
                    raise ValueError("A verified authorization scope is required")
                scope = AuthorizationScope(
                    settings.auth.disabled_local_user_id,
                    settings.auth.disabled_local_organization_id,
                )
            hits = await self._retriever.retrieve(query, top_k=top_k, scope=scope)
            span.set_attribute(SpanAttributes.RETRIEVAL_FUSED_HITS, len(hits))

        with context_assembly_span(query_id, self._max_context_tokens, len(hits)):
            ctx = assemble_context(hits, max_context_tokens=self._max_context_tokens)

        system, user = META_RAG_PROMPT.render(context=ctx.text, question=query)
        llm_response = await self._llm.generate(system, user)

        total_latency_ms = int((time.perf_counter() - t0) * 1000)
        log.info(
            "meta_rag.pipeline.completed",
            query_id=query_id,
            top_k=top_k,
            retrieved=len(hits),
            total_latency_ms=total_latency_ms,
        )
        return MetaPipelineResult(
            query_id=query_id,
            answer=llm_response.text,
            llm=llm_response,
            retrieved_traces=hits,
            assembled_context=ctx,
            total_latency_ms=total_latency_ms,
        )
