"""Process-level pipeline lifecycle.

The /query router needs *one* :class:`RAGPipeline` shared across requests
(the embedder + Qdrant client + reranker model all hold resources). This
module owns construction and teardown so the FastAPI lifespan can wire it
without touching the pipeline internals.

It also fires the async eval task. Per the spec, eval runs after the
response goes out — never on the request-path latency budget.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import structlog

from src.auth.principal import Principal
from src.config.settings import settings
from src.evaluation.evaluator import EvalResult, Evaluator
from src.generation.llm_client import LLMClientConfigError, get_llm_client
from src.ingestion.embedder import Embedder, EmbedderConfigError, MockEmbedder
from src.meta_rag.meta_pipeline import MetaRAGPipeline
from src.meta_rag.meta_retriever import build_meta_retriever
from src.meta_rag.trace_indexer import TraceIndexer, init_meta_collection
from src.pipeline.rag_pipeline import PipelineResult, RAGPipeline
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.retriever import HybridRetriever
from src.storage import defect_store, eval_store, trace_store
from src.storage.database import session_scope

if TYPE_CHECKING:
    from src.generation.llm_client import BaseLLMClient
    from src.ingestion.embedder import EmbedderProtocol

log = structlog.get_logger(__name__)

META_INDEXER_INTERVAL_SECONDS = 15 * 60  # 15-minute cadence per spec
META_INDEXER_WINDOW_MINUTES = 30  # overlap the cadence to be safe
META_INDEXER_INITIAL_DELAY_SECONDS = 30  # let app finish booting first


class PipelineRunner:
    """Holds long-lived collaborators + the assembled :class:`RAGPipeline`."""

    def __init__(self) -> None:
        self._embedder: EmbedderProtocol | None = None
        self._retriever: HybridRetriever | None = None
        self._reranker: CrossEncoderReranker | None = None
        self._llm: BaseLLMClient | None = None
        self._pipeline: RAGPipeline | None = None
        self._evaluator: Evaluator | None = None
        self._build_error: str | None = None
        # Meta-RAG collaborators (Phase 7).
        self._meta_retriever: HybridRetriever | None = None
        self._meta_pipeline: MetaRAGPipeline | None = None
        self._meta_indexer: TraceIndexer | None = None
        self._meta_indexer_task: asyncio.Task[None] | None = None

    # -------------------------------------------------------------- lifecycle

    async def startup(self) -> None:
        """Construct collaborators. Tolerant of missing API keys.

        When ``LLM__PROVIDER=mock`` we also use the :class:`MockEmbedder` so
        the whole pipeline runs keyless. Otherwise we use the real OpenAI
        embedder and fail-fast if its key is missing.
        """
        try:
            if settings.llm.provider == "mock":
                self._embedder = MockEmbedder()
            else:
                self._embedder = Embedder()
        except EmbedderConfigError as exc:
            self._build_error = f"embedder-unavailable: {exc}"
            log.error("pipeline.startup.embedder_failed", error=str(exc))
            return

        try:
            self._llm = get_llm_client()
        except LLMClientConfigError as exc:
            self._build_error = f"llm-unavailable: {exc}"
            log.error("pipeline.startup.llm_failed", error=str(exc))
            return

        self._retriever = HybridRetriever(embedder=self._embedder)
        self._reranker = CrossEncoderReranker()
        self._pipeline = RAGPipeline(
            retriever=self._retriever,
            reranker=self._reranker,
            llm=self._llm,
        )
        self._evaluator = Evaluator()

        # Meta-RAG wiring. Shares the embedder + LLM with the production
        # pipeline — there's no reason to spin a second copy of either.
        self._meta_retriever = build_meta_retriever(embedder=self._embedder)
        self._meta_pipeline = MetaRAGPipeline(
            retriever=self._meta_retriever,
            llm=self._llm,
        )
        self._meta_indexer = TraceIndexer(embedder=self._embedder)
        try:
            await init_meta_collection()
        except Exception as exc:  # noqa: BLE001 — don't crash boot on Qdrant blip
            log.warning(
                "pipeline.startup.meta_collection_init_failed",
                error=str(exc),
                error_type=type(exc).__name__,
            )
        self._meta_indexer_task = asyncio.create_task(self._meta_indexer_loop())

        log.info(
            "pipeline.startup.ready",
            llm_provider=self._llm.provider,
            llm_model=self._llm.model,
            embedder_model=self._embedder.model,
            meta_indexer_scheduled=True,
        )

    async def shutdown(self) -> None:
        if self._meta_indexer_task is not None:
            self._meta_indexer_task.cancel()
            import contextlib

            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._meta_indexer_task
            self._meta_indexer_task = None
        if self._meta_indexer is not None:
            await self._meta_indexer.aclose()
        if self._meta_retriever is not None:
            await self._meta_retriever.aclose()
        if self._retriever:
            await self._retriever.aclose()
        if self._embedder:
            await self._embedder.aclose()
        if self._llm:
            await self._llm.aclose()
        self._pipeline = None
        self._meta_pipeline = None
        log.info("pipeline.shutdown.complete")

    # ------------------------------------------------------------- accessors

    @property
    def is_ready(self) -> bool:
        return self._pipeline is not None

    @property
    def build_error(self) -> str | None:
        return self._build_error

    @property
    def pipeline(self) -> RAGPipeline:
        if self._pipeline is None:
            raise RuntimeError(f"Pipeline not ready: {self._build_error}")
        return self._pipeline

    @property
    def meta_pipeline(self) -> MetaRAGPipeline:
        if self._meta_pipeline is None:
            raise RuntimeError(f"Meta pipeline not ready: {self._build_error}")
        return self._meta_pipeline

    # --------------------------------------------------------- meta indexer --

    async def _meta_indexer_loop(self) -> None:
        """Periodic background task: reindex the rolling trace window.

        Runs forever until cancelled by :meth:`shutdown`. Per-iteration
        errors are logged and swallowed so a transient DB or Qdrant outage
        doesn't terminate the loop.
        """
        await asyncio.sleep(META_INDEXER_INITIAL_DELAY_SECONDS)
        log.info(
            "meta_rag.indexer.loop.started",
            interval_seconds=META_INDEXER_INTERVAL_SECONDS,
            window_minutes=META_INDEXER_WINDOW_MINUTES,
        )
        while True:
            try:
                if self._meta_indexer is not None:
                    from datetime import datetime, timedelta

                    since = datetime.utcnow() - timedelta(minutes=META_INDEXER_WINDOW_MINUTES)
                    written = await self._meta_indexer.reindex(since=since)
                    log.info("meta_rag.indexer.loop.tick", written=written)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 — survive transient failures
                log.warning(
                    "meta_rag.indexer.loop.tick_failed",
                    error=str(exc),
                    error_type=type(exc).__name__,
                )
            try:
                await asyncio.sleep(META_INDEXER_INTERVAL_SECONDS)
            except asyncio.CancelledError:
                raise

    # ----------------------------------------------------------- persistence

    async def persist_result(
        self,
        result: PipelineResult,
        *,
        query_text: str,
        trace_id: str,
        top_k: int,
        principal: Principal,
    ) -> None:
        """Write the Trace + DefectEvent rows. Tolerant of DB failures.

        The /query response has *already* been computed when this runs; if
        Postgres is briefly down we log and move on rather than fail the
        request the user already received.
        """
        try:
            async with session_scope() as session:
                await trace_store.create_trace(
                    session,
                    query_id=result.query_id,
                    trace_id=trace_id,
                    query_text=query_text,
                    answer_text=result.answer,
                    context_truncated=result.assembled_context.was_truncated,
                    latency_ms=result.total_latency_ms,
                    top_k=top_k,
                    chunks_retrieved=len(result.retrieved_chunks),
                    model_used=result.llm.model,
                    user_id=principal.subject,
                    organization_id=principal.organization_id,
                )
                if result.defects:
                    await defect_store.bulk_create_defects(
                        session,
                        [
                            {
                                "id": d.defect_id,
                                "query_id": d.query_id,
                                "trace_id": d.trace_id,
                                "defect_type": d.defect_type.value,
                                "severity": d.severity.value,
                                "description": d.description,
                                "event_metadata": d.metadata,
                                "user_id": principal.subject,
                                "organization_id": principal.organization_id,
                            }
                            for d in result.defects
                        ],
                    )
            log.info(
                "pipeline.persist.completed",
                query_id=result.query_id,
                defects_written=len(result.defects),
            )
        except Exception as exc:  # noqa: BLE001 — persistence is best-effort
            log.warning(
                "pipeline.persist.failed",
                query_id=result.query_id,
                error=str(exc),
                error_type=type(exc).__name__,
            )

    # --------------------------------------------------------- async eval --

    def schedule_eval(
        self,
        result: PipelineResult,
        query: str,
        trace_id: str,
        principal: Principal,
    ) -> bool:
        """Fire-and-forget eval task. Returns whether the task was scheduled.

        We sample at ``settings.observability.sample_rate_for_eval`` so a
        production-volume system does not pay LLM-judge cost on every query.
        """
        if self._evaluator is None or not result.assembled_context.text:
            return False
        evaluator = self._evaluator  # capture non-None binding for the closure

        import random

        if random.random() >= settings.observability.sample_rate_for_eval:
            return False

        async def _eval_task() -> None:
            try:
                ev: EvalResult = await evaluator.evaluate_async(
                    query_id=result.query_id,
                    trace_id=trace_id,
                    question=query,
                    contexts=[c.text for c in result.assembled_context.included_chunks],
                    answer=result.answer,
                )
                log.info(
                    "pipeline.eval.completed",
                    query_id=result.query_id,
                    status=ev.status,
                    gate=ev.quality_gate_result,
                )
                await _persist_eval(ev, principal)
            except Exception as exc:  # noqa: BLE001 — eval must never crash app
                log.warning(
                    "pipeline.eval.task_error",
                    query_id=result.query_id,
                    error=str(exc),
                    error_type=type(exc).__name__,
                )

        asyncio.create_task(_eval_task())
        return True


async def _persist_eval(ev: EvalResult, principal: Principal) -> None:
    """Best-effort eval persistence; same tolerance policy as trace/defect writes."""
    try:
        async with session_scope() as session:
            await eval_store.create_eval(
                session,
                query_id=ev.query_id,
                trace_id=ev.trace_id,
                faithfulness=ev.faithfulness,
                context_recall=ev.context_recall,
                answer_relevancy=ev.answer_relevancy,
                quality_gate_result=ev.quality_gate_result,
                evaluation_latency_ms=ev.evaluation_latency_ms,
                judge_model=ev.judge_model,
                status=ev.status,
                error=ev.error,
                user_id=principal.subject,
                organization_id=principal.organization_id,
            )
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "pipeline.eval.persist_failed",
            query_id=ev.query_id,
            error=str(exc),
            error_type=type(exc).__name__,
        )


# Module-level singleton consumed by the FastAPI app.
pipeline_runner = PipelineRunner()
