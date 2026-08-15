"""Cross-encoder reranker.

Uses ``cross-encoder/ms-marco-MiniLM-L-6-v2`` from HuggingFace to rescore the
top hybrid candidates. Cross-encoders see the (query, passage) pair jointly,
so they catch relevance signals a bi-encoder misses — typically a meaningful
nDCG bump for the top few results.

The model and the ``sentence-transformers`` package are *optional* (the
``[reranker]`` extra). If they're not installed:

* The reranker logs a one-time warning.
* :meth:`CrossEncoderReranker.rerank` returns the input unchanged.

This keeps the import surface light by default while letting users opt in.
"""

from __future__ import annotations

import time
from threading import Lock
from typing import TYPE_CHECKING

import structlog

from src.observability.tracer import get_tracer

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

    from src.retrieval.retriever import RetrievedChunk

log = structlog.get_logger(__name__)

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_LOAD_LOCK = Lock()


class CrossEncoderReranker:
    """Optional rerank stage. No-op if sentence-transformers is not installed."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        enabled: bool = True,
    ) -> None:
        self._model_name = model_name
        self._enabled = enabled
        self._model: CrossEncoder | None = None
        self._unavailable_warning_logged = False
        self._tracer = get_tracer(__name__)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _ensure_model(self) -> CrossEncoder | None:
        if self._model is not None:
            return self._model
        with _LOAD_LOCK:
            if self._model is not None:
                return self._model
            try:
                from sentence_transformers import CrossEncoder
            except ImportError:
                if not self._unavailable_warning_logged:
                    log.warning(
                        "retrieval.reranker.unavailable",
                        message=(
                            "sentence-transformers not installed; reranker is a no-op. "
                            'Install with: pip install -e ".[reranker]"'
                        ),
                    )
                    self._unavailable_warning_logged = True
                return None
            log.info("retrieval.reranker.loading_model", model=self._model_name)
            t0 = time.perf_counter()
            self._model = CrossEncoder(self._model_name)
            log.info(
                "retrieval.reranker.model_loaded",
                model=self._model_name,
                load_ms=int((time.perf_counter() - t0) * 1000),
            )
            return self._model

    async def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """Reorder ``candidates`` by cross-encoder relevance. Pure if disabled."""
        if not candidates:
            return []
        if not self._enabled:
            return candidates if top_k is None else candidates[:top_k]

        model = self._ensure_model()
        if model is None:
            # Graceful fallback: return the original ordering, optionally capped.
            return candidates if top_k is None else candidates[:top_k]

        with self._tracer.start_as_current_span("rag.retrieval.rerank") as span:
            span.set_attribute("rag.retrieval.rerank.model", self._model_name)
            span.set_attribute("rag.retrieval.rerank.candidates", len(candidates))

            t0 = time.perf_counter()
            pairs = [(query, c.text) for c in candidates]
            # CrossEncoder.predict is synchronous CPU/GPU work — there's nothing
            # to await. Run inline; the surrounding pipeline is already async.
            scores = model.predict(pairs)
            latency_ms = int((time.perf_counter() - t0) * 1000)
            span.set_attribute("rag.retrieval.rerank.latency_ms", latency_ms)

        scored = sorted(
            zip(candidates, scores, strict=True),
            key=lambda pair: float(pair[1]),
            reverse=True,
        )
        keep = scored if top_k is None else scored[:top_k]
        reranked = [
            chunk.model_copy(update={"rrf_score": float(score), "rank": new_rank})
            for new_rank, (chunk, score) in enumerate(keep)
        ]
        log.info(
            "retrieval.reranker.completed",
            candidates=len(candidates),
            kept=len(reranked),
            latency_ms=latency_ms,
        )
        return reranked
