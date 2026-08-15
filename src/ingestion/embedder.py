"""Embedding model abstraction.

Wraps the OpenAI embeddings API (``text-embedding-3-small`` by default) behind
an interface that batches automatically, retries with exponential backoff +
jitter on transient failures, and wraps every batch call in an OTel span.

The span attributes use literal names here in Phase 2. Phase 4 will replace
them with constants from :mod:`src.observability.attributes`.
"""

from __future__ import annotations

import time
from typing import Protocol, runtime_checkable

import structlog
from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, RateLimitError
from opentelemetry.trace import Status, StatusCode
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from src.config.settings import settings
from src.observability.tracer import get_tracer

log = structlog.get_logger(__name__)

DEFAULT_BATCH_SIZE = 100


class EmbedderConfigError(RuntimeError):
    """Raised when the embedder is instantiated without the required API key."""


@runtime_checkable
class EmbedderProtocol(Protocol):
    """Structural type for anything that produces query/chunk embeddings."""

    @property
    def model(self) -> str: ...

    @property
    def dimensions(self) -> int: ...

    async def embed_texts(self, texts: list[str]) -> list[list[float]]: ...

    async def aclose(self) -> None: ...


class Embedder:
    """Async batching wrapper around the OpenAI embeddings endpoint."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        batch_size: int = DEFAULT_BATCH_SIZE,
        dimensions: int | None = None,
    ) -> None:
        if settings.openai_api_key is None:
            raise EmbedderConfigError(
                "OPENAI_API_KEY is not set; required for the default embedder."
            )
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")

        self._model = model
        self._batch_size = batch_size
        self._dimensions = dimensions or settings.qdrant.vector_size
        self._client = AsyncOpenAI(
            api_key=settings.openai_api_key.get_secret_value(),
            timeout=settings.llm.timeout_seconds,
        )
        self._tracer = get_tracer(__name__)

    @property
    def model(self) -> str:
        return self._model

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed ``texts``, batching transparently."""
        if not texts:
            return []
        out: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            out.extend(await self._embed_batch(batch))
        return out

    async def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        with self._tracer.start_as_current_span("rag.embedding.batch") as span:
            span.set_attribute("rag.embedding.model", self._model)
            span.set_attribute("rag.embedding.batch_size", len(batch))
            span.set_attribute("rag.embedding.dimensions", self._dimensions)

            t0 = time.perf_counter()
            try:
                vectors = await self._call_with_retry(batch)
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                raise
            latency_ms = int((time.perf_counter() - t0) * 1000)

            total_tokens = sum(len(t) for t in batch) // 4  # rough char/4 heuristic
            span.set_attribute("rag.embedding.latency_ms", latency_ms)
            span.set_attribute("rag.embedding.input_chars", sum(len(t) for t in batch))

            log.info(
                "ingestion.embedder.batch",
                model=self._model,
                batch_size=len(batch),
                latency_ms=latency_ms,
                approx_input_tokens=total_tokens,
            )
            return vectors

    @retry(
        reraise=True,
        stop=stop_after_attempt(settings.llm.max_retries),
        wait=wait_exponential_jitter(initial=1, max=10),
        retry=retry_if_exception_type((APITimeoutError, APIConnectionError, RateLimitError)),
    )
    async def _call_with_retry(self, batch: list[str]) -> list[list[float]]:
        resp = await self._client.embeddings.create(
            model=self._model,
            input=batch,
            dimensions=self._dimensions,
        )
        return [item.embedding for item in resp.data]

    async def aclose(self) -> None:
        await self._client.close()


class MockEmbedder:
    """Deterministic random-vector embedder for keyless / offline runs.

    Same input text → same vector, every time. Vector dimensionality matches
    ``settings.qdrant.vector_size`` so the output is directly usable as a
    Qdrant query vector. Scores produced against real corpora are
    meaningless; pair with sparse retrieval (BM25) for any real signal.

    Selected automatically by the pipeline runner when
    ``LLM__PROVIDER=mock`` so the API can serve ``/query`` without
    ``OPENAI_API_KEY``.
    """

    def __init__(self, dimensions: int | None = None) -> None:
        from src.config.settings import settings as _settings

        self._dimensions = dimensions or _settings.qdrant.vector_size

    @property
    def model(self) -> str:
        return "mock-embedder-1"

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        import random as _random

        out: list[list[float]] = []
        for t in texts:
            rng = _random.Random(hash(t) & 0xFFFFFFFF)
            out.append([rng.gauss(0, 1) for _ in range(self._dimensions)])
        return out

    async def aclose(self) -> None:
        return None
