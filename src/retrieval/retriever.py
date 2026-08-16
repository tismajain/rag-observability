"""Hybrid retrieval: dense (Qdrant) + sparse (BM25) fused with Reciprocal Rank Fusion.

The dense path embeds the query and runs a Qdrant cosine search.
The sparse path scores the query against an in-process BM25 index built by
scrolling all current points out of Qdrant. RRF combines the two rankings
without needing score normalization.

Known limitation: the BM25 index lives in this process's memory and is
rebuilt lazily on first query (or on demand via :meth:`refresh_sparse_index`).
For large corpora the rebuild can take tens of seconds — that is the cost the
spec calls out in the README's "Known limitations" section.
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any, cast

import structlog
from opentelemetry.trace import Status, StatusCode
from pydantic import BaseModel, Field
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue
from rank_bm25 import BM25Okapi

from src.auth.principal import AuthorizationScope
from src.config.settings import settings
from src.ingestion.chunker import ChunkMetadata
from src.ingestion.embedder import EmbedderProtocol
from src.observability.tracer import get_tracer

log = structlog.get_logger(__name__)

DEFAULT_TOP_K = 5
DEFAULT_CANDIDATES_PER_PATH = 20
RRF_K_CONSTANT = 60  # spec value; smoothing constant in RRF formula


class RetrievedChunk(BaseModel):
    """One retrieval result, fused or single-path."""

    chunk_id: str
    text: str
    metadata: ChunkMetadata
    dense_score: float | None = None
    sparse_score: float | None = None
    rrf_score: float
    rank: int = Field(..., ge=0)


def _tokenize(text: str) -> list[str]:
    """Cheap whitespace + lowercase tokenizer for BM25.

    BM25 quality is relatively insensitive to tokenizer sophistication for
    Phase 3 needs. nltk/regex tokenizers can replace this without changing
    callers if Phase 4+ wants finer recall.
    """
    return [t for t in text.lower().split() if t]


class HybridRetriever:
    """Dense + BM25 retrieval combined with Reciprocal Rank Fusion."""

    def __init__(
        self,
        embedder: EmbedderProtocol,
        client: AsyncQdrantClient | None = None,
        collection: str | None = None,
        candidates_per_path: int = DEFAULT_CANDIDATES_PER_PATH,
        rrf_k: int = RRF_K_CONSTANT,
    ) -> None:
        if candidates_per_path < 1:
            raise ValueError("candidates_per_path must be >= 1")
        self._embedder = embedder
        self._client = client or AsyncQdrantClient(
            host=settings.qdrant.host,
            port=settings.qdrant.port,
            prefer_grpc=False,
        )
        self._collection = collection or settings.qdrant.collection_name
        self._candidates_per_path = candidates_per_path
        self._rrf_k = rrf_k
        self._tracer = get_tracer(__name__)

        # BM25 lazily built on first query.
        self._bm25: BM25Okapi | None = None
        self._bm25_chunk_ids: list[str] = []
        self._bm25_payloads: dict[str, dict[str, Any]] = {}
        self._bm25_scoped: dict[
            str, tuple[BM25Okapi | None, list[str], dict[str, dict[str, Any]]]
        ] = {}

    # ------------------------------------------------------------------ public

    async def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        *,
        scope: AuthorizationScope,
    ) -> list[RetrievedChunk]:
        """Run hybrid retrieval. Returns the top-``top_k`` fused chunks."""
        if not query.strip():
            return []
        if top_k < 1:
            raise ValueError("top_k must be >= 1")

        with self._tracer.start_as_current_span("rag.retrieval.hybrid") as span:
            span.set_attribute("rag.retrieval.top_k", top_k)
            span.set_attribute("rag.retrieval.candidates_per_path", self._candidates_per_path)
            span.set_attribute("rag.retrieval.collection", self._collection)
            t0 = time.perf_counter()
            try:
                auth_filter = _authorization_filter(scope)
                dense = await self._dense(query, auth_filter)
                sparse = await self._sparse(query, scope, auth_filter)
                fused = self._fuse(dense, sparse, top_k)
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                raise

            latency_ms = int((time.perf_counter() - t0) * 1000)
            span.set_attribute("rag.retrieval.latency_ms", latency_ms)
            span.set_attribute("rag.retrieval.dense_hits", len(dense))
            span.set_attribute("rag.retrieval.sparse_hits", len(sparse))
            span.set_attribute("rag.retrieval.fused_hits", len(fused))

        log.info(
            "retrieval.hybrid.completed",
            query_len=len(query),
            top_k=top_k,
            dense_hits=len(dense),
            sparse_hits=len(sparse),
            fused_hits=len(fused),
            latency_ms=latency_ms,
        )
        return fused

    async def refresh_sparse_index(self, scope: AuthorizationScope) -> int:
        """Force-rebuild the BM25 index by scrolling Qdrant. Returns doc count."""
        return await self._build_bm25(scope, _authorization_filter(scope))

    def invalidate_sparse_indexes(self) -> None:
        """Drop every scoped BM25 corpus after share, ingestion, or deletion changes."""
        self._bm25_scoped.clear()

    async def aclose(self) -> None:
        await self._client.close()

    # ------------------------------------------------------------------ dense

    async def _dense(self, query: str, auth_filter: Filter) -> list[RetrievedChunk]:
        with self._tracer.start_as_current_span("rag.retrieval.dense") as span:
            vec = (await self._embedder.embed_texts([query]))[0]
            response = await self._client.query_points(
                collection_name=self._collection,
                query=vec,
                limit=self._candidates_per_path,
                with_payload=True,
                with_vectors=False,
                query_filter=auth_filter,
            )
            hits = response.points
            span.set_attribute("rag.retrieval.dense.hits", len(hits))

        out: list[RetrievedChunk] = []
        for rank, hit in enumerate(hits):
            payload = dict(hit.payload or {})
            chunk_id = str(hit.id)
            out.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    text=payload.get("text", ""),
                    metadata=_metadata_from_payload(payload),
                    dense_score=float(hit.score),
                    sparse_score=None,
                    rrf_score=0.0,  # set in fusion
                    rank=rank,
                )
            )
        return out

    # ------------------------------------------------------------------ sparse

    async def _sparse(
        self, query: str, scope: AuthorizationScope, auth_filter: Filter
    ) -> list[RetrievedChunk]:
        with self._tracer.start_as_current_span("rag.retrieval.sparse") as span:
            scoped = self._bm25_scoped.get(scope.fingerprint)
            if scoped is None:
                await self._build_bm25(scope, auth_filter)
                scoped = self._bm25_scoped.get(scope.fingerprint)
            if scoped is None:
                span.set_attribute("rag.retrieval.sparse.hits", 0)
                return []
            bm25, chunk_ids, payloads = scoped
            if not bm25 or not chunk_ids:
                span.set_attribute("rag.retrieval.sparse.hits", 0)
                return []

            scores = bm25.get_scores(_tokenize(query))
            # argsort descending, take top N. We deliberately do NOT filter
            # zero-score hits: with a small corpus BM25 IDF can collapse to
            # zero even for relevant matches; RRF and the reranker handle the
            # final quality ordering.
            ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            ranked_idx = ranked_idx[: self._candidates_per_path]
            span.set_attribute("rag.retrieval.sparse.hits", len(ranked_idx))

        out: list[RetrievedChunk] = []
        for rank, idx in enumerate(ranked_idx):
            chunk_id = chunk_ids[idx]
            payload = payloads[chunk_id]
            out.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    text=payload.get("text", ""),
                    metadata=_metadata_from_payload(payload),
                    dense_score=None,
                    sparse_score=float(scores[idx]),
                    rrf_score=0.0,
                    rank=rank,
                )
            )
        return out

    async def _build_bm25(self, scope: AuthorizationScope, auth_filter: Filter) -> int:
        """Scroll the entire collection and rebuild the in-memory BM25 index."""
        chunk_ids: list[str] = []
        payloads: dict[str, dict[str, Any]] = {}
        tokenized_corpus: list[list[str]] = []

        offset: Any = None
        total = 0
        while True:
            points, offset = await self._client.scroll(
                collection_name=self._collection,
                limit=512,
                offset=offset,
                with_payload=True,
                with_vectors=False,
                scroll_filter=auth_filter,
            )
            if not points:
                break
            for p in points:
                payload = dict(p.payload or {})
                text = payload.get("text", "")
                if not text:
                    continue
                cid = str(p.id)
                chunk_ids.append(cid)
                payloads[cid] = payload
                tokenized_corpus.append(_tokenize(text))
                total += 1
            if offset is None:
                break

        self._bm25_chunk_ids = chunk_ids
        self._bm25_payloads = payloads
        self._bm25 = BM25Okapi(tokenized_corpus) if tokenized_corpus else None
        self._bm25_scoped[scope.fingerprint] = (self._bm25, chunk_ids, payloads)

        log.info("retrieval.sparse.index_built", documents=total, collection=self._collection)
        return total

    # ------------------------------------------------------------------ fuse

    def _fuse(
        self,
        dense: list[RetrievedChunk],
        sparse: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Reciprocal Rank Fusion: ``score(d) = sum 1 / (k + rank_i(d))``.

        Score normalization is not needed — RRF is rank-based. The same chunk
        appearing in both lists contributes one term per list.
        """
        rrf_scores: dict[str, float] = defaultdict(float)
        first_seen: dict[str, RetrievedChunk] = {}

        for hit in dense:
            rrf_scores[hit.chunk_id] += 1.0 / (self._rrf_k + hit.rank + 1)
            first_seen.setdefault(hit.chunk_id, hit)
        for hit in sparse:
            rrf_scores[hit.chunk_id] += 1.0 / (self._rrf_k + hit.rank + 1)
            prev = first_seen.get(hit.chunk_id)
            if prev is None:
                first_seen[hit.chunk_id] = hit
            else:
                # Merge sparse score onto an existing dense-only entry.
                if prev.sparse_score is None and hit.sparse_score is not None:
                    prev.sparse_score = hit.sparse_score

        ranked_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)
        fused: list[RetrievedChunk] = []
        for new_rank, cid in enumerate(ranked_ids[:top_k]):
            base = first_seen[cid]
            fused.append(base.model_copy(update={"rrf_score": rrf_scores[cid], "rank": new_rank}))
        return fused


# ----------------------------------------------------------------- payload --


def _metadata_from_payload(payload: dict[str, Any]) -> ChunkMetadata:
    """Reconstruct a :class:`ChunkMetadata` from a Qdrant payload.

    ``ingested_at`` defaults via :class:`ChunkMetadata`'s default_factory when
    absent, so we only pass it through when the payload actually carries it.
    """
    kwargs: dict[str, Any] = {
        "source_file": payload.get("source_file", "<unknown>"),
        "chunk_index": int(payload.get("chunk_index", 0)),
        "total_chunks": max(1, int(payload.get("total_chunks", 1))),
        "chunking_strategy": payload.get("chunking_strategy", "recursive"),
        "char_start": int(payload.get("char_start", 0)),
        "char_end": int(payload.get("char_end", 0)),
        "token_count": int(payload.get("token_count", 0)),
        "document_id": payload.get("document_id"),
        "owner_user_id": payload.get("owner_user_id"),
        "organization_id": payload.get("organization_id"),
        "visibility": payload.get("visibility", "private"),
    }
    if "ingested_at" in payload:
        kwargs["ingested_at"] = payload["ingested_at"]
    return ChunkMetadata(**kwargs)


def _authorization_filter(scope: AuthorizationScope) -> Filter:
    """Mandatory pre-ranking filter. Legacy/unscoped points never match."""
    must = [
        FieldCondition(key="authorization_ready", match=MatchValue(value=True)),
        FieldCondition(key="searchable", match=MatchValue(value=True)),
    ]
    if scope.can_access_all_documents:
        return Filter(must=cast(Any, must))
    should: list[FieldCondition] = [
        FieldCondition(key="access_subjects", match=MatchValue(value=f"user:{scope.user_id}")),
    ]
    if scope.shared_document_ids:
        should.append(
            FieldCondition(key="document_id", match=MatchAny(any=list(scope.shared_document_ids)))
        )
    return Filter(must=cast(Any, must), should=cast(Any, should))
