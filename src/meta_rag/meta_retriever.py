"""Retrieve trace documents from the meta Qdrant collection.

The meta-RAG layer reuses the production :class:`HybridRetriever` — same
dense + BM25 + RRF stack — just pointed at a different collection. Keeping
this as a thin factory means improvements to retrieval (better fusion,
filters, etc.) flow into both code paths for free.

If a future phase needs meta-specific retrieval behavior (e.g. metadata
filters that don't make sense on the primary collection), this is the file
that grows; the production retriever stays untouched.
"""

from __future__ import annotations

import structlog
from qdrant_client import AsyncQdrantClient

from src.config.settings import settings
from src.ingestion.embedder import EmbedderProtocol
from src.retrieval.retriever import DEFAULT_CANDIDATES_PER_PATH, HybridRetriever

log = structlog.get_logger(__name__)


def build_meta_retriever(
    embedder: EmbedderProtocol,
    client: AsyncQdrantClient | None = None,
    candidates_per_path: int = DEFAULT_CANDIDATES_PER_PATH,
) -> HybridRetriever:
    """Construct a :class:`HybridRetriever` bound to the meta collection.

    A separate Qdrant client instance is allocated so the meta retriever's
    BM25 scroll doesn't share connection state with the primary retriever.
    """
    return HybridRetriever(
        embedder=embedder,
        client=client,
        collection=settings.qdrant.meta_collection_name,
        candidates_per_path=candidates_per_path,
    )
