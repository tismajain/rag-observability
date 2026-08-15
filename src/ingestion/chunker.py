"""Chunking strategies.

Three strategies are exposed via :func:`get_chunker`:

* ``recursive`` (default) — :class:`RecursiveChunker`. Splits on a hierarchy of
  separators (``\\n\\n``, ``\\n``, ``. ``, `` ``) using a tiktoken-aware length
  function. Best general-purpose default.
* ``fixed`` — :class:`FixedChunker`. Hard 512-token windows, no overlap. Fast,
  predictable, semantically blunt — useful for very large corpora.
* ``semantic`` — :class:`SemanticChunker`. Embeds rolling sentence windows and
  splits on similarity drops. Higher quality, more expensive (one embedding
  batch per document).

Every chunk carries a :class:`ChunkMetadata` block recording where it came from
so retrieval-time filters and defect detection have the provenance they need.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal

import structlog
import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

from src.ingestion.loader import Document

if TYPE_CHECKING:
    from src.ingestion.embedder import Embedder

log = structlog.get_logger(__name__)

ChunkingStrategy = Literal["recursive", "fixed", "semantic"]

DEFAULT_CHUNK_TOKENS = 512
DEFAULT_OVERLAP_TOKENS = 64
SEMANTIC_WINDOW_SENTENCES = 3
SEMANTIC_SIMILARITY_BREAKPOINT = 0.85

# tiktoken's cl100k_base is shared by GPT-4/4o, Claude (close approximation),
# and the text-embedding-3-small tokenizer. Single source of truth for token
# counts across the pipeline.
_ENCODING = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_ENCODING.encode(text))


class ChunkMetadata(BaseModel):
    source_file: str
    chunk_index: int = Field(..., ge=0)
    total_chunks: int = Field(..., ge=1)
    chunking_strategy: ChunkingStrategy
    char_start: int = Field(..., ge=0)
    char_end: int = Field(..., ge=0)
    token_count: int = Field(..., ge=0)
    ingested_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    document_id: str | None = None
    owner_user_id: str | None = None
    organization_id: str | None = None
    visibility: Literal["private", "organization", "shared"] = "private"


class Chunk(BaseModel):
    text: str
    metadata: ChunkMetadata


class BaseChunker(ABC):
    """Common interface. All chunkers are async to allow the semantic chunker to
    call the embedder; sync strategies just don't await anything."""

    strategy: ChunkingStrategy

    @abstractmethod
    async def chunk(self, document: Document) -> list[Chunk]: ...

    def _build_chunks(
        self,
        document: Document,
        text_chunks: list[str],
    ) -> list[Chunk]:
        """Convert raw text chunks into :class:`Chunk` objects with metadata.

        Char offsets are computed by walking the source text and finding each
        chunk's position. This handles overlap correctly because we advance the
        search cursor by only ``len(chunk) - overlap_chars`` per step — but for
        general safety we just search from the previous chunk's start.
        """
        chunks: list[Chunk] = []
        cursor = 0
        for i, text in enumerate(text_chunks):
            start = document.text.find(text, cursor)
            if start == -1:
                # Fallback: text was transformed (rare with recursive splitter).
                start = cursor
            end = start + len(text)
            cursor = max(cursor, start + 1)  # ensure forward progress
            chunks.append(
                Chunk(
                    text=text,
                    metadata=ChunkMetadata(
                        source_file=document.source_file,
                        chunk_index=i,
                        total_chunks=len(text_chunks),
                        chunking_strategy=self.strategy,
                        char_start=start,
                        char_end=end,
                        token_count=count_tokens(text),
                    ),
                )
            )
        return chunks


class RecursiveChunker(BaseChunker):
    """Token-aware recursive splitter — the default."""

    strategy: ChunkingStrategy = "recursive"

    def __init__(
        self,
        chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
        overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    ) -> None:
        self._chunk_tokens = chunk_tokens
        self._overlap_tokens = overlap_tokens
        self._splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=chunk_tokens,
            chunk_overlap=overlap_tokens,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    async def chunk(self, document: Document) -> list[Chunk]:
        pieces = self._splitter.split_text(document.text)
        chunks = self._build_chunks(document, pieces)
        log.info(
            "ingestion.chunker.recursive",
            source_file=document.source_file,
            chunks=len(chunks),
            chunk_tokens=self._chunk_tokens,
            overlap_tokens=self._overlap_tokens,
        )
        return chunks


class FixedChunker(BaseChunker):
    """Fixed-size 512-token windows, no overlap. Token-exact, semantically blunt."""

    strategy: ChunkingStrategy = "fixed"

    def __init__(self, chunk_tokens: int = DEFAULT_CHUNK_TOKENS) -> None:
        self._chunk_tokens = chunk_tokens

    async def chunk(self, document: Document) -> list[Chunk]:
        token_ids = _ENCODING.encode(document.text)
        pieces: list[str] = []
        for i in range(0, len(token_ids), self._chunk_tokens):
            window = token_ids[i : i + self._chunk_tokens]
            pieces.append(_ENCODING.decode(window))
        chunks = self._build_chunks(document, pieces)
        log.info(
            "ingestion.chunker.fixed",
            source_file=document.source_file,
            chunks=len(chunks),
            chunk_tokens=self._chunk_tokens,
        )
        return chunks


# Splits on sentence-final punctuation followed by whitespace, while preserving
# the punctuation in the previous sentence. Good enough without dragging nltk
# in for Phase 2.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'\[])")


def _split_sentences(text: str) -> list[str]:
    sentences = [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]
    return sentences or [text]


def _cosine(a: list[float], b: list[float]) -> float:
    import numpy as np

    av = np.asarray(a, dtype=np.float32)
    bv = np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    if denom == 0.0:
        return 0.0
    return float(np.dot(av, bv) / denom)


class SemanticChunker(BaseChunker):
    """Group sentences by embedding similarity; split when similarity dips.

    Embeds sliding 3-sentence windows once per document, then walks the
    similarity sequence cutting whenever it dips below
    ``similarity_breakpoint`` (default 0.85). Costs one embedding batch per
    document — the spec calls this out as the expensive strategy.
    """

    strategy: ChunkingStrategy = "semantic"

    def __init__(
        self,
        embedder: Embedder,
        window: int = SEMANTIC_WINDOW_SENTENCES,
        similarity_breakpoint: float = SEMANTIC_SIMILARITY_BREAKPOINT,
    ) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        self._embedder = embedder
        self._window = window
        self._breakpoint = similarity_breakpoint

    async def chunk(self, document: Document) -> list[Chunk]:
        sentences = _split_sentences(document.text)
        if len(sentences) <= self._window:
            return self._build_chunks(document, [document.text])

        # Build rolling windows of `window` sentences.
        windows = [
            " ".join(sentences[i : i + self._window])
            for i in range(len(sentences) - self._window + 1)
        ]
        embeddings = await self._embedder.embed_texts(windows)

        # Cut points: indices in `sentences` where the topic shifted.
        cut_points: list[int] = []
        for i in range(len(embeddings) - 1):
            sim = _cosine(embeddings[i], embeddings[i + 1])
            if sim < self._breakpoint:
                # The window at index i ends at sentence i + window - 1, so cut
                # after that sentence.
                cut_points.append(i + self._window)

        pieces: list[str] = []
        prev = 0
        for cut in cut_points:
            if cut <= prev:
                continue
            pieces.append(" ".join(sentences[prev:cut]).strip())
            prev = cut
        tail = " ".join(sentences[prev:]).strip()
        if tail:
            pieces.append(tail)

        chunks = self._build_chunks(document, pieces or [document.text])
        log.info(
            "ingestion.chunker.semantic",
            source_file=document.source_file,
            chunks=len(chunks),
            sentences=len(sentences),
            cut_points=len(cut_points),
            similarity_breakpoint=self._breakpoint,
        )
        return chunks


def get_chunker(
    strategy: ChunkingStrategy = "recursive",
    *,
    embedder: Embedder | None = None,
) -> BaseChunker:
    """Factory. ``embedder`` is required only for ``strategy="semantic"``."""
    if strategy == "recursive":
        return RecursiveChunker()
    if strategy == "fixed":
        return FixedChunker()
    if strategy == "semantic":
        if embedder is None:
            raise ValueError("SemanticChunker requires an embedder")
        return SemanticChunker(embedder=embedder)
    raise ValueError(f"Unknown chunking strategy: {strategy!r}")
