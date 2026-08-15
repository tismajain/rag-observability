"""Boundary tests for the three chunking strategies."""

from __future__ import annotations

import pytest

from src.ingestion.chunker import (
    DEFAULT_CHUNK_TOKENS,
    DEFAULT_OVERLAP_TOKENS,
    FixedChunker,
    RecursiveChunker,
    SemanticChunker,
    count_tokens,
    get_chunker,
)
from src.ingestion.loader import Document


def _doc(text: str, source: str = "test.txt") -> Document:
    return Document(source_file=source, text=text, file_type="txt", byte_size=len(text))


# ---------- RecursiveChunker --------------------------------------------------


async def test_recursive_single_chunk_for_short_text() -> None:
    chunker = RecursiveChunker()
    chunks = await chunker.chunk(_doc("Hello world. This is a short doc."))
    assert len(chunks) == 1
    assert chunks[0].metadata.chunking_strategy == "recursive"
    assert chunks[0].metadata.chunk_index == 0
    assert chunks[0].metadata.total_chunks == 1
    assert chunks[0].metadata.token_count > 0


async def test_recursive_respects_chunk_size_bound() -> None:
    chunker = RecursiveChunker(chunk_tokens=64, overlap_tokens=8)
    long_text = (f"Sentence number {i}. " * 5 for i in range(200))
    text = "\n\n".join(long_text)
    chunks = await chunker.chunk(_doc(text))
    assert len(chunks) > 1
    # No chunk should exceed the configured token cap by more than the splitter's
    # own tolerance (separator-related slack); allow +20%.
    for c in chunks:
        assert c.metadata.token_count <= int(64 * 1.2)


async def test_recursive_metadata_contiguous_indices() -> None:
    chunker = RecursiveChunker(chunk_tokens=32, overlap_tokens=4)
    text = "Para one.\n\n" + ("Word " * 200) + "\n\nPara end."
    chunks = await chunker.chunk(_doc(text))
    indices = [c.metadata.chunk_index for c in chunks]
    assert indices == list(range(len(chunks)))
    assert all(c.metadata.total_chunks == len(chunks) for c in chunks)


async def test_recursive_defaults_match_spec() -> None:
    chunker = RecursiveChunker()
    assert chunker._chunk_tokens == DEFAULT_CHUNK_TOKENS == 512
    assert chunker._overlap_tokens == DEFAULT_OVERLAP_TOKENS == 64


# ---------- FixedChunker ------------------------------------------------------


async def test_fixed_chunker_exact_window_size() -> None:
    chunker = FixedChunker(chunk_tokens=50)
    # Build text with > 50 tokens.
    text = " ".join(f"word{i}" for i in range(500))
    chunks = await chunker.chunk(_doc(text))
    assert len(chunks) > 1
    # All chunks except possibly the last should hit exactly the window size.
    for c in chunks[:-1]:
        assert c.metadata.token_count == 50
    assert chunks[-1].metadata.token_count <= 50


async def test_fixed_chunker_no_overlap_in_token_counts() -> None:
    chunker = FixedChunker(chunk_tokens=20)
    text = " ".join(f"tok{i}" for i in range(150))
    chunks = await chunker.chunk(_doc(text))
    total_tokens_in_chunks = sum(c.metadata.token_count for c in chunks)
    # Fixed chunker has no overlap → sum of chunk tokens equals source tokens.
    assert total_tokens_in_chunks == count_tokens(text)


# ---------- SemanticChunker ---------------------------------------------------


class _MockEmbedder:
    """Returns one of two orthogonal vectors per call so similarity drops
    sharply at a controlled boundary."""

    def __init__(self, switch_after: int) -> None:
        self.switch_after = switch_after
        self.dimensions = 4

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [
            [1.0, 0.0, 0.0, 0.0] if i < self.switch_after else [0.0, 1.0, 0.0, 0.0]
            for i, _ in enumerate(texts)
        ]


async def test_semantic_short_text_returns_single_chunk() -> None:
    chunker = SemanticChunker(embedder=_MockEmbedder(switch_after=1))
    chunks = await chunker.chunk(_doc("Just one sentence."))
    assert len(chunks) == 1
    assert chunks[0].metadata.chunking_strategy == "semantic"


async def test_semantic_splits_on_similarity_drop() -> None:
    # 10 sentences. Mock embedder flips orthogonal at window index 4 → boundary.
    text = " ".join(f"Sentence {i}." for i in range(10))
    chunker = SemanticChunker(
        embedder=_MockEmbedder(switch_after=4),
        window=3,
        similarity_breakpoint=0.5,
    )
    chunks = await chunker.chunk(_doc(text))
    assert len(chunks) >= 2  # at least one cut happened
    assert all(c.metadata.chunking_strategy == "semantic" for c in chunks)


# ---------- get_chunker factory ----------------------------------------------


def test_get_chunker_recursive() -> None:
    assert isinstance(get_chunker("recursive"), RecursiveChunker)


def test_get_chunker_fixed() -> None:
    assert isinstance(get_chunker("fixed"), FixedChunker)


def test_get_chunker_semantic_requires_embedder() -> None:
    with pytest.raises(ValueError, match="embedder"):
        get_chunker("semantic")


def test_get_chunker_unknown_strategy() -> None:
    with pytest.raises(ValueError, match="Unknown chunking strategy"):
        get_chunker("magical")  # type: ignore[arg-type]
