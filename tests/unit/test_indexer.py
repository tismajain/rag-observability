"""Tests for the ingestion indexer's deterministic ID logic.

The :class:`QdrantIndexer` upsert path needs network I/O to verify; what we
*can* verify cheaply is that the point ID derivation is stable across runs
and reacts correctly to content edits. That stability is what makes
re-ingestion idempotent — a regression here would silently duplicate every
chunk on the next ingest.
"""

from __future__ import annotations

import uuid

import pytest

from src.ingestion.indexer import _content_hash, _point_id


def test_content_hash_is_deterministic() -> None:
    h1 = _content_hash("alpha beta gamma")
    h2 = _content_hash("alpha beta gamma")
    assert h1 == h2
    assert len(h1) == 16  # 16-char SHA-256 prefix


def test_content_hash_changes_with_input() -> None:
    assert _content_hash("a") != _content_hash("b")
    # Whitespace matters.
    assert _content_hash("a b") != _content_hash("ab")


def test_point_id_is_deterministic_for_same_inputs() -> None:
    a = _point_id("doc.md", 0, "alpha")
    b = _point_id("doc.md", 0, "alpha")
    assert a == b
    # Output is a valid UUID string.
    uuid.UUID(a)


def test_point_id_changes_when_source_file_changes() -> None:
    a = _point_id("doc1.md", 0, "alpha")
    b = _point_id("doc2.md", 0, "alpha")
    assert a != b


def test_point_id_changes_when_chunk_index_changes() -> None:
    a = _point_id("doc.md", 0, "alpha")
    b = _point_id("doc.md", 1, "alpha")
    assert a != b


def test_point_id_changes_when_content_changes() -> None:
    """Edited chunks must produce new IDs — otherwise the prior version sticks."""
    a = _point_id("doc.md", 0, "alpha")
    b = _point_id("doc.md", 0, "alpha (revised)")
    assert a != b


@pytest.mark.parametrize(
    "source,index,text",
    [
        ("a.md", 0, "x"),
        ("nested/path/file.txt", 12, "long content " * 10),
        ("doc with spaces.pdf", 5, "single"),
    ],
)
def test_point_id_format_is_uuid_string(source: str, index: int, text: str) -> None:
    point_id = _point_id(source, index, text)
    parsed = uuid.UUID(point_id)
    assert str(parsed) == point_id
