"""Document loading from files and directories.

Supports plain text, Markdown, and PDF. Each loaded source becomes a
:class:`Document` carrying the raw text plus provenance metadata used downstream
by the chunker and indexer.

URL loading is intentionally out of scope for Phase 2.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import structlog
from pydantic import BaseModel, Field

log = structlog.get_logger(__name__)

SUPPORTED_EXTENSIONS: tuple[str, ...] = (".txt", ".md", ".markdown", ".pdf")


class Document(BaseModel):
    """A loaded source document, pre-chunking."""

    source_file: str = Field(..., description="Absolute or repo-relative path of the source.")
    text: str
    loaded_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    file_type: str = Field(..., description="Lowercased extension, e.g. 'pdf', 'md', 'txt'.")
    byte_size: int = Field(..., ge=0)


def _read_pdf(path: Path) -> str:
    # Lazy import keeps pypdf out of the loader's import-time cost when only
    # text/markdown files are processed.
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(p.strip() for p in pages if p.strip())


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def load_file(path: str | Path) -> Document:
    """Load a single file into a :class:`Document`.

    Raises:
        FileNotFoundError: path does not exist.
        ValueError: extension is not in :data:`SUPPORTED_EXTENSIONS`.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Not a file: {p}")

    ext = p.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported extension {ext!r} for {p.name}. "
            f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    text = _read_pdf(p) if ext == ".pdf" else _read_text(p)
    file_type = ext.lstrip(".")

    log.info(
        "ingestion.loader.file_loaded",
        source_file=str(p),
        file_type=file_type,
        char_count=len(text),
        byte_size=p.stat().st_size,
    )
    return Document(
        source_file=str(p),
        text=text,
        file_type=file_type,
        byte_size=p.stat().st_size,
    )


def load_directory(directory: str | Path, recursive: bool = True) -> list[Document]:
    """Load every supported file under ``directory``.

    Args:
        directory: Path to walk.
        recursive: When True, recurse into subdirectories.

    Returns:
        List of :class:`Document`. Empty if no supported files were found.

    Unsupported files are skipped with a warning rather than raising — partial
    ingestion is usually preferable to a hard stop in the middle of a corpus.
    """
    d = Path(directory)
    if not d.is_dir():
        raise NotADirectoryError(f"Not a directory: {d}")

    pattern = "**/*" if recursive else "*"
    docs: list[Document] = []
    skipped: list[str] = []

    for path in sorted(d.glob(pattern)):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            skipped.append(path.name)
            continue
        try:
            docs.append(load_file(path))
        except Exception as exc:  # noqa: BLE001 — broad catch is intentional here
            log.warning(
                "ingestion.loader.file_failed",
                source_file=str(path),
                error=str(exc),
                error_type=type(exc).__name__,
            )

    log.info(
        "ingestion.loader.directory_loaded",
        directory=str(d),
        loaded=len(docs),
        skipped=len(skipped),
        recursive=recursive,
    )
    return docs
