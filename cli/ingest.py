"""Ingestion CLI.

Walk a directory (or single file) of supported documents, chunk them, embed
the chunks, and upsert into Qdrant. Idempotent — re-running with the same
source produces stable point IDs.

Usage::

    python -m cli.ingest --dir ./data/documents
    python -m cli.ingest --file ./data/documents/policy.pdf
    python -m cli.ingest --dir ./data/documents --strategy semantic
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import uuid
from pathlib import Path

import structlog

from src.auth.principal import AuthorizationScope
from src.config.logging import setup_logging
from src.config.settings import settings
from src.ingestion.chunker import Chunk, ChunkingStrategy, get_chunker
from src.ingestion.embedder import Embedder, EmbedderConfigError
from src.ingestion.indexer import QdrantIndexer, init_qdrant
from src.ingestion.loader import Document, load_directory, load_file
from src.observability.tracer import setup_tracing, shutdown_tracing

log = structlog.get_logger("cli.ingest")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="cli.ingest",
        description="Ingest documents into the Qdrant vector store.",
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--dir", type=Path, help="Directory of documents to ingest.")
    src.add_argument("--file", type=Path, help="Single document to ingest.")
    parser.add_argument(
        "--strategy",
        choices=["recursive", "fixed", "semantic"],
        default="recursive",
        help="Chunking strategy (default: recursive).",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="When using --dir, do not descend into subdirectories.",
    )
    return parser.parse_args(argv)


async def _run(args: argparse.Namespace) -> int:
    setup_logging()
    setup_tracing()
    if not settings.ingestion_service_user_id or not settings.ingestion_service_organization_id:
        print(
            "ERROR: trusted INGESTION_SERVICE_USER_ID and INGESTION_SERVICE_ORGANIZATION_ID are required",
            file=sys.stderr,
        )
        return 2
    scope = AuthorizationScope(
        user_id=settings.ingestion_service_user_id,
        organization_id=settings.ingestion_service_organization_id,
    )

    docs: list[Document]
    if args.file:
        docs = [load_file(args.file)]
    else:
        docs = load_directory(args.dir, recursive=not args.no_recursive)

    if not docs:
        log.warning("cli.ingest.no_documents", source=str(args.file or args.dir))
        return 0

    try:
        embedder = Embedder()
    except EmbedderConfigError as exc:
        log.error("cli.ingest.embedder_config_error", error=str(exc))
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    strategy: ChunkingStrategy = args.strategy
    chunker = get_chunker(strategy, embedder=embedder if strategy == "semantic" else None)

    await init_qdrant()
    indexer = QdrantIndexer()

    total_chunks = 0
    try:
        for doc in docs:
            chunks: list[Chunk] = await chunker.chunk(doc)
            if not chunks:
                log.warning("cli.ingest.empty_document", source_file=doc.source_file)
                continue
            embeddings = await embedder.embed_texts([c.text for c in chunks])
            await indexer.index(
                chunks,
                embeddings,
                scope=scope,
                document_id=str(uuid.uuid4()),
            )
            total_chunks += len(chunks)
    finally:
        await embedder.aclose()
        await indexer.aclose()
        shutdown_tracing()

    log.info(
        "cli.ingest.complete",
        documents=len(docs),
        chunks=total_chunks,
        strategy=strategy,
    )
    print(
        f"Ingested {len(docs)} document(s) into '{strategy}' chunks "
        f"({total_chunks} total) and upserted to Qdrant.",
        file=sys.stderr,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
