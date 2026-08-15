"""Trace-indexing CLI.

Pull trace history out of Postgres and upsert it into the meta Qdrant
collection. Use this for one-shot reindexes (e.g. backfilling after a fresh
deploy); the API process also runs a periodic background indexer.

Usage::

    python -m cli.index_traces                          # everything (bounded)
    python -m cli.index_traces --window-minutes 60      # last hour only
    python -m cli.index_traces --limit 5000             # raise the row cap
"""

from __future__ import annotations

import argparse
import asyncio
import sys

import structlog

from src.config.logging import setup_logging
from src.config.settings import settings
from src.ingestion.embedder import Embedder, EmbedderConfigError, MockEmbedder
from src.meta_rag.trace_indexer import init_meta_collection, reindex_recent
from src.observability.tracer import setup_tracing, shutdown_tracing
from src.storage.database import init_engine, shutdown_engine

log = structlog.get_logger("cli.index_traces")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="cli.index_traces",
        description="Index trace history into the meta Qdrant collection.",
    )
    parser.add_argument(
        "--window-minutes",
        type=int,
        default=None,
        help="Only index traces newer than this many minutes (default: all).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Maximum number of traces to fetch in a single run (default: 1000).",
    )
    return parser.parse_args(argv)


async def _run(args: argparse.Namespace) -> int:
    setup_logging()
    setup_tracing()
    init_engine()

    embedder: MockEmbedder | Embedder
    try:
        embedder = MockEmbedder() if settings.llm.provider == "mock" else Embedder()
    except EmbedderConfigError as exc:
        log.error("cli.index_traces.embedder_config_error", error=str(exc))
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    try:
        await init_meta_collection()
        written = await reindex_recent(
            embedder=embedder,
            window_minutes=args.window_minutes,
            limit=args.limit,
        )
    finally:
        await embedder.aclose()
        await shutdown_engine()
        shutdown_tracing()

    log.info("cli.index_traces.complete", written=written)
    print(
        f"Indexed {written} trace(s) into '{settings.qdrant.meta_collection_name}'.",
        file=sys.stderr,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
