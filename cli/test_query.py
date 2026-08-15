"""End-to-end retrieval CLI.

Runs hybrid retrieval (+ optional rerank) against the live Qdrant collection
and prints the assembled context plus a per-chunk score breakdown. Useful for
manual relevance checks before the /query API endpoint lands in Phase 5.

Usage::

    python -m cli.test_query --query "What is the on-call rotation?"
    python -m cli.test_query --query "deploy policy" --top-k 3 --no-rerank
    python -m cli.test_query --query "x" --fake-query-embedding   # offline
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys

import structlog

from src.auth.principal import AuthorizationScope
from src.config.logging import setup_logging
from src.config.settings import settings
from src.ingestion.embedder import Embedder, EmbedderConfigError
from src.observability.tracer import setup_tracing, shutdown_tracing
from src.retrieval.context_assembler import AssembledContext, assemble_context
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.retriever import HybridRetriever, RetrievedChunk

log = structlog.get_logger("cli.test_query")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="cli.test_query",
        description="Run a sample retrieval against the live Qdrant collection.",
    )
    parser.add_argument("--query", required=True, help="Natural-language query.")
    parser.add_argument("--top-k", type=int, default=5, help="Final result count (default 5).")
    parser.add_argument(
        "--candidates",
        type=int,
        default=20,
        help="Candidates per retrieval path before fusion (default 20).",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Skip the cross-encoder rerank stage.",
    )
    parser.add_argument(
        "--max-context-tokens",
        type=int,
        default=3000,
        help="Token budget for assembled context (default 3000).",
    )
    parser.add_argument(
        "--fake-query-embedding",
        action="store_true",
        help=(
            "Embed the query with a deterministic random vector instead of "
            "calling OpenAI. Useful when no OpenAI quota is available; "
            "dense scores will be meaningless but BM25 still works."
        ),
    )
    parser.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help="Emit machine-readable JSON instead of the pretty summary.",
    )
    return parser.parse_args(argv)


class _DeterministicFakeEmbedder:
    """Returns a deterministic random vector. Same query → same vector."""

    def __init__(self) -> None:
        self.dimensions = settings.qdrant.vector_size

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            rng = random.Random(hash(t) & 0xFFFFFFFF)
            out.append([rng.gauss(0, 1) for _ in range(self.dimensions)])
        return out

    async def aclose(self) -> None:  # parity with real Embedder
        return None


async def _run(args: argparse.Namespace) -> int:
    setup_logging()
    setup_tracing()

    embedder: object  # real Embedder or fake
    try:
        if args.fake_query_embedding:  # noqa: SIM108 — two different types branch here
            embedder = _DeterministicFakeEmbedder()
        else:
            embedder = Embedder()
    except EmbedderConfigError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    retriever = HybridRetriever(
        embedder=embedder,  # type: ignore[arg-type]
        candidates_per_path=args.candidates,
    )
    reranker = CrossEncoderReranker(enabled=not args.no_rerank)

    try:
        hits = await retriever.retrieve(
            args.query,
            top_k=max(args.top_k, args.candidates),
            scope=AuthorizationScope(
                settings.auth.disabled_local_user_id,
                settings.auth.disabled_local_organization_id,
            ),
        )
        if not args.no_rerank:
            hits = await reranker.rerank(args.query, hits, top_k=args.top_k)
        else:
            hits = hits[: args.top_k]
        context = assemble_context(hits, max_context_tokens=args.max_context_tokens)
    finally:
        await retriever.aclose()
        if hasattr(embedder, "aclose"):
            await embedder.aclose()
        shutdown_tracing()

    if args.as_json:
        payload = {
            "query": args.query,
            "hits": [h.model_dump(mode="json") for h in hits],
            "context": context.model_dump(mode="json"),
        }
        print(json.dumps(payload, indent=2, default=str))
    else:
        _print_summary(args.query, hits, context)
    return 0


def _print_summary(
    query: str,
    hits: list[RetrievedChunk],
    context: AssembledContext,
) -> None:
    print(f"\nQuery: {query}\n")
    print(f"Retrieved {len(hits)} chunks (after rerank if enabled):")
    for h in hits:
        print(
            f"  [#{h.rank}] id={h.chunk_id[:12]}…  "
            f"rrf={h.rrf_score:.4f}  dense={h.dense_score}  sparse={h.sparse_score}  "
            f"source={h.metadata.source_file} chunk={h.metadata.chunk_index} "
            f"tokens={h.metadata.token_count}"
        )
    print(
        f"\nAssembled context: {context.total_tokens} tokens, "
        f"truncated={context.was_truncated}, "
        f"included={len(context.included_chunks)}, "
        f"dropped={len(context.dropped_chunks)}\n"
    )
    print("---- CONTEXT (preview, first 800 chars) ----")
    print(context.text[:800] + ("…" if len(context.text) > 800 else ""))
    print("--------------------------------------------\n")


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(_run(_parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
