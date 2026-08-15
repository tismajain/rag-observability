"""Span attribute name constants.

Per the spec: no magic strings anywhere. Every span attribute name lives
here, accessed via the :class:`SpanAttributes` namespace. This keeps the
schema discoverable in one place and makes renames safe (one edit + ruff
catches missed call sites).

Naming convention: ``rag.<stage>.<attribute>`` for our domain attributes.
OTel semantic conventions (``http.*``, ``service.*``, ``deployment.*``) are
NOT redefined here — those come from the OTel SDK.
"""

from __future__ import annotations


class SpanAttributes:
    """Constants for every OTel span attribute the pipeline emits."""

    # ----- Query -----
    QUERY_TEXT = "rag.query.text"
    QUERY_ID = "rag.query.id"
    QUERY_TRACE_ID = "rag.query.trace_id"

    # ----- Retrieval (hybrid + per-path) -----
    RETRIEVAL_TOP_K = "rag.retrieval.top_k"
    RETRIEVAL_STRATEGY = "rag.retrieval.strategy"
    RETRIEVAL_CANDIDATES_PER_PATH = "rag.retrieval.candidates_per_path"
    RETRIEVAL_COLLECTION = "rag.retrieval.collection"
    RETRIEVAL_LATENCY_MS = "rag.retrieval.latency_ms"
    RETRIEVAL_DENSE_HITS = "rag.retrieval.dense.hits"
    RETRIEVAL_SPARSE_HITS = "rag.retrieval.sparse.hits"
    RETRIEVAL_FUSED_HITS = "rag.retrieval.fused_hits"
    RETRIEVAL_CHUNK_IDS = "rag.retrieval.chunk_ids"  # JSON array
    RETRIEVAL_SCORES = "rag.retrieval.scores"  # JSON array
    RETRIEVAL_MIN_SCORE = "rag.retrieval.min_score"
    RETRIEVAL_MEAN_SCORE = "rag.retrieval.mean_score"
    RETRIEVAL_RERANKED = "rag.retrieval.reranked"
    RERANK_MODEL = "rag.retrieval.rerank.model"
    RERANK_CANDIDATES = "rag.retrieval.rerank.candidates"
    RERANK_LATENCY_MS = "rag.retrieval.rerank.latency_ms"

    # ----- Embedding (ingestion + query) -----
    EMBEDDING_MODEL = "rag.embedding.model"
    EMBEDDING_BATCH_SIZE = "rag.embedding.batch_size"
    EMBEDDING_DIMENSIONS = "rag.embedding.dimensions"
    EMBEDDING_LATENCY_MS = "rag.embedding.latency_ms"
    EMBEDDING_INPUT_CHARS = "rag.embedding.input_chars"

    # ----- Indexing -----
    INDEXING_COLLECTION = "rag.indexing.collection"
    INDEXING_TOTAL_POINTS = "rag.indexing.total_points"
    INDEXING_LATENCY_MS = "rag.indexing.latency_ms"

    # ----- Context assembly -----
    CONTEXT_TOKEN_COUNT = "rag.context.total_tokens"
    CONTEXT_MAX_TOKENS = "rag.context.max_tokens"
    CONTEXT_TRUNCATED = "rag.context.was_truncated"
    CONTEXT_CHUNKS_INCLUDED = "rag.context.included"
    CONTEXT_CHUNKS_DROPPED = "rag.context.dropped"
    CONTEXT_CANDIDATE_COUNT = "rag.context.candidate_count"

    # ----- Generation -----
    GENERATION_MODEL = "rag.generation.model"
    GENERATION_PROVIDER = "rag.generation.provider"
    GENERATION_PROMPT_TOKENS = "rag.generation.prompt_tokens"
    GENERATION_COMPLETION_TOKENS = "rag.generation.completion_tokens"
    GENERATION_LATENCY_MS = "rag.generation.latency_ms"
    GENERATION_FINISH_REASON = "rag.generation.finish_reason"

    # ----- Defects -----
    DEFECT_COUNT = "rag.defect.count"
    DEFECT_TYPES = "rag.defect.types"  # JSON array
    DEFECT_SEVERITIES = "rag.defect.severities"  # JSON array
    DEFECT_TYPE = "rag.defect.type"  # single, used on defect-emit span
    DEFECT_SEVERITY = "rag.defect.severity"
    DEFECT_DESCRIPTION = "rag.defect.description"

    # ----- Eval -----
    EVAL_FAITHFULNESS = "rag.eval.faithfulness"
    EVAL_CONTEXT_RECALL = "rag.eval.context_recall"
    EVAL_ANSWER_RELEVANCE = "rag.eval.answer_relevance"
    EVAL_QUALITY_GATE = "rag.eval.quality_gate"
    EVAL_LATENCY_MS = "rag.eval.latency_ms"
    EVAL_JUDGE_MODEL = "rag.eval.judge_model"
    EVAL_STATUS = "rag.eval.status"  # "ok" | "skipped" | "failed"
