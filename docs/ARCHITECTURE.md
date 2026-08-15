# Architecture

Deep dive into how the RAG observability system fits together. Companion to
the high-level diagram in the [README](../README.md).

## Layers

The codebase is organized as concentric layers; lower layers don't import
upper ones.

| Layer | Package | Responsibility |
|-------|---------|----------------|
| Config | `src.config` | Typed settings (`pydantic-settings`) + structlog setup |
| Storage | `src.storage` | Async SQLAlchemy engine, ORM models, three CRUD stores, Alembic migrations |
| Observability | `src.observability` | OTel tracer setup, span helpers, defect detector, attribute constants |
| Ingestion | `src.ingestion` | Loader (txt/md/pdf), three chunkers, embedder, Qdrant indexer |
| Retrieval | `src.retrieval` | Hybrid retriever (dense + BM25 + RRF), optional cross-encoder reranker, context assembler |
| Generation | `src.generation` | Prompt templates + LLM client (Anthropic / OpenAI / Mock) |
| Evaluation | `src.evaluation` | RAGAS-backed async evaluator + quality gate |
| Pipeline | `src.pipeline` | End-to-end orchestrator + process-level runner singleton |
| Meta-RAG | `src.meta_rag` | Trace indexer, meta retriever, meta pipeline |
| API | `src.api` | FastAPI routers, middleware, rate limiting, error handlers |
| CLI | `cli` | Standalone scripts: `ingest`, `index_traces`, `test_query` |
| Dashboard | `dashboard` | Streamlit pages over the API |

## Request lifecycle: `POST /query`

```
client
  │
  ▼
BodySizeLimitMiddleware  ─── 413 if Content-Length > 1 MiB
  │
  ▼
RequestIDMiddleware      ─── generates X-Request-ID, binds to structlog
  │
  ▼
TimingMiddleware         ─── starts perf timer
  │
  ▼
SlowAPIMiddleware        ─── counts hits per IP; 429 if limit exceeded
  │
  ▼
@limiter.limit("30/min") ─── per-route rate gate
  │
  ▼
query handler
  │
  ├──▶ root_span("rag.query")             [OTel span on every request]
  │      │
  │      ▼
  │   RAGPipeline.run()
  │      │
  │      ├─▶ retrieval_span               [child span]
  │      │     ├─▶ HybridRetriever.retrieve
  │      │     │      ├─ embedder.embed_texts → OpenAI
  │      │     │      ├─ Qdrant.query_points  (dense)
  │      │     │      ├─ BM25 in-process     (sparse)
  │      │     │      └─ RRF fusion
  │      │     └─▶ CrossEncoderReranker.rerank (optional)
  │      │
  │      ├─▶ context_assembly_span
  │      │     └─▶ token-budget greedy packing
  │      │
  │      ├─▶ generation span
  │      │     └─▶ LLMClient.generate → Anthropic
  │      │
  │      └─▶ defect_span
  │            └─▶ DefectDetector.detect (5 checks)
  │
  ├──▶ pipeline_runner.persist_result()   [best-effort: trace + defects to Postgres]
  │
  ├──▶ pipeline_runner.schedule_eval()    [async, sampled, fire-and-forget]
  │      └─▶ Evaluator.evaluate_async
  │            ├─ RAGAS faithfulness / context_recall / answer_relevancy
  │            ├─ quality gate (pass / warn / fail)
  │            └─ persist EvalScore to Postgres
  │
  └──▶ return QueryResponse to client
```

Every span gets exported to Phoenix via OTLP/gRPC by the BatchSpanProcessor.

## Meta-RAG flow

The meta-RAG layer is a *second* RAG instance indexed on the system's own
trace history. Two paths:

1. **Indexing** (writer):
   * `TraceIndexer.reindex()` pulls `Trace + DefectEvent + EvalScore` rows
     from Postgres (joined via `selectin`).
   * `format_trace_document()` renders each trace as a single text block
     with all named fields (query, answer, defects, scores, gate, timestamp).
   * Documents are embedded with the same embedder the prod pipeline uses
     and upserted into a separate Qdrant collection (`rag_traces`).
   * Point IDs are deterministic UUIDv5 from `query_id` → idempotent upsert.
   * Runs every 15 min as a background asyncio task in the API lifespan; can
     also be triggered on demand via `cli.index_traces`.

2. **Querying** (reader):
   * `POST /meta/query` → `MetaRAGPipeline.run()` (retrieve → assemble → generate)
   * Uses `META_RAG_PROMPT` ("You are an AI observability analyst…")
   * Same hybrid retrieval (dense + BM25 + RRF) pointed at `rag_traces`
   * No defect detection or eval scheduling — meta is a debugging surface,
     not a quality-gated user surface

## Key design decisions

### Forward-design pattern (spans before tracer)

Phase 2/3 modules called `tracer.start_as_current_span(...)` even though
Phase 4 hadn't wired the `TracerProvider` yet. OTel returns a no-op tracer
when no provider is set, so spans cost ~nanoseconds and emit nothing. The
instant Phase 4 called `setup_tracing()`, every pre-existing call site lit
up — no module changes needed.

### Idempotent ingestion

Point IDs for both collections are UUIDv5 keyed on stable inputs:

* Primary collection: `(source_file, chunk_index, content_hash)` so an edit
  to a chunk produces a new ID and the old one ages out, but unchanged
  chunks keep their ID across re-ingestion.
* Meta collection: `query_id` alone, since each query has a single trace doc
  that gets re-rendered as defects/evals arrive.

### Three-stage failure tolerance

The `/query` path has three layers of "must not fail":

1. **DB persistence** (best-effort): if Postgres is down, log a warning and
   continue. The response was already computed.
2. **Async eval scheduling** (sampled): RAGAS evaluation runs after the
   response goes out, never on the request-path latency budget. Sampled at
   `OBSERVABILITY__SAMPLE_RATE_FOR_EVAL` so production volume doesn't burn
   the LLM-judge budget.
3. **Defect detection** (sync, in-band): runs inline because its output is
   part of the response, but the five checks are pure functions over the
   pipeline result — no I/O, no failure modes worth handling.

### Mock mode

Set `LLM__PROVIDER=mock` and both the LLM client (`MockClient`) and embedder
(`MockEmbedder`) switch to deterministic stubs. The entire pipeline runs
end-to-end without API keys — useful for offline demos, integration tests,
and verifying the trace/persistence layer in isolation.

### Hybrid retrieval

Dense (Qdrant cosine over text-embedding-3-small) + BM25 (in-process,
rank-bm25) fused via Reciprocal Rank Fusion with k=60. The defect
detector's "low retrieval quality" check prefers dense_score over the RRF
score because RRF tops out near 0.03 and would always trip a 0.65 threshold.

## Module boundaries

* `src.config.settings` — single source of truth for all runtime config.
  Importers do `from src.config.settings import settings`. Never read env
  vars directly elsewhere.
* `src.observability.spans` — all top-level pipeline stages use these
  context managers for uniform span naming + attribute conventions. Modules
  can still call `tracer.start_as_current_span` directly for sub-spans.
* `src.observability.attributes` — `SpanAttributes` namespace constants so
  no magic strings leak into the codebase.
* `src.pipeline.pipeline_runner` — the *only* place that constructs the
  shared `RAGPipeline` and `MetaRAGPipeline`. Routers reach in via the
  module-level `pipeline_runner` singleton; never instantiate pipelines
  per-request.

## Deployment topology

```
                         ┌──────────────────┐
                         │  load balancer   │
                         └────────┬─────────┘
                                  │
              ┌───────────────────┼───────────────────┐
              ▼                   ▼                   ▼
        ┌─────────┐         ┌─────────┐         ┌─────────┐
        │  API    │         │  API    │         │  API    │   ← stateless
        │  pod    │         │  pod    │         │  pod    │     horizontally
        └────┬────┘         └────┬────┘         └────┬────┘     scalable
             │                   │                   │
             └─────────┬─────────┴─────────┬─────────┘
                       │                   │
                       ▼                   ▼
              ┌────────────────┐  ┌────────────────┐
              │   Postgres     │  │   Qdrant       │
              │  (HA, RW)      │  │  (3+ nodes)    │
              └────────────────┘  └────────────────┘
                       ▲                   ▲
                       │                   │
                  trace, defect,      doc + trace
                  eval rows           embeddings
                       │
                       ▼
              ┌────────────────┐
              │  Streamlit     │   ← single instance acceptable
              │  dashboard     │     (read-mostly over API)
              └────────────────┘
                       │
                       ▼
              ┌────────────────┐
              │  Phoenix       │   ← trace ingest + UI
              │  (OTLP gRPC)   │
              └────────────────┘
```

API pods are stateless. Multi-instance deployments need:

* A shared rate-limit backend (slowapi defaults to in-memory; for multi-pod
  swap to Redis via `RATELIMIT_STORAGE_URL`).
* Sticky sessions are NOT required — the trace store + meta collection are
  the system's memory.
* The background meta indexer runs on every pod by default. Either gate it
  to one pod (e.g. a leader-election flag) or accept the redundant work —
  upserts are idempotent so duplicate runs only burn embeddings cost.

## Document storage and ingestion

`src.storage.object_store:ObjectStore` is the provider-neutral byte-store boundary.
`S3ObjectStore` uses local MinIO through a custom endpoint and standard AWS S3 when that endpoint
is omitted. PostgreSQL is authoritative for identity, ownership, shares, opaque object keys,
hashes, lifecycle state, and ingestion jobs. Qdrant and process-local BM25 corpora are derived.

Uploads create a UUID-keyed object and a `queued` job. The worker transitions
`queued -> processing -> completed|failed`. It writes points with `searchable=false`; only after
all batches and chunk records succeed are they enabled. Failure removes partial points and rows.

The bundled worker executes in the API process. Job state survives restarts, but multi-replica
production needs a dedicated worker with PostgreSQL row leases (`FOR UPDATE SKIP LOCKED`) or an
external durable queue before uploads are horizontally scaled.
