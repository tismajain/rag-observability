# RAG Observability

Production-grade Retrieval-Augmented Generation system with a full AI observability
layer: OpenTelemetry tracing into Arize Phoenix, RAGAS-based evaluation, automatic
defect detection, persistent trace history, a Streamlit dashboard, and a meta-RAG
that lets engineers query the system's own behavior in natural language.

> **Build status:** All 10 phases complete. See the [build phases table](#build-phases)
> for what each phase delivered.

## Quickstart

```bash
cp .env.example .env
# Edit .env: set ANTHROPIC_API_KEY and OPENAI_API_KEY

make install          # editable install + dev tools
make dev              # docker compose up + uvicorn --reload
curl http://localhost:8000/health
make ingest           # ingest documents from ./data/documents
```

Then open the dashboard at <http://localhost:8501> and fire a query.

Access points (all on `localhost`):

| Service   | URL                          | Purpose                                                  |
|-----------|------------------------------|----------------------------------------------------------|
| API       | <http://localhost:8000>      | FastAPI: `/query`, `/meta/query`, `/traces`, etc.        |
| Dashboard | <http://localhost:8501>      | Streamlit: traces, defects, eval trends, meta-query      |
| Phoenix   | <http://localhost:6006>      | OTel trace UI — full span tree per request               |
| Qdrant    | <http://localhost:6333>      | Vector store (collections: `rag_documents`, `rag_traces`)|
| Postgres  | `localhost:5432`             | Trace / defect / eval storage                            |

## Architecture

```
                  +------------------+
   user query --> |  FastAPI /query  | --+ OTel spans
                  +------------------+   |
                          |              v
                          v        +-----+--------+
                 +--------+-------+|   Phoenix    |
                 | RAG pipeline   ||   (OTLP)     |
                 |  retrieve →    |+--------------+
                 |  rerank →      |
                 |  assemble →    |     +----------+
                 |  generate →    |---->| Defect   |
                 |  detect        |     | Detector |
                 +-------+--------+     +----+-----+
                         |                   |
            +------------+------------+      |
            |            |            |      |
            v            v            v      v
       +--------+   +-----------+   +---------------+
       | Qdrant |   | LLM       |   |  PostgreSQL   |
       |  docs  |   | Anthropic |   |  traces /     |
       +--------+   +-----------+   |  defects /    |
                                    |  eval scores  |
                                    +-------+-------+
                                            |
                                periodic    v
                                indexer  +---------------+
                                ─────────►|  Meta-RAG     |
                                          |  Qdrant       |
                                          |  trace docs   |
                                          +-------+-------+
                                                  ▲
                                                  │ POST /meta/query
                                          natural-language
                                          questions about
                                          system behavior
```

## API surface

| Method | Path                  | Purpose                                                |
|--------|-----------------------|--------------------------------------------------------|
| GET    | `/health`             | Service identity + Postgres/Qdrant/Phoenix health      |
| POST   | `/query`              | Run the RAG pipeline against `rag_documents`           |
| POST   | `/meta/query`         | Natural-language query over indexed trace history      |
| GET    | `/traces`             | Paginated list of past `/query` interactions           |
| GET    | `/traces/{trace_id}`  | Single trace with its defects + eval scores            |
| GET    | `/defects`            | Paginated defect feed (filters: severity, type, since) |
| GET    | `/evals`              | Paginated eval scores                                  |
| GET    | `/evals/summary`      | Aggregate scores + quality-gate distribution           |

Full OpenAPI schema: <http://localhost:8000/docs>.

## Defect types

The defect detector runs synchronously inside `/query` after generation and attaches
events to the OTel span:

| Defect                        | Severity | Trigger                                                   |
|-------------------------------|----------|-----------------------------------------------------------|
| `DEFECT_EMPTY_RETRIEVAL`      | CRITICAL | Retrieval returned zero chunks                            |
| `DEFECT_LOW_RETRIEVAL_QUALITY`| HIGH     | Best chunk's dense (or sparse) score < threshold          |
| `DEFECT_CONTEXT_TRUNCATED`    | MEDIUM   | Retrieved chunks exceeded the context token budget        |
| `DEFECT_LOW_CHUNK_DIVERSITY`  | LOW      | All retrieved chunks came from a single source document   |
| `DEFECT_HALLUCINATION_SIGNAL` | HIGH     | Generated answer's content-word overlap with context < 15%|

## Build phases

| Phase | Scope                                                           | Status   |
|-------|-----------------------------------------------------------------|----------|
| 1     | Config, logging, Docker Compose, `/health`                      | Complete |
| 2     | Ingestion (loader, chunker, embedder, indexer)                  | Complete |
| 3     | Hybrid retrieval (dense + BM25 + RRF) + reranker + assembler    | Complete |
| 4     | OTel tracing → Phoenix, defect detection, RAGAS evaluation      | Complete |
| 5     | API endpoints (`/query` + generation + pipeline composition)    | Complete |
| 6     | Persistence (SQLAlchemy async + Alembic + 3 CRUD stores)        | Complete |
| 7     | Meta-RAG over trace history (`/meta/query` + periodic indexer)  | Complete |
| 8     | Streamlit dashboard (traces, defects, evals, meta query)        | Complete |
| 9     | Test suites (139 unit + API tests, in-process integration)      | Complete |
| 10    | Production hardening: rate limits, error envelope, prompt-injection, docs sweep | Complete |

## Environment variables

All settings live in `.env`. Nested fields use `__` as the delimiter.

| Variable                                       | Required | Default                          | Notes                                            |
|------------------------------------------------|:--------:|----------------------------------|--------------------------------------------------|
| `ENVIRONMENT`                                  |          | `development`                    | `development` / `staging` / `production`         |
| `LOG_LEVEL`                                    |          | `INFO`                           |                                                  |
| `ANTHROPIC_API_KEY`                            | yes      | -                                | App fails fast at import if missing              |
| `OPENAI_API_KEY`                               |          | -                                | Required for embeddings + RAGAS judge            |
| `DATABASE_URL`                                 |          | `postgresql+asyncpg://...`       | asyncpg URL                                      |
| `QDRANT__HOST`                                 |          | `localhost`                      |                                                  |
| `QDRANT__PORT`                                 |          | `6333`                           |                                                  |
| `QDRANT__COLLECTION_NAME`                      |          | `rag_documents`                  | Document chunks                                  |
| `QDRANT__META_COLLECTION_NAME`                 |          | `rag_traces`                     | Meta-RAG trace index                             |
| `QDRANT__VECTOR_SIZE`                          |          | `1536`                           | Must match embedding model                       |
| `LLM__PROVIDER`                                |          | `anthropic`                      | `anthropic`, `openai`, or `mock` (keyless echo)  |
| `LLM__MODEL`                                   |          | `claude-sonnet-4-6`              |                                                  |
| `LLM__TEMPERATURE`                             |          | `0.0`                            |                                                  |
| `LLM__MAX_TOKENS`                              |          | `2048`                           |                                                  |
| `LLM__TIMEOUT_SECONDS`                         |          | `30`                             |                                                  |
| `LLM__MAX_RETRIES`                             |          | `3`                              |                                                  |
| `OBSERVABILITY__PHOENIX_ENDPOINT`              |          | `http://localhost:4317`          | OTLP gRPC                                        |
| `OBSERVABILITY__ENABLE_CONSOLE_EXPORTER`       |          | `false`                          | Local debug only                                 |
| `OBSERVABILITY__SAMPLE_RATE_FOR_EVAL`          |          | `0.2`                            | Fraction of queries evaluated by RAGAS           |
| `OBSERVABILITY__DEFECT_SIMILARITY_THRESHOLD`   |          | `0.65`                           | Below this triggers LOW_RETRIEVAL_QUALITY defect |
| `RATE_LIMIT_ENABLED`                           |          | `true`                           | Set `false` to disable per-IP rate limits        |
| `RATE_LIMIT_QUERY`                             |          | `30/minute`                      | slowapi limit string for `/query`                |
| `RATE_LIMIT_META_QUERY`                        |          | `20/minute`                      | slowapi limit string for `/meta/query`           |
| `MAX_BODY_BYTES`                               |          | `1048576`                        | Max request body size (1 MiB default)            |

## Make targets

```
make install    # editable install + dev tools
make dev        # docker compose up -d && uvicorn --reload
make ingest     # python -m cli.ingest --dir ./data/documents
make test       # full test suite
make test-unit  # unit tests only
make lint       # ruff + mypy
make migrate    # alembic upgrade head
make down       # stop docker services
make clean      # stop and wipe volumes
```

## Optional extras

Heavy deps live behind `pip install -e ".[<extra>]"` so the base install stays light:

| Extra        | Brings in                                | Use when                                 |
|--------------|------------------------------------------|------------------------------------------|
| `dev`        | pytest, ruff, mypy, ipython              | local development                        |
| `dashboard`  | streamlit, pandas                        | running the Streamlit container          |
| `reranker`   | sentence-transformers (~2GB torch)       | enabling cross-encoder reranking         |
| `eval`       | ragas, datasets (langchain stack)        | enabling in-process RAGAS evaluation     |

## Trace indexing for meta-RAG

The meta-RAG layer indexes `Trace + DefectEvent + EvalScore` rows into a separate
Qdrant collection (`rag_traces`). The API process runs a background indexer every
15 minutes; for ad-hoc reindexing run:

```bash
python -m cli.index_traces                          # full reindex (bounded by --limit)
python -m cli.index_traces --window-minutes 60      # last hour only
docker compose exec api python -m cli.index_traces  # inside the container
```

Then ask the system about itself: `POST /meta/query`, or open the **Meta Query**
page in the dashboard.

## Documentation

* [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — layer breakdown, request lifecycle, meta-RAG flow, design decisions, deployment topology.
* [docs/OPERATIONS.md](docs/OPERATIONS.md) — runbook: key rotation, scaling, common issues, health probes, shutdown checklist.
* [docs/SECURITY.md](docs/SECURITY.md) — threat model, prompt-injection defenses, what's enforced and what isn't.

## Production hardening (Phase 10)

* Provider-neutral OIDC, typed principals, explicit permissions, and a separate document-content-admin permission.
* Redis-backed per-user/per-IP/per-endpoint, concurrency, and model-budget controls in staging/production; memory-only controls are restricted to development/tests.
* Mandatory pre-ranking Qdrant and scope-specific BM25 authorization filters; legacy unscoped vectors are quarantined.
* 1 MiB request-body cap before Pydantic parsing, configured under `RESOURCES__MAX_BODY_BYTES`.
* Centralized error handler returns structured `{error, detail, request_id}` envelopes — no stack traces leak to clients.
* `RAG_ANSWER_PROMPT` hardened against prompt-injection from ingested documents; see [docs/SECURITY.md](docs/SECURITY.md).
* The repository-specific requirement map and product-decision boundaries are in [docs/OIDC_RBAC_IMPLEMENTATION_PLAN.md](docs/OIDC_RBAC_IMPLEMENTATION_PLAN.md).

## Notes

- Docker Compose requires explicit PostgreSQL and Redis passwords and does not
  publish database/vector-store ports by default.
- `make ingest` requires `OPENAI_API_KEY` to be set (for embeddings).
- The Dockerfile starts uvicorn with `--loop asyncio` rather than the default
  uvloop — RAGAS's `nest_asyncio` patching cannot patch uvloop.
- API keys committed to `.env` should be rotated periodically; conversation logs
  and screenshots can leak them inadvertently.
