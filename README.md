# RAG Observability

An observable Retrieval-Augmented Generation service built with FastAPI. It combines
authorization-scoped hybrid retrieval, LLM generation, OpenTelemetry traces, heuristic defect
detection, sampled RAGAS evaluation, and a second RAG pipeline over the system's own trace history.

> [!IMPORTANT]
> This repository is a reference implementation, not a turnkey production service. Ingestion and
> evaluation run as in-process background work, the optional reranker is not included in the
> standard API image, and production deployment still requires an external OIDC provider,
> durable job execution, secret management, and normal operational hardening.

## What is implemented

- FastAPI endpoints for document lifecycle, queries, traces, defects, evaluations, and meta-RAG.
- Owner-or-explicit-share document access. The verified identity determines the retrieval scope
  before either dense or BM25 ranking.
- OpenAI embeddings with Qdrant dense search, in-process BM25, and Reciprocal Rank Fusion (RRF).
- Anthropic, OpenAI, and deterministic mock generation providers.
- Optional cross-encoder reranking when the `reranker` extra is installed.
- OpenTelemetry spans exported to Arize Phoenix.
- Synchronous heuristic defect detection and sampled, asynchronous RAGAS evaluation.
- PostgreSQL persistence, MinIO/S3-compatible document storage, and a Streamlit dashboard.
- Meta-RAG over trace, defect, and evaluation history, refreshed periodically by the API process.

## Architecture

```text
upload -> MinIO/S3 -> chunk -> embed -> Qdrant
                    \-> document metadata + access rules -> PostgreSQL

query -> verified principal -> authorized dense + BM25 retrieval -> RRF
      -> optional reranker -> context assembly -> LLM -> defect checks
      -> PostgreSQL history + sampled evaluation
      -> OpenTelemetry -> Phoenix

trace history -> periodic trace indexer -> Qdrant meta collection -> /meta/query
```

Retrieval is fail-closed: only points marked both `authorization_ready=true` and
`searchable=true`, and matching the caller's trusted access subjects, can enter ranking or the
prompt. See [Security](docs/SECURITY.md) for the full trust model.

## Local quickstart

Prerequisites: Docker with Compose, Git, and enough memory for the service images. The mock mode
below avoids calls to Anthropic and OpenAI.

1. Create the local configuration:

   ```bash
   cp .env.example .env
   ```

   Replace the four `replace-with-...` values used by Compose
   (`POSTGRES_PASSWORD`, `REDIS_PASSWORD`, `MINIO_ROOT_USER`, and
   `MINIO_ROOT_PASSWORD`). Then set:

   ```dotenv
   LLM__PROVIDER=mock
   ANTHROPIC_API_KEY=local-placeholder
   AUTH__ENABLED=false
   AUTH__PROTECT_API_DOCS=false
   ```

   `ANTHROPIC_API_KEY` must currently be non-empty because it is a required application setting;
   mock mode does not send it to a provider.

2. Build the images and start the dependencies:

   ```bash
   docker compose build
   docker compose up -d --wait postgres redis qdrant minio phoenix
   ```

3. Apply the database migrations. The bind mount is needed because `alembic.ini` is not copied
   into the current API image:

   ```bash
   docker compose run --rm \
     -v "$PWD/alembic.ini:/app/alembic.ini:ro" \
     api alembic upgrade head
   ```

4. Start the application:

   ```bash
   docker compose up -d --wait api dashboard
   curl http://localhost:8000/health/live
   ```

Open the [dashboard](http://localhost:8501), [Phoenix](http://localhost:6006), or the
[OpenAPI UI](http://localhost:8000/docs). The OpenAPI UI is available in this quickstart because
`AUTH__PROTECT_API_DOCS=false`.

The default Compose network publishes only the application-facing ports:

| Service | Local URL | Purpose |
| --- | --- | --- |
| API | <http://localhost:8000> | FastAPI and OpenAPI |
| Dashboard | <http://localhost:8501> | Trace, defect, evaluation, and query views |
| Phoenix | <http://localhost:6006> | OpenTelemetry trace UI |

PostgreSQL, Redis, Qdrant, and MinIO remain internal to the Compose network. The
`docker-compose.integration.yml` override publishes their ports for host-run integration tests.

## Try the API

With authentication disabled, requests run as one local development user. Upload a supported
`.txt`, `.md`, `.markdown`, or `.pdf` file:

```bash
curl -X POST http://localhost:8000/documents \
  -F "file=@docs/SECURITY.md" \
  -F "title=Security model"
```

The upload returns `202` while ingestion continues in an in-process background task. Poll
`GET /documents` until `ingestion_status` is `completed`, then query it:

```bash
curl -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"query":"How is document access enforced?","top_k":5}'
```

Mock mode uses deterministic local embeddings and echoes the assembled context. For real model
output, select `LLM__PROVIDER=anthropic` or `openai` and provide the matching provider key.

## API

| Method | Path | Permission | Purpose |
| --- | --- | --- | --- |
| `GET` | `/health/live` | public | Process liveness |
| `GET` | `/health/ready` | observability admin | PostgreSQL, Qdrant, and Phoenix status |
| `POST` | `/documents` | document create | Upload and queue ingestion |
| `GET` | `/documents` | document read | List owned or explicitly shared documents |
| `GET/PATCH/DELETE` | `/documents/{id}` | route-specific document permission | Read, rename, or delete a document |
| `POST/DELETE` | `/documents/{id}/shares...` | document share | Manage explicit user shares |
| `POST` | `/query` | query execute | Run the document RAG pipeline |
| `GET` | `/traces`, `/defects`, `/evals` | observability read | Inspect persisted telemetry |
| `GET` | `/evals/summary` | observability read | Aggregate evaluation and quality-gate results |
| `POST` | `/meta/query` | observability read | Query indexed trace history |

When `AUTH__ENABLED=true`, the API verifies JWT signatures and configured issuer/audience claims
through JWKS. Roles and explicit permissions come only from the verified token. Observability
administration does not grant access to document content; that requires the separate
`document:content:admin` permission.

## Retrieval and observability semantics

- Dense and BM25 candidates are independently retrieved within the same authorization scope and
  fused with RRF (`k=60`).
- Cross-encoder reranking is optional. The standard Dockerfile installs `dev` and `eval`, but not
  `reranker`, so Compose preserves the RRF order unless the image is extended.
- Defect detection is synchronous and heuristic. It flags empty/low-quality retrieval, context
  truncation, low source diversity, and low answer/context word overlap.
- RAGAS evaluation is sampled after generation and scheduled in the API process. The current
  `context_recall` value uses the generated answer as its reference and should be interpreted as a
  proxy, not independent answer accuracy.
- Trace, defect, and evaluation persistence is best-effort. The API response may succeed during a
  telemetry database failure.
- The meta indexer starts in the API process, waits 30 seconds, then refreshes a rolling window
  every 15 minutes. Use `python -m cli.index_traces` for an explicit backfill.

## Configuration

Settings are loaded from `.env`; nested fields use `__` (for example,
`QDRANT__COLLECTION_NAME`). Start from [.env.example](.env.example), which documents the complete
set. The most important groups are:

| Group | Examples | Notes |
| --- | --- | --- |
| Identity | `AUTH__ENABLED`, `AUTH__ISSUER`, `AUTH__AUDIENCE`, `AUTH__JWKS_URI` | OIDC is mandatory in staging/production |
| Models | `LLM__PROVIDER`, `LLM__MODEL`, provider API keys | Mock mode is keyless at runtime |
| Storage | `DATABASE_URL`, `QDRANT__*`, `OBJECT_STORAGE__*` | MinIO is the local S3-compatible store |
| Controls | `RATE_LIMIT__*`, `RESOURCES__*` | Redis controls are mandatory outside development |
| Telemetry | `OBSERVABILITY__*` | Phoenix OTLP endpoint, sampling, and thresholds |

Do not commit `.env`. Use workload-scoped credentials and a secrets manager outside local
development.

## Development

The project supports Python 3.11 and 3.12.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

make test-unit
make lint
```

Optional dependency groups:

| Extra | Installs | Use |
| --- | --- | --- |
| `dev` | pytest, Ruff, mypy, IPython | Local development and tests |
| `dashboard` | Streamlit, pandas | Host-run dashboard |
| `reranker` | sentence-transformers and PyTorch | Cross-encoder reranking |
| `eval` | RAGAS and datasets | Sampled evaluation |

The test suite may need a pre-cached `cl100k_base` tokenizer asset in network-restricted
environments. Integration tests also require the published dependency ports:

```bash
docker compose -f docker-compose.yml -f docker-compose.integration.yml up -d --wait \
  postgres redis qdrant minio
pytest -m integration
```

The legacy `make ingest` CLI writes directly to Qdrant and does not complete the database-backed
document lifecycle. Prefer `POST /documents` for retrievable application content.

## Operations and security

- [Architecture](docs/ARCHITECTURE.md) — components, request lifecycle, and design decisions
- [Operations](docs/OPERATIONS.md) — probes, scaling, key rotation, and incident procedures
- [Security](docs/SECURITY.md) — trust boundaries, authorization, and residual risks
- [OIDC/RBAC implementation map](docs/OIDC_RBAC_IMPLEMENTATION_PLAN.md) — requirement-to-code map

Stop the stack with `docker compose down`. Add `-v` only when you intentionally want to delete all
local PostgreSQL, Qdrant, Redis, MinIO, and Phoenix volumes.
