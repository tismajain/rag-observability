# Operations runbook

For operators running this in any environment beyond a laptop.

## Common operations

### Restart just the API after a code change

```bash
docker compose up -d --build api
# OR force a full recreate (picks up env changes that --restart misses):
docker compose up -d --force-recreate api
```

`docker compose restart` does NOT re-read `.env` — use `--force-recreate`.

### Re-index trace history into the meta collection

Runs every 15 minutes automatically; for an immediate refresh:

```bash
docker compose exec api python -m cli.index_traces
```

Bound the window with `--window-minutes` (default: index everything) or raise
the row cap with `--limit` (default: 1000).

### Re-ingest documents

```bash
docker compose exec api python -m cli.ingest --dir /app/data/documents
# or a single file:
docker compose exec api python -m cli.ingest --file /app/data/documents/foo.md
```

Idempotent: same source produces same point IDs, so upserts overwrite cleanly.

### Tail logs

```bash
docker compose logs -f api          # one service
docker compose logs --since=5m      # everything in last 5 min
```

Every log line is JSON; pipe to `jq` for structured filtering:

```bash
docker compose logs --since=5m api | grep -v "^rag_obs_api  | " \
  | grep -E "^\{" | jq 'select(.event | startswith("observability.defect"))'
```

## Key rotation

API keys live in `.env` and are loaded by `pydantic-settings` at import time.
They are not re-read while the process is alive, so rotation requires a
recreate:

```bash
# 1. Generate a new key in the provider console (Anthropic / OpenAI).
# 2. Edit .env with the new value.
# 3. Recreate the api container so the new key is loaded:
docker compose up -d --force-recreate api
# 4. Revoke the old key in the provider console.
```

`docker compose restart api` does NOT pick up new env values — use
`--force-recreate`.

**The dashboard container does not need API keys** — it only talks to the
internal API service.

## Scaling

### Single-machine → multi-instance

The API is stateless. To run multiple replicas:

* Postgres + Qdrant must be reachable from every replica (shared backing
  services, not the local docker compose Postgres).
* The slowapi rate limiter defaults to in-memory counters, which are
  per-replica. Set `RATELIMIT_STORAGE_URL=redis://...` to share state.
* The background meta indexer runs on every replica by default. Options:
  * **Accept it**: upserts are idempotent, the only cost is wasted
    embedding calls (one OpenAI call per trace per replica per cycle).
  * **Gate to one replica**: pick a leader (Postgres advisory lock, k8s
    leader-election sidecar, env-var on a "worker" replica only) and only
    start the loop there.

### Connection pools

Postgres pool defaults: `pool_size=10`, `max_overflow=20`. With N replicas
and P pool size, the DB needs `N × (P + max_overflow)` available connections.
For Postgres 16 default `max_connections=100` and `N=4`, P+max_overflow must
stay ≤ 25 → tune via env vars or hard-code in `storage/database.py`.

Qdrant connections are HTTP — no pool sizing needed. Each retriever holds
one `AsyncQdrantClient` for its lifetime.

### Embedding cost

`text-embedding-3-small` is the dominant per-query cost when reranking is
off (one embed per query) and dominates ingest cost (one embed per chunk).
Knobs:

* Larger chunks → fewer embeds (current default 512 tokens, 64 overlap).
* `OBSERVABILITY__SAMPLE_RATE_FOR_EVAL=0.0` to disable RAGAS judging when
  the meta dashboard is sufficient signal.
* For the meta indexer, pass `--window-minutes` so the periodic job only
  re-embeds recently-changed traces.

## Health probes

`/health` returns:

* `200 status=ok` — all three deps (Postgres, Qdrant, Phoenix) healthy.
* `200 status=degraded` — Phoenix is down, traces buffer locally.
* `503 status=unhealthy` — Postgres or Qdrant is down; serving traffic is
  unsafe.

Use 503 as the readiness probe in Kubernetes / load balancers. Liveness
should be a separate, faster check (e.g. the `/health` endpoint with a
shorter timeout); a slow Qdrant should not kill the pod.

## Common issues

### `/query` returns 503 "Pipeline unavailable: embedder-unavailable"

OpenAI key is missing or invalid. Check `.env`, recreate the api container,
look for `pipeline.startup.embedder_failed` in the logs.

### `/meta/query` returns "no trace docs"

The meta collection (`rag_traces`) is empty. Either:

* No `/query` calls have happened yet (the indexer has nothing to index).
* The periodic indexer hasn't fired (initial delay is 30s after boot, then
  every 15 min). Trigger it manually with `cli.index_traces`.

### `DEFECT_LOW_RETRIEVAL_QUALITY` fires on every query

Threshold is calibrated for cosine similarity in the 0.4–0.7 range. If your
corpus is small or queries don't match topical content, the threshold may
be too aggressive. Lower it:

```bash
# in .env
OBSERVABILITY__DEFECT_SIMILARITY_THRESHOLD=0.3
docker compose up -d --force-recreate api
```

### Rate limit kicks in for legitimate dashboard use

The `/query` limit is `30/minute` per IP. Override:

```bash
# in .env
RATE_LIMIT_QUERY=120/minute
RATE_LIMIT_META_QUERY=60/minute
# or turn it off entirely for trusted networks:
RATE_LIMIT_ENABLED=false
```

### Database migrations out of sync

Apply pending migrations:

```bash
docker compose exec api alembic upgrade head
```

Check current revision:

```bash
docker compose exec api alembic current
```

### Phoenix not receiving traces

* `OBSERVABILITY__PHOENIX_ENDPOINT` must point at the OTLP gRPC port
  (default `:4317`, NOT the UI port `:6006`).
* `setup_tracing()` is called in the API lifespan; CLI scripts call it
  explicitly. If you run a script *without* `setup_tracing()`, spans are
  no-ops by design.
* Check `BatchSpanProcessor` is flushing: there's a 5–30s buffering window.
  Visit `localhost:6006` and look for spans under the `rag-observability`
  service name.

## Shutdown checklist

Before tearing down a production deployment:

* `docker compose down` (preserves named volumes) vs `docker compose down -v`
  (wipes Postgres + Qdrant + Phoenix data).
* The async eval task fires-and-forgets. In a hot shutdown, in-flight evals
  may be cancelled mid-flight; their failures are logged but not retried.
* The meta indexer task is cooperatively cancelled in `pipeline_runner.shutdown()`.

## Document backup and deletion

Back up PostgreSQL, the S3/MinIO bucket, and Qdrant snapshots as one recovery set. Restore
PostgreSQL first because it is the authorization source of truth. Never expose restored Qdrant
points until their document, authorization, searchable, and access-subject payloads match live
database rows. Rebuilding vectors from active objects is safer than adopting unscoped points.

Deletion first makes Qdrant points non-searchable, then tombstones PostgreSQL, removes vectors,
deletes the object, clears chunk rows, and finalizes the tombstone. Retry interrupted deletions;
never manually reactivate a partial deletion. Production S3 should use encryption, versioning,
retention policy, and a workload role restricted to the configured bucket prefix.

The default Compose network keeps PostgreSQL, Redis, Qdrant, and MinIO private. Only
`docker-compose.integration.yml` publishes their ports for local/CI tests. Production additionally
requires TLS, managed secrets, distinct migration/application/worker database roles, Redis HA, and
a durable ingestion worker lease or queue.
