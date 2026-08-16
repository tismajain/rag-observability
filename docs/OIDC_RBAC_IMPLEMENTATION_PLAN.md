# OIDC authentication, RBAC, and document isolation implementation plan

This plan maps the requested security specification to this repository as it existed before
implementation. The user-provided requirements are authoritative; no separate attached
specification file was present in the workspace.

## Architecture conflicts found before modification

1. `src/config/settings.py:Settings` has no authentication, authorization, Redis, proxy, or
   resource-control configuration and constructs a global settings object at import time.
2. `src/api/main.py:app` exposes OpenAPI, Swagger, and ReDoc unconditionally. All routers except
   rate-limited query endpoints are anonymous.
3. `src/api/routers/health.py:health` exposes detailed dependency state and exception text at the
   anonymous `/health` endpoint; there is no liveness/readiness split.
4. `src/api/rate_limit.py` uses SlowAPI's process-local storage and only keys two endpoints by IP.
   It cannot meet distributed production limits, per-user limits, concurrency limits, or LLM
   budgets.
5. There is no document HTTP router, document ORM model, ingestion-job model, sharing model,
   download/update/delete workflow, application cache, or task queue. Existing ingestion is a
   trusted CLI (`cli/ingest.py`) that writes chunks directly to Qdrant.
6. `src/ingestion/indexer.py:QdrantIndexer` stores path-derived deterministic point IDs and no
   server-derived owner, organization, visibility, or share payload.
7. `src/retrieval/retriever.py:HybridRetriever` performs unfiltered dense search. Its single global
   BM25 index is built by scrolling the entire collection, so unauthorized chunks enter the
   candidate corpus before ranking.
8. `src/pipeline/rag_pipeline.py:RAGPipeline.run`, `PipelineRunner.persist_result`, async evals,
   and the meta-indexer do not accept or retain a trusted principal/scope.
9. `src/storage/models.py` contains only observability tables. They lack user/organization identity,
   visibility constraints, share records, and PostgreSQL RLS. Legacy rows therefore have no owner.
10. `dashboard/api_client.py` has no OIDC flow or bearer propagation. Streamlit pages display API
    error bodies directly and observability content without authorization.
11. OpenTelemetry spans and persisted traces contain raw query/context-related content. Tokens are
    not deliberately captured today, but no centralized secret/claim redaction exists.
12. Docker Compose publishes PostgreSQL and Qdrant ports, uses default database credentials,
    floating image tags, and has no Redis service.
13. The workspace's `.git` directory is empty, so pre-existing changes cannot be distinguished with
    Git. Changes must be limited to files required by this plan.

## Requirement-to-code map and phases

### 1. Authentication configuration and startup validation

- Extend `src/config/settings.py` with provider-neutral `AuthSettings`, `RateLimitSettings`, and
  `ResourceControlSettings`; validate issuer, audience, HTTPS JWKS URI, explicitly pinned
  algorithms, claim names, production Redis requirements, and docs policy.
- Add examples to `.env.example`, `README.md`, and `docs/SECURITY.md`.
- Invoke explicit startup validation from `src/api/main.py:lifespan` before external services.

### 2. JWT/JWKS verification and typed principals

- Add `src/auth/principal.py` for immutable typed principals, roles, permissions, and authorization
  scopes.
- Add `src/auth/jwks.py` for bounded-timeout JWKS retrieval, cache TTL, unknown-`kid` refresh,
  stale-key failure behavior, and rotation.
- Add `src/auth/dependencies.py` for strict Bearer parsing and PyJWT verification using only the
  configured algorithm allow-list. Never use the JWT header to select an allowed algorithm.
- Add `src/security/redaction.py` and apply it to error/log boundaries.

### 3. Router RBAC and error envelopes

- Add router dependencies to `query`, `meta`, `traces`, `defects`, and `evals` using explicit
  permissions. Observability administration is distinct from document-content access.
- Split `src/api/routers/health.py` into minimal anonymous `/health/live` and admin-only
  `/health/ready`; retain `/health` as a minimal compatibility liveness alias.
- Configure `src/api/main.py` so docs/OpenAPI are all disabled or all protected consistently.
- Preserve `{error, detail, request_id}` in `src/api/error_handlers.py`, including auth and limits.

### 4. Identity, persistence, document authorization, and migration

- Add organization, user, document, document-share, ingestion-job, and chunk identity models in
  `src/storage/models.py`, with non-enumerable UUID identifiers and visibility constraints.
- Add a migration under `src/storage/migrations/versions/` for identity columns, constraints,
  indexes, legacy-row quarantine, session-context functions, and PostgreSQL RLS policies.
- Add `src/storage/document_store.py` with access predicates derived solely from a verified
  principal and trusted DB rows. Direct-ID misses and forbidden resources both return 404.
- Pass principal identity through query persistence, eval tasks, and meta-indexing. Legacy
  observability records remain unavailable to non-admin users unless explicitly adopted.

### 5. Retrieval isolation and Streamlit OIDC

- Extend chunk metadata and Qdrant payloads with server-derived `document_id`, `owner_user_id`,
  `organization_id`, `visibility`, and share subjects.
- Add mandatory Qdrant filters to dense `query_points` and filtered `scroll` calls. Build/cache a
  separate BM25 corpus per authorization-scope fingerprint; never post-filter ranked candidates.
- Thread the scope through `RAGPipeline`, meta-RAG, background tasks, and response serialization.
- Add provider-neutral Streamlit OIDC configuration/session handling and inject the access token
  only into outbound Authorization headers; never display or log it.

### 6. Rate limiting and resource controls

- Replace direct SlowAPI use with a limiter abstraction. Permit in-memory state only in development
  and tests; require Redis in staging/production.
- Enforce per-user, per-IP, and per-endpoint quotas; bounded per-user/global concurrency; request
  size; retrieval/top-k limits; and configured LLM input/output/daily-cost budgets.
- Use trusted proxy headers only when explicitly configured.

### 7. Infrastructure hardening

- Add an authenticated Redis service and production-oriented health checks to Compose.
- Remove public database/vector-store ports from the default stack, replace default credentials
  with required secrets, pin image versions/digests where a product-approved version is known,
  add read-only filesystems/tmpfs/capability drops, and update liveness probes.
- Product decision: deployment-approved image digests, ingress/TLS topology, secret manager, Redis
  HA topology, and PostgreSQL migration/application roles cannot be safely selected in-repository.

### 8. Tests and verification

- Generate ephemeral RSA keys and mock JWKS HTTP responses for issuer/audience/expiry/algorithm,
  rotation, outage, cache, malformed-token, and redaction tests.
- Add RBAC matrix, docs/readiness, disabled-auth, distributed/in-memory limiter, request/concurrency/
  LLM budget, persistence/legacy, Streamlit-header, RLS SQL, Qdrant-filter, BM25-corpus, direct-ID,
  cache-key, background-task, and meta-RAG isolation tests.
- Run `ruff format --check`, `ruff check`, strict `mypy`, Alembic upgrade/downgrade/upgrade against
  PostgreSQL when available, and the complete pytest suite. Do not report unavailable external
  verification as passed.

## Product decisions that remain explicit boundaries

- Which token claim (or trusted identity mapping table) is authoritative for organization
  membership and application roles.
- Whether organization-visible documents may be shared outside their organization and whether
  user/group shares are required.
- Whether observability traces may contain document text/query text at all; this implementation
  defaults toward minimizing content but retention/redaction policy needs product ownership.
- Exact role/permission assignments, rate/cost budgets, cache backend, data retention periods,
  audit-event sink, and incident response integration.
- Migration/adoption policy for legacy Qdrant points and legacy observability rows. Secure default:
  quarantine them from normal user retrieval rather than treating them as globally visible.

## Document lifecycle follow-up

The lifecycle implementation uses the stricter owner-or-explicit-share rule. Organization
visibility remains a legacy schema value but is not an access grant. Primary symbols are
`src.api.routers.documents:router`, `src.storage.object_store:S3ObjectStore`,
`src.documents.service:DocumentLifecycleService`, and migration `d4e81f70a922`.

The remaining production decision is durable worker topology. The repository records job state
durably and executes asynchronously in-process for the single local API replica. Multi-replica
deployment must add leased job claiming or an external queue before enabling uploads.
