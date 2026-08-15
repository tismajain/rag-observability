"""Typed application settings.

Single source of truth for every runtime configuration value. Importers do::

    from src.config.settings import settings

Settings are loaded from environment variables (and a `.env` file in development).
Nested fields use the `__` delimiter, e.g. ``QDRANT__HOST=qdrant``.

Required fields (currently ``anthropic_api_key``) raise at import time if missing —
this implements the spec's "fail fast on startup" rule.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class QdrantSettings(BaseModel):
    """Connection and collection layout for the Qdrant vector store."""

    host: str = "localhost"
    port: int = 6333
    collection_name: str = Field(
        default="rag_documents",
        description="Primary collection holding ingested document chunks.",
    )
    meta_collection_name: str = Field(
        default="rag_traces",
        description="Secondary collection used by the meta-RAG layer to index trace history.",
    )
    vector_size: int = Field(
        default=1536,
        description="Dimensionality of stored embeddings. Must match the active embedding model.",
    )


class LLMSettings(BaseModel):
    """Generation-model configuration. Provider is switchable at runtime.

    ``mock`` returns a deterministic canned answer that echoes the context;
    useful for end-to-end demos when no real API key is available.
    """

    provider: Literal["anthropic", "openai", "mock"] = "anthropic"
    model: str = "claude-sonnet-4-6"
    temperature: float = 0.0
    max_tokens: int = 2048
    timeout_seconds: int = 30
    max_retries: int = 3


class ObservabilitySettings(BaseModel):
    """Tracing, evaluation sampling, and defect-detection thresholds."""

    phoenix_endpoint: str = Field(
        default="http://localhost:4317",
        description="OTLP gRPC endpoint for the Arize Phoenix collector.",
    )
    enable_console_exporter: bool = Field(
        default=False,
        description="Also export spans to stdout. Useful for local debugging; noisy in prod.",
    )
    sample_rate_for_eval: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Fraction of completed queries that trigger an async RAGAS evaluation.",
    )
    defect_similarity_threshold: float = Field(
        default=0.65,
        description="Retrieval scores below this raise the LOW_RETRIEVAL_QUALITY defect.",
    )


class AuthSettings(BaseModel):
    """Provider-neutral OIDC resource-server configuration."""

    enabled: bool = False
    issuer: str | None = None
    audience: str | None = None
    jwks_uri: str | None = None
    algorithms: tuple[str, ...] = ("RS256",)
    roles_claim: str = "roles"
    organization_claim: str = "org_id"
    permissions_claim: str = "permissions"
    jwks_cache_ttl_seconds: int = Field(default=300, ge=30, le=86400)
    jwks_timeout_seconds: float = Field(default=3.0, gt=0, le=15)
    clock_skew_seconds: int = Field(default=30, ge=0, le=300)
    protect_api_docs: bool = True
    disabled_local_user_id: str = "local-development"
    disabled_local_organization_id: str = "local-development"

    @model_validator(mode="after")
    def validate_oidc(self) -> AuthSettings:
        if not self.algorithms or any(a.lower() == "none" for a in self.algorithms):
            raise ValueError("AUTH__ALGORITHMS must contain pinned asymmetric algorithms")
        if any(not a.startswith(("RS", "ES", "PS")) for a in self.algorithms):
            raise ValueError("Only asymmetric JWT algorithms are accepted")
        if self.enabled:
            missing = [
                name
                for name, value in (
                    ("issuer", self.issuer),
                    ("audience", self.audience),
                    ("jwks_uri", self.jwks_uri),
                )
                if not value
            ]
            if missing:
                raise ValueError(f"OIDC enabled but missing: {', '.join(missing)}")
            if not str(self.jwks_uri).startswith("https://"):
                raise ValueError("AUTH__JWKS_URI must use HTTPS when authentication is enabled")
        return self


class RateLimitSettings(BaseModel):
    backend: Literal["memory", "redis"] = "memory"
    redis_url: SecretStr | None = None
    query_per_minute: int = Field(default=30, ge=1)
    meta_query_per_minute: int = Field(default=20, ge=1)
    per_ip_per_minute: int = Field(default=120, ge=1)
    max_concurrent_per_user: int = Field(default=2, ge=1)
    max_concurrent_global: int = Field(default=50, ge=1)


class ResourceControlSettings(BaseModel):
    max_body_bytes: int = Field(default=1_048_576, ge=1024)
    max_query_characters: int = Field(default=2000, ge=1)
    max_top_k: int = Field(default=20, ge=1, le=100)
    max_llm_input_tokens: int = Field(default=16_000, ge=1)
    max_llm_output_tokens: int = Field(default=2048, ge=1)
    daily_llm_cost_units_per_user: int = Field(default=100_000, ge=1)


class Settings(BaseSettings):
    """Top-level application settings, composed from environment variables."""

    app_name: str = "rag-observability"
    environment: Literal["development", "staging", "production"] = "development"
    log_level: str = "INFO"

    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/rag_obs"
    ingestion_service_user_id: str | None = None
    ingestion_service_organization_id: str | None = None

    anthropic_api_key: SecretStr
    openai_api_key: SecretStr | None = None

    qdrant: QdrantSettings = QdrantSettings()
    llm: LLMSettings = LLMSettings()
    observability: ObservabilitySettings = ObservabilitySettings()
    auth: AuthSettings = AuthSettings()
    rate_limit: RateLimitSettings = RateLimitSettings()
    resources: ResourceControlSettings = ResourceControlSettings()

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
        case_sensitive=False,
    )

    @model_validator(mode="after")
    def validate_environment_security(self) -> Settings:
        if self.environment in {"staging", "production"}:
            if not self.auth.enabled:
                raise ValueError("Authentication must be enabled outside development")
            if self.rate_limit.backend != "redis" or self.rate_limit.redis_url is None:
                raise ValueError("Staging/production requires a Redis rate-limit backend")
        return self


settings = Settings()  # type: ignore[call-arg]
"""Module-level singleton. Construct once, import everywhere."""
