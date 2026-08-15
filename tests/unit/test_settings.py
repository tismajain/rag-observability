"""Sanity tests for the Phase 1 settings module."""

from __future__ import annotations


def test_settings_defaults_load() -> None:
    from src.config.settings import settings

    assert settings.app_name == "rag-observability"
    assert settings.environment == "development"
    assert settings.qdrant.vector_size == 1536
    assert settings.qdrant.collection_name == "rag_documents"
    assert settings.qdrant.meta_collection_name == "rag_traces"


def test_llm_defaults() -> None:
    """Check the *defaults* on a fresh model — not the env-loaded singleton,
    which may legitimately differ in dev environments."""
    from src.config.settings import LLMSettings

    fresh = LLMSettings()
    assert fresh.provider == "anthropic"
    assert fresh.model == "claude-sonnet-4-6"
    assert fresh.temperature == 0.0
    assert fresh.max_retries == 3


def test_observability_defaults() -> None:
    """Defaults on a fresh model — not the env-loaded singleton."""
    from src.config.settings import ObservabilitySettings, settings

    fresh = ObservabilitySettings()
    assert fresh.phoenix_endpoint.startswith("http")
    assert 0.0 <= fresh.sample_rate_for_eval <= 1.0
    assert fresh.defect_similarity_threshold == 0.65

    # The env-loaded singleton is allowed to override the default.
    assert 0.0 <= settings.observability.defect_similarity_threshold <= 1.0


def test_anthropic_api_key_is_secret() -> None:
    from pydantic import SecretStr

    from src.config.settings import settings

    assert isinstance(settings.anthropic_api_key, SecretStr)
    # str() of a SecretStr must not leak the underlying value.
    assert "test-key" not in str(settings.anthropic_api_key)
