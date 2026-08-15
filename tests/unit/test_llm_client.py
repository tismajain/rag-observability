"""Coverage for the LLM client surface beyond what test_generation.py exercises.

We focus on construction-time validation:

* :class:`AnthropicClient` fails fast when ``settings.anthropic_api_key`` is None.
* :class:`OpenAIClient` fails fast when ``settings.openai_api_key`` is None.
* :class:`OpenAIClient` substitutes ``gpt-4o-mini`` when the configured model
  is a Claude id — switching provider without changing ``LLM__MODEL`` would
  otherwise pass a Claude id to OpenAI's chat completions API.
* The factory dispatches on ``settings.llm.provider`` when no explicit
  ``provider`` argument is given.
"""

from __future__ import annotations

import pytest
from pydantic import SecretStr

from src.generation.llm_client import (
    AnthropicClient,
    LLMClientConfigError,
    MockClient,
    OpenAIClient,
    get_llm_client,
)

# ----------------------------------------------------------------- anthropic


def test_anthropic_client_raises_when_key_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.generation.llm_client.settings.anthropic_api_key", None)
    with pytest.raises(LLMClientConfigError, match="ANTHROPIC_API_KEY"):
        AnthropicClient()


def test_anthropic_client_uses_configured_model_when_key_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("src.generation.llm_client.settings.anthropic_api_key", SecretStr("test"))
    client = AnthropicClient()
    assert client.provider == "anthropic"
    # default from LLMSettings.model
    assert client.model.startswith("claude")


# ------------------------------------------------------------------- openai


def test_openai_client_raises_when_key_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.generation.llm_client.settings.openai_api_key", None)
    with pytest.raises(LLMClientConfigError, match="OPENAI_API_KEY"):
        OpenAIClient()


def test_openai_client_swaps_claude_id_for_gpt_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Switching provider to openai while LLM__MODEL still names a Claude id
    should not pass ``claude-...`` to OpenAI."""
    monkeypatch.setattr("src.generation.llm_client.settings.openai_api_key", SecretStr("test"))
    monkeypatch.setattr("src.generation.llm_client.settings.llm.model", "claude-sonnet-4-6")
    client = OpenAIClient()
    assert client.model == "gpt-4o-mini"


def test_openai_client_keeps_explicit_gpt_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("src.generation.llm_client.settings.openai_api_key", SecretStr("test"))
    client = OpenAIClient(model="gpt-4o")
    assert client.model == "gpt-4o"


# ------------------------------------------------------------------ factory


def test_factory_falls_through_to_settings_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No explicit provider → factory reads settings.llm.provider."""
    monkeypatch.setattr("src.generation.llm_client.settings.llm.provider", "mock")
    client = get_llm_client()
    assert isinstance(client, MockClient)


def test_factory_explicit_provider_overrides_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("src.generation.llm_client.settings.llm.provider", "anthropic")
    client = get_llm_client(provider="mock")
    assert isinstance(client, MockClient)


async def test_mock_client_response_shape_is_stable() -> None:
    """The MockClient is the keyless fallback — its response shape must stay stable
    so /query keeps working in mock mode."""
    client = MockClient()
    resp = await client.generate("system", "user")
    assert resp.provider == "mock"
    assert resp.model == "mock-echo-1"
    assert resp.text.startswith("[mock answer]")
    assert resp.prompt_tokens > 0
    assert resp.completion_tokens > 0
    assert resp.latency_ms >= 0
    assert resp.finish_reason == "end_turn"
