"""Unit tests for prompt templates and the mock LLM client."""

from __future__ import annotations

import pytest

from src.generation.llm_client import (
    LLMClientConfigError,
    MockClient,
    get_llm_client,
)
from src.generation.prompt_templates import RAG_ANSWER_PROMPT, PromptTemplate

# ----------------------------------------------------------- prompt templates


def test_rag_prompt_renders_both_variables() -> None:
    system, user = RAG_ANSWER_PROMPT.render(
        context="Acme Corp uses blue-green deployments.",
        question="How does Acme deploy to production?",
    )
    assert "blue-green deployments" in user
    assert "How does Acme deploy to production?" in user
    assert "context-grounded" in system or "based on" in system.lower()


def test_prompt_missing_variable_leaves_placeholder() -> None:
    tpl = PromptTemplate(name="custom", version="v1", system="sys", user="hello {missing} world")
    _, user = tpl.render()
    assert "{missing}" in user  # not crashed


def test_prompt_template_is_immutable() -> None:
    # frozen=True dataclasses raise FrozenInstanceError (a subclass of AttributeError).
    with pytest.raises(AttributeError):
        RAG_ANSWER_PROMPT.system = "tampered"  # type: ignore[misc]


# --------------------------------------------------------------- mock client


async def test_mock_client_returns_canned_response() -> None:
    client = MockClient()
    resp = await client.generate("you are helpful", "Question: hello?\nContext: world")
    assert resp.provider == "mock"
    assert resp.text.startswith("[mock answer]")
    assert resp.prompt_tokens > 0
    assert resp.completion_tokens > 0
    assert resp.finish_reason == "end_turn"


async def test_mock_client_includes_context_for_overlap_signal() -> None:
    """The mock echoes the user prompt so defect detection sees overlap."""
    client = MockClient()
    resp = await client.generate("sys", "Question: rotate the on-call schedule")
    assert "on-call" in resp.text or "rotate" in resp.text


# --------------------------------------------------------------- factory


def test_factory_returns_mock_when_provider_is_mock() -> None:
    client = get_llm_client(provider="mock")
    assert isinstance(client, MockClient)


def test_factory_rejects_unknown_provider() -> None:
    with pytest.raises(LLMClientConfigError, match="Unknown LLM provider"):
        get_llm_client(provider="cosmic-llm")  # type: ignore[arg-type]
