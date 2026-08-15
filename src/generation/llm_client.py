"""LLM client abstraction.

Three implementations behind a common :class:`BaseLLMClient` interface:

* :class:`AnthropicClient` — Claude (default; matches the spec's primary model)
* :class:`OpenAIClient` — GPT (fallback when ``LLM__PROVIDER=openai``)
* :class:`MockClient` — deterministic echo; lets ``/query`` run end-to-end
  with no API key, useful for demos and integration tests

The factory :func:`get_llm_client` picks the right implementation from
``settings.llm.provider``.

Every real call is wrapped in:

* an OTel span (``rag.generation``) — already correct from Phase 4 helpers
* tenacity retry with exponential backoff + jitter on transient errors
* a settings-driven timeout

Output is :class:`LLMResponse` so downstream code never depends on the
provider's native response shape.
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal

import structlog
from anthropic import (
    APIConnectionError as AnthropicAPIConnectionError,
)
from anthropic import (
    APITimeoutError as AnthropicAPITimeoutError,
)
from anthropic import (
    AsyncAnthropic,
    RateLimitError,
)
from openai import (
    APIConnectionError as OpenAIAPIConnectionError,
)
from openai import (
    APITimeoutError as OpenAIAPITimeoutError,
)
from openai import (
    AsyncOpenAI,
)
from openai import (
    RateLimitError as OpenAIRateLimitError,
)
from opentelemetry.trace import Status, StatusCode
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from src.config.settings import settings
from src.ingestion.chunker import count_tokens
from src.observability.attributes import SpanAttributes
from src.observability.tracer import get_tracer

if TYPE_CHECKING:
    pass

log = structlog.get_logger(__name__)


class LLMClientConfigError(RuntimeError):
    """Raised when a provider is selected without the required credentials."""


class LLMResponse(BaseModel):
    text: str
    model: str
    provider: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str | None = None
    latency_ms: int


class BaseLLMClient(ABC):
    provider: str
    model: str

    @abstractmethod
    async def generate(self, system: str, user: str) -> LLMResponse: ...

    async def aclose(self) -> None:
        return None


# --------------------------------------------------------------------- mock --


class MockClient(BaseLLMClient):
    """Deterministic echo client. Always returns a canned answer + a short
    digest of the context so /query stays observable without an API key.
    """

    provider = "mock"

    def __init__(self, model: str | None = None) -> None:
        self.model = model or "mock-echo-1"
        self._tracer = get_tracer(__name__)

    async def generate(self, system: str, user: str) -> LLMResponse:
        with self._tracer.start_as_current_span("rag.generation") as span:
            span.set_attribute(SpanAttributes.GENERATION_PROVIDER, self.provider)
            span.set_attribute(SpanAttributes.GENERATION_MODEL, self.model)

            t0 = time.perf_counter()
            # Pull out a 200-char preview of the user prompt so the echo gives
            # the defect-detector a meaningful overlap signal.
            preview = user.replace("\n", " ").strip()[:400]
            text = (
                "[mock answer] Based on the provided context, here is a "
                "context-grounded response: "
                f"{preview}"
            )
            latency_ms = int((time.perf_counter() - t0) * 1000)
            prompt_tokens = count_tokens(system) + count_tokens(user)
            completion_tokens = count_tokens(text)

            span.set_attribute(SpanAttributes.GENERATION_LATENCY_MS, latency_ms)
            span.set_attribute(SpanAttributes.GENERATION_PROMPT_TOKENS, prompt_tokens)
            span.set_attribute(SpanAttributes.GENERATION_COMPLETION_TOKENS, completion_tokens)
            span.set_attribute(SpanAttributes.GENERATION_FINISH_REASON, "end_turn")

        log.info(
            "generation.mock.completed",
            model=self.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
        )
        return LLMResponse(
            text=text,
            model=self.model,
            provider=self.provider,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason="end_turn",
            latency_ms=latency_ms,
        )


# --------------------------------------------------------------- anthropic --


class AnthropicClient(BaseLLMClient):
    provider = "anthropic"

    def __init__(self, model: str | None = None) -> None:
        if settings.anthropic_api_key is None:
            raise LLMClientConfigError("ANTHROPIC_API_KEY is not set")
        self.model = model or settings.llm.model
        self._client = AsyncAnthropic(
            api_key=settings.anthropic_api_key.get_secret_value(),
            timeout=settings.llm.timeout_seconds,
        )
        self._tracer = get_tracer(__name__)

    async def generate(self, system: str, user: str) -> LLMResponse:
        with self._tracer.start_as_current_span("rag.generation") as span:
            span.set_attribute(SpanAttributes.GENERATION_PROVIDER, self.provider)
            span.set_attribute(SpanAttributes.GENERATION_MODEL, self.model)

            t0 = time.perf_counter()
            try:
                resp = await self._call_with_retry(system, user)
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                raise

            text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
            latency_ms = int((time.perf_counter() - t0) * 1000)
            prompt_tokens = resp.usage.input_tokens
            completion_tokens = resp.usage.output_tokens
            finish_reason = resp.stop_reason or "end_turn"

            span.set_attribute(SpanAttributes.GENERATION_LATENCY_MS, latency_ms)
            span.set_attribute(SpanAttributes.GENERATION_PROMPT_TOKENS, prompt_tokens)
            span.set_attribute(SpanAttributes.GENERATION_COMPLETION_TOKENS, completion_tokens)
            span.set_attribute(SpanAttributes.GENERATION_FINISH_REASON, finish_reason)

        log.info(
            "generation.anthropic.completed",
            model=self.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            finish_reason=finish_reason,
        )
        return LLMResponse(
            text=text,
            model=self.model,
            provider=self.provider,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason=finish_reason,
            latency_ms=latency_ms,
        )

    @retry(
        reraise=True,
        stop=stop_after_attempt(settings.llm.max_retries),
        wait=wait_exponential_jitter(initial=1, max=10),
        retry=retry_if_exception_type(
            (AnthropicAPITimeoutError, AnthropicAPIConnectionError, RateLimitError)
        ),
    )
    async def _call_with_retry(self, system: str, user: str):  # type: ignore[no-untyped-def]
        return await self._client.messages.create(
            model=self.model,
            system=system,
            messages=[{"role": "user", "content": user}],
            max_tokens=settings.llm.max_tokens,
            temperature=settings.llm.temperature,
        )

    async def aclose(self) -> None:
        await self._client.close()


# ----------------------------------------------------------------- openai --


class OpenAIClient(BaseLLMClient):
    provider = "openai"

    def __init__(self, model: str | None = None) -> None:
        if settings.openai_api_key is None:
            raise LLMClientConfigError("OPENAI_API_KEY is not set")
        # When the user has switched provider to openai but kept the Claude
        # model id in settings, fall back to a sensible OpenAI default.
        chosen = model or settings.llm.model
        if chosen.startswith("claude"):
            chosen = "gpt-4o-mini"
        self.model = chosen
        self._client = AsyncOpenAI(
            api_key=settings.openai_api_key.get_secret_value(),
            timeout=settings.llm.timeout_seconds,
        )
        self._tracer = get_tracer(__name__)

    async def generate(self, system: str, user: str) -> LLMResponse:
        with self._tracer.start_as_current_span("rag.generation") as span:
            span.set_attribute(SpanAttributes.GENERATION_PROVIDER, self.provider)
            span.set_attribute(SpanAttributes.GENERATION_MODEL, self.model)

            t0 = time.perf_counter()
            try:
                resp = await self._call_with_retry(system, user)
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                raise

            choice = resp.choices[0]
            text = choice.message.content or ""
            latency_ms = int((time.perf_counter() - t0) * 1000)
            prompt_tokens = resp.usage.prompt_tokens if resp.usage else 0
            completion_tokens = resp.usage.completion_tokens if resp.usage else 0
            finish_reason = choice.finish_reason or "stop"

            span.set_attribute(SpanAttributes.GENERATION_LATENCY_MS, latency_ms)
            span.set_attribute(SpanAttributes.GENERATION_PROMPT_TOKENS, prompt_tokens)
            span.set_attribute(SpanAttributes.GENERATION_COMPLETION_TOKENS, completion_tokens)
            span.set_attribute(SpanAttributes.GENERATION_FINISH_REASON, finish_reason)

        log.info(
            "generation.openai.completed",
            model=self.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            finish_reason=finish_reason,
        )
        return LLMResponse(
            text=text,
            model=self.model,
            provider=self.provider,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason=finish_reason,
            latency_ms=latency_ms,
        )

    @retry(
        reraise=True,
        stop=stop_after_attempt(settings.llm.max_retries),
        wait=wait_exponential_jitter(initial=1, max=10),
        retry=retry_if_exception_type(
            (OpenAIAPITimeoutError, OpenAIAPIConnectionError, OpenAIRateLimitError)
        ),
    )
    async def _call_with_retry(self, system: str, user: str):  # type: ignore[no-untyped-def]
        return await self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=settings.llm.max_tokens,
            temperature=settings.llm.temperature,
        )

    async def aclose(self) -> None:
        await self._client.close()


# ----------------------------------------------------------------- factory --

ProviderName = Literal["anthropic", "openai", "mock"]


def get_llm_client(provider: ProviderName | None = None) -> BaseLLMClient:
    """Instantiate the configured LLM client."""
    chosen = provider or settings.llm.provider
    if chosen == "anthropic":
        return AnthropicClient()
    if chosen == "openai":
        return OpenAIClient()
    if chosen == "mock":
        return MockClient()
    raise LLMClientConfigError(f"Unknown LLM provider: {chosen!r}")


# Helper: quiet `asyncio` from spawning warnings about un-awaited coroutines
# in tests that construct + discard a client. Touch nothing at import time.
async def _noop() -> None:  # pragma: no cover
    await asyncio.sleep(0)
