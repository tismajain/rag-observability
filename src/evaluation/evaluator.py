"""RAGAS-based async evaluator.

Design contract: **evaluation must never crash the request path**. The /query
endpoint will spawn :meth:`Evaluator.evaluate_async` via ``asyncio.create_task``
after the response is returned. Anything that goes wrong here — missing
optional deps, missing LLM keys, RAGAS itself raising — must be logged and
surfaced via :class:`EvalResult.status`, not propagated.

Three failure modes are first-class:

1. ``ragas`` not installed (the ``[eval]`` extra was not selected).
2. No LLM API key configured (Anthropic for the default judge).
3. RAGAS itself errored at runtime.

In all three cases :meth:`evaluate_async` returns an :class:`EvalResult`
with ``status="skipped"`` or ``"failed"`` and the metric scores left as
``None``. The quality gate then returns ``"skip"`` — the request still
succeeded, eval just didn't happen.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Literal

import structlog
from pydantic import BaseModel, Field

from src.config.settings import settings
from src.evaluation.metrics import MetricName
from src.evaluation.quality_gate import QualityGateResult, evaluate_quality_gate
from src.observability.attributes import SpanAttributes
from src.observability.spans import eval_span

log = structlog.get_logger(__name__)

EvalStatus = Literal["ok", "skipped", "failed"]


class EvalResult(BaseModel):
    query_id: str
    trace_id: str
    status: EvalStatus
    faithfulness: float | None = None
    context_recall: float | None = None
    answer_relevancy: float | None = None
    quality_gate_result: QualityGateResult = "skip"
    evaluation_latency_ms: int = 0
    judge_model: str = ""
    error: str | None = None
    evaluated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Evaluator:
    """Async RAGAS runner. Graceful by construction."""

    def __init__(self, judge_model: str | None = None) -> None:
        self._judge_model = judge_model or settings.llm.model

    async def evaluate_async(
        self,
        query_id: str,
        trace_id: str,
        question: str,
        contexts: list[str],
        answer: str,
    ) -> EvalResult:
        """Score one query/answer pair against the retrieved contexts.

        Returns an :class:`EvalResult` even on failure — the caller can pass
        it directly to the eval store / log it / set span attributes.
        """
        t0 = time.perf_counter()
        with eval_span(query_id, self._judge_model) as span:
            # Fast-path skips: missing optional deps or missing keys.
            skip_reason = self._should_skip()
            if skip_reason is not None:
                span.set_attribute(SpanAttributes.EVAL_STATUS, "skipped")
                log.info(
                    "evaluation.skipped",
                    query_id=query_id,
                    reason=skip_reason,
                    judge_model=self._judge_model,
                )
                return EvalResult(
                    query_id=query_id,
                    trace_id=trace_id,
                    status="skipped",
                    judge_model=self._judge_model,
                    error=skip_reason,
                )

            try:
                scores = await self._run_ragas(question, contexts, answer)
            except Exception as exc:  # noqa: BLE001 — eval must never re-raise
                latency_ms = int((time.perf_counter() - t0) * 1000)
                span.set_attribute(SpanAttributes.EVAL_STATUS, "failed")
                span.set_attribute(SpanAttributes.EVAL_LATENCY_MS, latency_ms)
                log.warning(
                    "evaluation.failed",
                    query_id=query_id,
                    judge_model=self._judge_model,
                    error=str(exc),
                    error_type=type(exc).__name__,
                )
                return EvalResult(
                    query_id=query_id,
                    trace_id=trace_id,
                    status="failed",
                    judge_model=self._judge_model,
                    evaluation_latency_ms=latency_ms,
                    error=str(exc),
                )

            latency_ms = int((time.perf_counter() - t0) * 1000)
            gate = evaluate_quality_gate(
                {
                    MetricName.FAITHFULNESS: scores.get("faithfulness"),
                    MetricName.CONTEXT_RECALL: scores.get("context_recall"),
                    MetricName.ANSWER_RELEVANCY: scores.get("answer_relevancy"),
                }
            )

            span.set_attribute(SpanAttributes.EVAL_STATUS, "ok")
            span.set_attribute(SpanAttributes.EVAL_LATENCY_MS, latency_ms)
            if (v := scores.get("faithfulness")) is not None:
                span.set_attribute(SpanAttributes.EVAL_FAITHFULNESS, float(v))
            if (v := scores.get("context_recall")) is not None:
                span.set_attribute(SpanAttributes.EVAL_CONTEXT_RECALL, float(v))
            if (v := scores.get("answer_relevancy")) is not None:
                span.set_attribute(SpanAttributes.EVAL_ANSWER_RELEVANCE, float(v))
            span.set_attribute(SpanAttributes.EVAL_QUALITY_GATE, gate)

            log.info(
                "evaluation.completed",
                query_id=query_id,
                judge_model=self._judge_model,
                latency_ms=latency_ms,
                quality_gate=gate,
                **{f"score.{k}": v for k, v in scores.items()},
            )

            return EvalResult(
                query_id=query_id,
                trace_id=trace_id,
                status="ok",
                faithfulness=scores.get("faithfulness"),
                context_recall=scores.get("context_recall"),
                answer_relevancy=scores.get("answer_relevancy"),
                quality_gate_result=gate,
                evaluation_latency_ms=latency_ms,
                judge_model=self._judge_model,
            )

    # ----------------------------------------------------------------- skip

    def _should_skip(self) -> str | None:
        """Decide whether to skip eval up-front. Returns a reason or None."""
        try:
            _shim_langchain_vertexai()
            import ragas  # noqa: F401
        except ImportError as exc:
            return f'ragas-not-installed ({exc}); install with `pip install -e ".[eval]"`'

        provider = settings.llm.provider
        if provider == "anthropic" and settings.anthropic_api_key is None:
            return "anthropic-api-key-missing"
        if provider == "openai" and settings.openai_api_key is None:
            return "openai-api-key-missing"
        # RAGAS always needs OPENAI_API_KEY for its default judge + embeddings,
        # regardless of which LLM the main pipeline uses for generation.
        if settings.openai_api_key is None:
            return "openai-api-key-missing-for-ragas-judge"
        return None

    # ----------------------------------------------------------------- ragas

    async def _run_ragas(
        self,
        question: str,
        contexts: list[str],
        answer: str,
    ) -> dict[str, float | None]:
        """Invoke RAGAS. Isolated so the import is only paid when eval runs.

        This is intentionally minimal — we wire just the three metrics the
        spec calls out. Phase 5+ can add more (answer_correctness,
        context_precision) by extending this method.
        """
        # Local imports keep the heavy ragas/langchain graph out of module
        # import time. They are only resolved when an actual eval runs.
        _shim_langchain_vertexai()
        _propagate_openai_key_to_env()
        from datasets import Dataset
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_recall,
            faithfulness,
        )

        sample = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
            # context_recall needs a ground truth; without one, RAGAS will
            # treat the answer as the reference (a reasonable proxy when no
            # gold answer exists).
            "ground_truth": [answer],
        }
        dataset = Dataset.from_dict(sample)

        # RAGAS' evaluate() is synchronous. Run it on the default executor so
        # we don't block the event loop.
        import asyncio

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: ragas_evaluate(
                dataset,
                metrics=[faithfulness, context_recall, answer_relevancy],
            ),
        )

        # RAGAS returns a Result object with a .scores attribute (list of dicts
        # per row). We sent one row.
        row = result.scores[0] if getattr(result, "scores", None) else {}
        return {
            "faithfulness": _as_float(row.get("faithfulness")),
            "context_recall": _as_float(row.get("context_recall")),
            "answer_relevancy": _as_float(row.get("answer_relevancy")),
        }


def _propagate_openai_key_to_env() -> None:
    """RAGAS internals read ``OPENAI_API_KEY`` from os.environ, not our settings.

    Mirror the pydantic-loaded key into the process env so RAGAS's default
    LangChain LLM/embeddings can pick it up. No-op if it's already set or
    the key isn't configured.
    """
    import os

    if os.environ.get("OPENAI_API_KEY"):
        return
    if settings.openai_api_key is None:
        return
    os.environ["OPENAI_API_KEY"] = settings.openai_api_key.get_secret_value()


def _shim_langchain_vertexai() -> None:
    """Inject a stub for ``langchain_community.chat_models.vertexai``.

    RAGAS imports ``ChatVertexAI`` at module load time, but recent
    ``langchain-community`` versions removed that submodule (Vertex AI moved
    to its own package). We don't use Vertex; inject a tiny stub so the
    import succeeds. No-op if the real module is available.
    """
    import sys
    import types

    name = "langchain_community.chat_models.vertexai"
    if name in sys.modules:
        return
    try:
        import langchain_community.chat_models  # noqa: F401
    except ImportError:
        return  # nothing to shim into; ragas import will fail with a clearer error
    module = types.ModuleType(name)

    class _StubChatVertexAI:  # pragma: no cover — never instantiated
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise RuntimeError(
                "ChatVertexAI is unavailable in this build; configure a "
                "different RAGAS judge or install langchain-google-vertexai."
            )

    module.ChatVertexAI = _StubChatVertexAI  # type: ignore[attr-defined]
    sys.modules[name] = module


def _as_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    # RAGAS occasionally returns NaN for failed-to-score rows.
    if f != f:  # NaN check
        return None
    return f
