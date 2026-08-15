"""Tests for the RAGAS evaluator.

We don't actually run RAGAS here — that requires real OpenAI calls and the
heavy langchain dep graph. Instead we verify the *contract* the evaluator
guarantees to its callers:

* It never raises into the request path. All failure modes are returned via
  :class:`EvalResult.status` (``"skipped"`` or ``"failed"``).
* The three skip paths (no ragas installed, no Anthropic key when provider
  is anthropic, no OpenAI key for RAGAS judge) each return a recognizable
  reason in ``EvalResult.error``.
* Runtime failures inside ``_run_ragas`` are caught and surfaced as
  ``status="failed"`` without re-raising.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import AsyncMock, patch

import pytest

from src.evaluation.evaluator import EvalResult, Evaluator


def _stub_ragas_installed() -> types.ModuleType:
    """Inject a stub ``ragas`` module so ``import ragas`` succeeds."""
    stub = types.ModuleType("ragas")
    sys.modules["ragas"] = stub
    return stub


def _remove_ragas() -> None:
    sys.modules.pop("ragas", None)


# ----------------------------------------------------------------- skip paths


async def test_skip_when_ragas_not_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    _remove_ragas()
    # Force the import to fail by overriding the module loader.
    monkeypatch.setitem(sys.modules, "ragas", None)

    evaluator = Evaluator()
    result: EvalResult = await evaluator.evaluate_async(
        query_id="q1", trace_id="t1", question="x", contexts=["c"], answer="a"
    )
    assert result.status == "skipped"
    assert result.error is not None
    assert "ragas" in result.error.lower()
    # All three metric scores must be None when skipped.
    assert result.faithfulness is None
    assert result.context_recall is None
    assert result.answer_relevancy is None
    assert result.quality_gate_result == "skip"


async def test_skip_when_openai_key_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_ragas_installed()
    # Settings is a module-level singleton; patch the attribute directly.
    monkeypatch.setattr("src.evaluation.evaluator.settings.openai_api_key", None)

    evaluator = Evaluator()
    result = await evaluator.evaluate_async(
        query_id="q2", trace_id="t2", question="x", contexts=["c"], answer="a"
    )
    assert result.status == "skipped"
    assert result.error is not None
    assert "openai-api-key-missing" in result.error


# ----------------------------------------------------------------- failed path


async def test_failure_caught_and_surfaced_as_failed_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A runtime exception in _run_ragas must NOT propagate to the caller."""
    _stub_ragas_installed()

    from pydantic import SecretStr

    monkeypatch.setattr(
        "src.evaluation.evaluator.settings.openai_api_key",
        SecretStr("test-key"),
    )

    evaluator = Evaluator()
    with patch.object(
        evaluator,
        "_run_ragas",
        new=AsyncMock(side_effect=RuntimeError("simulated ragas blow-up")),
    ):
        result = await evaluator.evaluate_async(
            query_id="q3",
            trace_id="t3",
            question="x",
            contexts=["c"],
            answer="a",
        )

    assert result.status == "failed"
    assert result.error is not None
    assert "simulated ragas blow-up" in result.error
    assert result.evaluation_latency_ms >= 0


# ----------------------------------------------------------------- success path


async def test_success_path_surfaces_scores_and_quality_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When _run_ragas returns scores, the evaluator computes a quality gate."""
    _stub_ragas_installed()

    from pydantic import SecretStr

    monkeypatch.setattr(
        "src.evaluation.evaluator.settings.openai_api_key",
        SecretStr("test-key"),
    )

    evaluator = Evaluator()
    fake_scores = {"faithfulness": 0.95, "context_recall": 0.88, "answer_relevancy": 0.91}
    with patch.object(evaluator, "_run_ragas", new=AsyncMock(return_value=fake_scores)):
        result = await evaluator.evaluate_async(
            query_id="q4",
            trace_id="t4",
            question="x",
            contexts=["c"],
            answer="a",
        )

    assert result.status == "ok"
    assert result.faithfulness == pytest.approx(0.95)
    assert result.context_recall == pytest.approx(0.88)
    assert result.answer_relevancy == pytest.approx(0.91)
    # All three above their pass thresholds → quality gate passes.
    assert result.quality_gate_result == "pass"
    assert result.judge_model  # populated from settings.llm.model
