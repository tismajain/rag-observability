"""Evaluation metric definitions + thresholds.

These are the three RAGAS metrics the pipeline scores against:

* **faithfulness** — does the answer only assert claims that the retrieved
  context supports? Low score → likely hallucination.
* **context_recall** — how much of the *answer-relevant* information was
  actually present in the retrieved context? Low score → retrieval missed
  something.
* **answer_relevancy** — does the answer address the question? Low score →
  the model wandered.

The quality gate evaluates a result against per-metric thresholds and emits
``pass``/``warn``/``fail``. Centralizing thresholds here means we can tune
gates without touching the evaluator or the API layer.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class MetricName(StrEnum):
    FAITHFULNESS = "faithfulness"
    CONTEXT_RECALL = "context_recall"
    ANSWER_RELEVANCY = "answer_relevancy"


class MetricThreshold(BaseModel):
    """Per-metric thresholds for the quality gate.

    Semantics:
    * ``score >= warn`` → ``pass``
    * ``fail <= score < warn`` → ``warn``
    * ``score < fail`` → ``fail``
    """

    warn: float = Field(..., ge=0.0, le=1.0)
    fail: float = Field(..., ge=0.0, le=1.0)

    def evaluate(self, score: float | None) -> str:
        if score is None:
            return "skip"
        if score >= self.warn:
            return "pass"
        if score >= self.fail:
            return "warn"
        return "fail"


# Defaults tuned for "noisy but useful" — most production RAG systems run in
# this range. Move them stricter (warn=0.85, fail=0.70) when content quality
# improves.
DEFAULT_THRESHOLDS: dict[MetricName, MetricThreshold] = {
    MetricName.FAITHFULNESS: MetricThreshold(warn=0.75, fail=0.50),
    MetricName.CONTEXT_RECALL: MetricThreshold(warn=0.70, fail=0.50),
    MetricName.ANSWER_RELEVANCY: MetricThreshold(warn=0.75, fail=0.50),
}
