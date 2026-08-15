"""Aggregate per-metric verdicts into a single pass/warn/fail decision.

Rule: the gate is as strict as its strictest metric.

    any fail   → fail
    any warn   → warn
    otherwise  → pass
    all skip   → skip       (no metric was scored — eval didn't run)

This deliberately ignores metric weighting; if we want to weight metrics
later we add a ``policy`` argument. For now, the strictest-wins rule is
the right default — one bad faithfulness score should not be averaged
away by good relevancy.
"""

from __future__ import annotations

from typing import Literal

from src.evaluation.metrics import DEFAULT_THRESHOLDS, MetricName, MetricThreshold

QualityGateResult = Literal["pass", "warn", "fail", "skip"]


def evaluate_quality_gate(
    scores: dict[MetricName, float | None],
    thresholds: dict[MetricName, MetricThreshold] | None = None,
) -> QualityGateResult:
    """Return the overall gate verdict for one query's RAGAS scores."""
    thresholds = thresholds or DEFAULT_THRESHOLDS
    verdicts: list[str] = []
    for metric, score in scores.items():
        threshold = thresholds.get(metric)
        if threshold is None:
            continue
        verdicts.append(threshold.evaluate(score))

    if not verdicts or all(v == "skip" for v in verdicts):
        return "skip"
    if "fail" in verdicts:
        return "fail"
    if "warn" in verdicts:
        return "warn"
    return "pass"
