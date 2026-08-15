"""Quality gate boundary tests.

Verifies the strictest-wins rule across the three metrics and that the
``skip`` verdict is returned when no metric is scored.
"""

from __future__ import annotations

from src.evaluation.metrics import DEFAULT_THRESHOLDS, MetricName
from src.evaluation.quality_gate import evaluate_quality_gate


def test_all_pass_returns_pass() -> None:
    scores = {
        MetricName.FAITHFULNESS: 0.95,
        MetricName.CONTEXT_RECALL: 0.90,
        MetricName.ANSWER_RELEVANCY: 0.88,
    }
    assert evaluate_quality_gate(scores) == "pass"


def test_one_warn_returns_warn() -> None:
    scores = {
        MetricName.FAITHFULNESS: 0.95,
        MetricName.CONTEXT_RECALL: 0.60,  # warn band (>=0.50, <0.70)
        MetricName.ANSWER_RELEVANCY: 0.85,
    }
    assert evaluate_quality_gate(scores) == "warn"


def test_one_fail_returns_fail() -> None:
    scores = {
        MetricName.FAITHFULNESS: 0.40,  # fail (<0.50)
        MetricName.CONTEXT_RECALL: 0.95,
        MetricName.ANSWER_RELEVANCY: 0.95,
    }
    assert evaluate_quality_gate(scores) == "fail"


def test_warn_and_fail_returns_fail() -> None:
    scores = {
        MetricName.FAITHFULNESS: 0.40,  # fail
        MetricName.CONTEXT_RECALL: 0.60,  # warn
        MetricName.ANSWER_RELEVANCY: 0.85,
    }
    assert evaluate_quality_gate(scores) == "fail"


def test_all_none_returns_skip() -> None:
    scores = {
        MetricName.FAITHFULNESS: None,
        MetricName.CONTEXT_RECALL: None,
        MetricName.ANSWER_RELEVANCY: None,
    }
    assert evaluate_quality_gate(scores) == "skip"


def test_empty_returns_skip() -> None:
    assert evaluate_quality_gate({}) == "skip"


def test_boundary_score_at_warn_threshold_is_pass() -> None:
    # faithfulness warn threshold = 0.75 → exactly 0.75 should still pass
    scores = {
        MetricName.FAITHFULNESS: DEFAULT_THRESHOLDS[MetricName.FAITHFULNESS].warn,
        MetricName.CONTEXT_RECALL: 0.95,
        MetricName.ANSWER_RELEVANCY: 0.95,
    }
    assert evaluate_quality_gate(scores) == "pass"


def test_boundary_score_at_fail_threshold_is_warn() -> None:
    # exactly at fail threshold → still warn, not fail
    scores = {
        MetricName.FAITHFULNESS: DEFAULT_THRESHOLDS[MetricName.FAITHFULNESS].fail,
        MetricName.CONTEXT_RECALL: 0.95,
        MetricName.ANSWER_RELEVANCY: 0.95,
    }
    assert evaluate_quality_gate(scores) == "warn"


def test_score_just_below_fail_threshold_is_fail() -> None:
    fail = DEFAULT_THRESHOLDS[MetricName.FAITHFULNESS].fail
    scores = {
        MetricName.FAITHFULNESS: fail - 0.01,
        MetricName.CONTEXT_RECALL: 0.95,
        MetricName.ANSWER_RELEVANCY: 0.95,
    }
    assert evaluate_quality_gate(scores) == "fail"


def test_partial_scores_one_scored_metric() -> None:
    # Only one metric was successfully scored; others None
    scores = {
        MetricName.FAITHFULNESS: 0.95,
        MetricName.CONTEXT_RECALL: None,
        MetricName.ANSWER_RELEVANCY: None,
    }
    assert evaluate_quality_gate(scores) == "pass"


def test_custom_thresholds_strict() -> None:
    from src.evaluation.metrics import MetricThreshold

    strict = {
        MetricName.FAITHFULNESS: MetricThreshold(warn=0.95, fail=0.80),
        MetricName.CONTEXT_RECALL: MetricThreshold(warn=0.95, fail=0.80),
        MetricName.ANSWER_RELEVANCY: MetricThreshold(warn=0.95, fail=0.80),
    }
    scores = {
        MetricName.FAITHFULNESS: 0.85,
        MetricName.CONTEXT_RECALL: 0.90,
        MetricName.ANSWER_RELEVANCY: 0.92,
    }
    # Under strict thresholds all of these are in the warn band
    assert evaluate_quality_gate(scores, thresholds=strict) == "warn"
