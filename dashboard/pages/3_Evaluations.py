"""Eval Trends page.

RAGAS scores for evaluated queries: headline aggregates, quality-gate
distribution, and a per-query table sorted newest-first.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.api_client import APIError, evals_summary, list_evals
from dashboard.auth import require_oidc

st.set_page_config(page_title="Evaluations", layout="wide")
require_oidc()
st.title("Eval Trends")
st.caption("RAGAS scores from sampled /query interactions.")


# ----------------------------------------------------------------- summary

with st.sidebar:
    st.header("Period")
    period = st.selectbox("Window", ["24h", "7d", "30d", "90d"], index=1)

try:
    summary = evals_summary(period=period)
except APIError as exc:
    st.error(f"Could not fetch eval summary: {exc}")
    st.stop()

st.subheader(f"Aggregate (last {period})")

cols = st.columns(4)
cols[0].metric("Total evaluations", summary.get("total_evaluated", 0))


def _fmt(value: float | None) -> str:
    return f"{value:.3f}" if value is not None else "—"


cols[1].metric("Mean faithfulness", _fmt(summary.get("mean_faithfulness")))
cols[2].metric("Mean context recall", _fmt(summary.get("mean_context_recall")))
cols[3].metric("Mean answer relevancy", _fmt(summary.get("mean_answer_relevancy")))

# Gate distribution as a small bar chart so the eye reads pass/fail balance fast.
gate_dist = summary.get("quality_gate_distribution", {}) or {}
if gate_dist:
    gate_df = pd.DataFrame(
        {
            "gate": list(gate_dist.keys()),
            "count": [int(v) for v in gate_dist.values()],
        }
    )
    st.subheader("Quality gate distribution")
    st.bar_chart(gate_df, x="gate", y="count", height=220)

st.divider()


# ----------------------------------------------------------------- list

st.subheader("Recent evaluations")

try:
    evals = list_evals(limit=50)
except APIError as exc:
    st.error(f"Could not fetch evals: {exc}")
    st.stop()

items = evals.get("items", [])
if not items:
    st.info(
        "No evaluations recorded yet. Sampling rate is controlled by `OBSERVABILITY__SAMPLE_RATE_FOR_EVAL`."
    )
else:
    rows = [
        {
            "evaluated_at": e["evaluated_at"],
            "query_id": e["query_id"][:8] + "…",
            "trace_id": e["trace_id"][:12] + "…",
            "faithfulness": e.get("faithfulness"),
            "context_recall": e.get("context_recall"),
            "answer_relevancy": e.get("answer_relevancy"),
            "quality_gate": e.get("quality_gate_result"),
            "status": e.get("status"),
            "latency_ms": e.get("evaluation_latency_ms"),
            "judge_model": e.get("judge_model"),
        }
        for e in items
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)
