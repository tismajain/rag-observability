"""RAG Observability dashboard — overview page.

Single-page landing surface. Subpages (Traces, Defects, Evaluations, Meta
Query, New Query) live under ``dashboard/pages/`` and Streamlit auto-discovers
them, ordering by filename prefix.

The dashboard is read-mostly over the FastAPI service plus a single write
surface (POST /query, POST /meta/query). It does not touch Postgres or Qdrant
directly — the API is the single source of truth for shape and auth.
"""

from __future__ import annotations

import streamlit as st

from dashboard.api_client import API_BASE_URL, APIError, evals_summary, get_health, list_defects
from dashboard.auth import require_oidc

st.set_page_config(
    page_title="RAG Observability",
    layout="wide",
    initial_sidebar_state="expanded",
)
require_oidc()


# ----------------------------------------------------------------- sidebar


def _render_sidebar() -> None:
    st.sidebar.header("System")
    st.sidebar.caption(f"API: `{API_BASE_URL}`")
    try:
        health = get_health()
        status = health.get("status", "unknown")
        if status == "ok":
            st.sidebar.success(f"API status: {status}")
        else:
            st.sidebar.warning(f"API status: {status}")
        for dep in health.get("dependencies", []):
            icon = "✓" if dep.get("healthy") else "✗"
            st.sidebar.write(f"{icon} {dep['name']}")
    except APIError as exc:
        st.sidebar.error(f"API unreachable: {exc}")

    st.sidebar.divider()
    st.sidebar.caption(
        "Phoenix UI for trace-level drill-down: [localhost:6006](http://localhost:6006)"
    )


# -------------------------------------------------------------- overview


def _render_overview() -> None:
    st.title("RAG Observability")
    st.caption(
        "Operational dashboard for the production RAG pipeline. Trace explorer, "
        "defect monitor, eval trends, and natural-language meta-query."
    )

    # Eval headline numbers — quick reading of system health.
    try:
        summary = evals_summary(period="7d")
    except APIError as exc:
        st.warning(f"Could not fetch eval summary: {exc}")
        summary = None

    if summary:
        cols = st.columns(4)
        cols[0].metric("Evaluations (7d)", summary.get("total_evaluated", 0))
        _render_metric(cols[1], "Mean faithfulness", summary.get("mean_faithfulness"))
        _render_metric(cols[2], "Mean context recall", summary.get("mean_context_recall"))
        _render_metric(cols[3], "Mean answer relevancy", summary.get("mean_answer_relevancy"))

        st.subheader("Quality gate (last 7 days)")
        dist = summary.get("quality_gate_distribution", {})
        gate_cols = st.columns(4)
        for col, gate, color in zip(
            gate_cols,
            ["pass", "warn", "fail", "skip"],
            ["normal", "normal", "inverse", "off"],
            strict=False,
        ):
            col.metric(label=gate.title(), value=dist.get(gate, 0), delta_color=color)

    st.divider()

    # Recent defects feed — the operator's "what just went wrong" surface.
    st.subheader("Recent defects")
    try:
        recent = list_defects(limit=10)
    except APIError as exc:
        st.warning(f"Could not fetch defects: {exc}")
        recent = {"items": []}

    items = recent.get("items", [])
    if not items:
        st.info("No defects recorded yet.")
    else:
        for d in items:
            severity = d.get("severity", "MEDIUM")
            icon = _severity_icon(severity)
            with st.container(border=True):
                st.markdown(
                    f"**{icon} {d['defect_type']}** · `{severity}` · "
                    f"trace `{d['trace_id'][:12]}…` · {d['detected_at']}"
                )
                st.caption(d.get("description", ""))

    st.divider()
    st.markdown(
        "**Pages**: use the sidebar to drill into Traces, Defects, Evaluations, "
        "the natural-language Meta Query interface, or fire a New Query."
    )


def _render_metric(col: object, label: str, value: float | None) -> None:
    if value is None:
        col.metric(label, "—")  # type: ignore[attr-defined]
    else:
        col.metric(label, f"{value:.3f}")  # type: ignore[attr-defined]


def _severity_icon(severity: str) -> str:
    return {
        "CRITICAL": "🔴",
        "HIGH": "🟠",
        "MEDIUM": "🟡",
        "LOW": "🔵",
    }.get(severity.upper(), "⚪")


_render_sidebar()
_render_overview()
