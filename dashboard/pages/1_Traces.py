"""Trace Explorer page.

Paginated table of past /query interactions, with filters for defect presence
and eval quality gate. Click a trace_id to drill into its full record —
question, answer, defects, eval scores — fetched from /traces/{trace_id}.
"""

from __future__ import annotations

import streamlit as st

from dashboard.api_client import APIError, get_trace, list_traces
from dashboard.auth import require_oidc

st.set_page_config(page_title="Traces", layout="wide")
require_oidc()
st.title("Trace Explorer")
st.caption("Browse past /query interactions. Click a row to expand the full trace.")


def _fmt_score(value: float | None) -> str:
    return f"{value:.3f}" if value is not None else "—"


# ----------------------------------------------------------------- filters

with st.sidebar:
    st.header("Filters")
    page_size = st.selectbox("Page size", [10, 25, 50, 100], index=1)
    has_defects_choice = st.selectbox("Has defects?", ["any", "yes", "no"], index=0)
    quality_gate_choice = st.selectbox(
        "Quality gate", ["any", "pass", "warn", "fail", "skip"], index=0
    )

if "trace_page" not in st.session_state:
    st.session_state["trace_page"] = 0

page = int(st.session_state["trace_page"])
has_defects: bool | None = (
    True if has_defects_choice == "yes" else False if has_defects_choice == "no" else None
)
quality_gate = None if quality_gate_choice == "any" else quality_gate_choice

# ----------------------------------------------------------------- fetch

try:
    data = list_traces(
        limit=page_size,
        offset=page * page_size,
        has_defects=has_defects,
        quality_gate=quality_gate,
    )
except APIError as exc:
    st.error(f"Could not fetch traces: {exc}")
    st.stop()

items = data.get("items", [])
total = data.get("total", 0)

# ----------------------------------------------------------------- pager

pager_l, pager_m, pager_r = st.columns([1, 6, 1])
with pager_l:
    if st.button("← Prev", disabled=page == 0):
        st.session_state["trace_page"] = max(0, page - 1)
        st.rerun()
with pager_m:
    last_page = max(0, (total - 1) // page_size)
    st.caption(f"Showing {len(items)} of {total} · page {page + 1} / {last_page + 1}")
with pager_r:
    if st.button("Next →", disabled=(page + 1) * page_size >= total):
        st.session_state["trace_page"] = page + 1
        st.rerun()

# ----------------------------------------------------------------- table

if not items:
    st.info("No traces match these filters.")
else:
    for t in items:
        trace_id = t["trace_id"]
        title = f"`{trace_id[:12]}…` · {t['created_at']} · {t['model_used']} · {t['latency_ms']}ms"
        with st.expander(title):
            st.markdown(f"**Question:** {t['query_text']}")
            st.markdown(f"**Answer:** {t['answer_text']}")
            meta_cols = st.columns(4)
            meta_cols[0].metric("Top-K", t["top_k"])
            meta_cols[1].metric("Chunks retrieved", t["chunks_retrieved"])
            meta_cols[2].metric("Latency (ms)", t["latency_ms"])
            meta_cols[3].metric("Truncated", "yes" if t["context_truncated"] else "no")

            with st.spinner("Loading defects + eval scores…"):
                try:
                    detail = get_trace(trace_id)
                except APIError as exc:
                    st.warning(f"Could not load full trace: {exc}")
                    continue

            defects = detail.get("defects", [])
            evals = detail.get("evals", [])

            if defects:
                st.markdown("**Defects**")
                for d in defects:
                    st.markdown(
                        f"- `{d['severity']}` **{d['defect_type']}** — {d.get('description', '')}"
                    )
            else:
                st.caption("No defects.")

            if evals:
                st.markdown("**Eval scores**")
                for ev in evals:
                    cols = st.columns(4)
                    cols[0].metric(
                        "Faithfulness",
                        _fmt_score(ev.get("faithfulness")),
                    )
                    cols[1].metric(
                        "Context recall",
                        _fmt_score(ev.get("context_recall")),
                    )
                    cols[2].metric(
                        "Answer relevancy",
                        _fmt_score(ev.get("answer_relevancy")),
                    )
                    cols[3].metric(
                        "Quality gate",
                        ev.get("quality_gate_result", "—"),
                    )
            else:
                st.caption("Not evaluated.")
