"""Defect Monitor page.

Recent defects feed + breakdown bar charts by severity and by defect type.
The same data is the source for the operator's "what's broken right now"
read.
"""

from __future__ import annotations

from collections import Counter

import pandas as pd
import streamlit as st

from dashboard.api_client import APIError, list_defects
from dashboard.auth import require_oidc

st.set_page_config(page_title="Defects", layout="wide")
require_oidc()
st.title("Defect Monitor")
st.caption("Quality issues detected by the in-pipeline defect detector.")


_DEFECT_TYPES = [
    "DEFECT_EMPTY_RETRIEVAL",
    "DEFECT_LOW_RETRIEVAL_QUALITY",
    "DEFECT_CONTEXT_TRUNCATED",
    "DEFECT_LOW_CHUNK_DIVERSITY",
    "DEFECT_HALLUCINATION_SIGNAL",
]


# ----------------------------------------------------------------- filters

with st.sidebar:
    st.header("Filters")
    severity_choice = st.selectbox(
        "Severity", ["any", "CRITICAL", "HIGH", "MEDIUM", "LOW"], index=0
    )
    defect_type_choice = st.selectbox("Defect type", ["any", *_DEFECT_TYPES], index=0)
    fetch_limit = st.selectbox("Sample size (for charts)", [50, 100, 200], index=1)

severity = None if severity_choice == "any" else severity_choice
defect_type = None if defect_type_choice == "any" else defect_type_choice


# ----------------------------------------------------------------- fetch

try:
    data = list_defects(limit=fetch_limit, severity=severity, defect_type=defect_type)
except APIError as exc:
    st.error(f"Could not fetch defects: {exc}")
    st.stop()

items = data.get("items", [])
total = data.get("total", 0)


# ----------------------------------------------------------------- charts

if not items:
    st.info("No defects match these filters.")
    st.stop()

st.subheader(f"Distribution (over the latest {len(items)} of {total} matching)")

chart_left, chart_right = st.columns(2)

with chart_left:
    sev_counts = Counter(d["severity"] for d in items)
    sev_df = pd.DataFrame(
        {"severity": list(sev_counts.keys()), "count": list(sev_counts.values())}
    ).sort_values("severity")
    st.bar_chart(sev_df, x="severity", y="count", height=240)

with chart_right:
    type_counts = Counter(d["defect_type"] for d in items)
    type_df = pd.DataFrame(
        {
            "defect_type": [k.removeprefix("DEFECT_") for k in type_counts],
            "count": list(type_counts.values()),
        }
    ).sort_values("count", ascending=False)
    st.bar_chart(type_df, x="defect_type", y="count", height=240)


# ----------------------------------------------------------------- feed

st.subheader("Recent defects")
for d in items[:50]:
    icon = {
        "CRITICAL": "🔴",
        "HIGH": "🟠",
        "MEDIUM": "🟡",
        "LOW": "🔵",
    }.get(d["severity"], "⚪")
    with st.container(border=True):
        st.markdown(
            f"**{icon} {d['defect_type']}** · `{d['severity']}` · "
            f"trace `{d['trace_id'][:12]}…` · query `{d['query_id'][:8]}…` · "
            f"{d['detected_at']}"
        )
        st.caption(d.get("description", ""))
        meta = d.get("metadata") or {}
        if meta:
            with st.expander("Metadata"):
                st.json(meta)
