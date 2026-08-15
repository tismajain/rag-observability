"""New Query page — fire a /query and render the live response.

This is the write surface against the production pipeline: prompt → retrieve
→ generate → defect-detect → (sampled async eval). Everything that runs is
visible in Phoenix and in the Traces page within a few seconds.
"""

from __future__ import annotations

import streamlit as st

from dashboard.api_client import APIError, post_query
from dashboard.auth import require_oidc

st.set_page_config(page_title="New Query", layout="wide")
require_oidc()
st.title("Run a query")
st.caption(
    "Fires POST /query. The response renders below; the trace is also "
    "captured in Phoenix and persisted to the Postgres trace store."
)


with st.sidebar:
    st.header("Query settings")
    top_k = st.slider("top_k", min_value=1, max_value=20, value=5)
    rerank_choice = st.selectbox(
        "Reranker",
        ["server default", "force on", "force off"],
        index=0,
    )

rerank_flag: bool | None = {
    "server default": None,
    "force on": True,
    "force off": False,
}[rerank_choice]


with st.form("query_form"):
    query_text = st.text_area("Query", height=100, placeholder="Ask the RAG pipeline a question…")
    submitted = st.form_submit_button("Run query", type="primary")

if submitted:
    if not query_text.strip():
        st.warning("Enter a query first.")
        st.stop()

    with st.spinner("Calling /query…"):
        try:
            resp = post_query(query_text, top_k=top_k, enable_reranking=rerank_flag)
        except APIError as exc:
            st.error(f"Query failed: {exc}")
            st.stop()

    st.success(
        f"trace_id `{resp['trace_id']}` · latency {resp['latency_ms']}ms · "
        f"defects: {len(resp.get('defects_detected', []))} · "
        f"eval scheduled: {resp.get('eval_scheduled', False)}"
    )

    st.subheader("Answer")
    st.markdown(resp["answer"])

    if resp.get("context_truncated"):
        st.warning("Context was truncated — the retrieved chunks exceeded the token budget.")

    defects = resp.get("defects_detected", [])
    if defects:
        st.subheader("Defects detected")
        for d in defects:
            st.markdown(f"- `{d}`")

    st.subheader("Retrieved chunks")
    for c in resp.get("retrieved_chunks", []):
        with st.container(border=True):
            scores = [f"rrf={c['rrf_score']:.4f}"]
            if c.get("dense_score") is not None:
                scores.append(f"dense={c['dense_score']:.4f}")
            if c.get("sparse_score") is not None:
                scores.append(f"sparse={c['sparse_score']:.4f}")
            st.markdown(
                f"**{c['source_file']}** · chunk {c['chunk_index']} · " + " · ".join(scores)
            )
            st.caption(c.get("text_preview", ""))
