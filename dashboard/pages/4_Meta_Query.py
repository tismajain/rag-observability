"""Meta Query page — natural-language Q&A over trace history.

Calls POST /meta/query. The meta pipeline indexes Trace + DefectEvent +
EvalScore rows into a separate Qdrant collection ("rag_traces") and lets
an LLM answer "what's been happening in the system" questions over it.
"""

from __future__ import annotations

import streamlit as st

from dashboard.api_client import APIError, post_meta_query
from dashboard.auth import require_oidc

st.set_page_config(page_title="Meta Query", layout="wide")
require_oidc()
st.title("Meta Query — ask the system about itself")
st.caption(
    "Natural-language Q&A over indexed trace history. Useful for triaging "
    "patterns ('which defects are firing most?') or recall ('what did we "
    "answer about X last week?')."
)


with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Trace docs to retrieve", min_value=3, max_value=20, value=8)


if "meta_history" not in st.session_state:
    st.session_state["meta_history"] = []  # list[(role, content)]

# ----------------------------------------------------------------- chat replay

for role, content in st.session_state["meta_history"]:
    with st.chat_message(role):
        st.markdown(content)

# ----------------------------------------------------------------- input

prompt = st.chat_input("Ask about past queries, defects, or eval trends…")
if prompt:
    st.session_state["meta_history"].append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    with (
        st.chat_message("assistant"),
        st.spinner("Querying meta-RAG…"),
    ):
        try:
            resp = post_meta_query(prompt, top_k=top_k)
        except APIError as exc:
            err = f"Meta query failed: {exc}"
            st.error(err)
            st.session_state["meta_history"].append(("assistant", err))
        else:
            st.markdown(resp["answer"])
            st.session_state["meta_history"].append(("assistant", resp["answer"]))

            with st.expander(
                f"Retrieved {len(resp.get('retrieved_traces', []))} trace docs · "
                f"latency {resp.get('latency_ms', 0)}ms"
            ):
                for t in resp.get("retrieved_traces", []):
                    st.markdown(f"- `{t['query_id'][:8]}…` · rrf={t['rrf_score']:.4f}")
                    st.caption(t.get("text_preview", ""))


if st.session_state["meta_history"] and st.button("Clear conversation"):
    st.session_state["meta_history"] = []
    st.rerun()
