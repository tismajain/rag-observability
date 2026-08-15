"""Provider-neutral Streamlit OIDC gate.

Streamlit owns the authorization-code flow configured in `.streamlit/secrets.toml`.
Only the short-lived access token is retained in per-session memory and forwarded to
the API. It is never rendered, logged, persisted, or copied into URLs.
"""

from __future__ import annotations

import os

import streamlit as st


def require_oidc() -> None:
    if os.environ.get("AUTH__ENABLED", "false").lower() != "true":
        return
    if not getattr(st.user, "is_logged_in", False):
        st.title("Sign in")
        if st.button("Sign in with OpenID Connect", type="primary"):
            st.login()
        st.stop()
    token = getattr(st.user, "access_token", None)
    if not isinstance(token, str) or not token:
        st.error(
            "The configured OIDC integration did not provide an API access token. "
            "Configure the provider/API audience and Streamlit token forwarding."
        )
        if st.button("Sign out"):
            st.logout()
        st.stop()
    st.session_state["_api_access_token"] = token


def authorization_header() -> dict[str, str]:
    token = st.session_state.get("_api_access_token")
    if not isinstance(token, str) or not token:
        return {}
    return {"Authorization": f"Bearer {token}"}
