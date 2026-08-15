"""Thin HTTP client used by the Streamlit pages.

Centralizes:

* The base URL (``API_BASE_URL`` env var; defaults to localhost for dev runs
  outside Docker).
* httpx connection reuse via a single module-level client.
* Defensive error handling — the dashboard pages catch the exceptions raised
  here and render a Streamlit warning instead of stack-tracing the user.

Streamlit reruns the script top-to-bottom on each interaction, so the client
is created once per process and shared across reruns via Python's module
import cache.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from dashboard.auth import authorization_header

API_BASE_URL: str = os.environ.get("API_BASE_URL", "http://localhost:8000")
DEFAULT_TIMEOUT_SECONDS = 60.0


class APIError(RuntimeError):
    """Raised when the API returns a non-2xx response or is unreachable."""


_client: httpx.Client | None = None


def _get_client() -> httpx.Client:
    global _client
    if _client is None:
        _client = httpx.Client(
            base_url=API_BASE_URL,
            timeout=DEFAULT_TIMEOUT_SECONDS,
        )
    return _client


def _get(path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    try:
        resp = _get_client().get(path, params=params or {}, headers=authorization_header())
    except httpx.RequestError as exc:
        raise APIError(f"API request failed: {exc}") from exc
    if resp.status_code >= 400:
        raise APIError(_safe_api_error(resp))
    data: dict[str, Any] = resp.json()
    return data


def _post(path: str, json: dict[str, Any]) -> dict[str, Any]:
    try:
        resp = _get_client().post(path, json=json, headers=authorization_header())
    except httpx.RequestError as exc:
        raise APIError(f"API request failed: {exc}") from exc
    if resp.status_code >= 400:
        raise APIError(_safe_api_error(resp))
    data: dict[str, Any] = resp.json()
    return data


# ----------------------------------------------------------------- health


def get_health() -> dict[str, Any]:
    return _get("/health/ready")


def _safe_api_error(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return f"API returned HTTP {response.status_code}"
    error = body.get("error", "http_error") if isinstance(body, dict) else "http_error"
    request_id = body.get("request_id") if isinstance(body, dict) else None
    suffix = f" (request {request_id})" if isinstance(request_id, str) else ""
    return f"API returned HTTP {response.status_code}: {error}{suffix}"


# ----------------------------------------------------------------- traces


def list_traces(
    limit: int = 50,
    offset: int = 0,
    has_defects: bool | None = None,
    quality_gate: str | None = None,
) -> dict[str, Any]:
    params: dict[str, Any] = {"limit": limit, "offset": offset}
    if has_defects is not None:
        params["has_defects"] = str(has_defects).lower()
    if quality_gate is not None:
        params["quality_gate"] = quality_gate
    return _get("/traces", params=params)


def get_trace(trace_id: str) -> dict[str, Any]:
    return _get(f"/traces/{trace_id}")


# ----------------------------------------------------------------- defects


def list_defects(
    limit: int = 50,
    offset: int = 0,
    severity: str | None = None,
    defect_type: str | None = None,
) -> dict[str, Any]:
    params: dict[str, Any] = {"limit": limit, "offset": offset}
    if severity:
        params["severity"] = severity
    if defect_type:
        params["defect_type"] = defect_type
    return _get("/defects", params=params)


# ----------------------------------------------------------------- evals


def list_evals(query_id: str | None = None, limit: int = 50, offset: int = 0) -> dict[str, Any]:
    params: dict[str, Any] = {"limit": limit, "offset": offset}
    if query_id:
        params["query_id"] = query_id
    return _get("/evals", params=params)


def evals_summary(period: str = "7d") -> dict[str, Any]:
    return _get("/evals/summary", params={"period": period})


# ----------------------------------------------------------------- query


def post_query(query: str, top_k: int = 5, enable_reranking: bool | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"query": query, "top_k": top_k}
    if enable_reranking is not None:
        payload["enable_reranking"] = enable_reranking
    return _post("/query", json=payload)


def post_meta_query(query: str, top_k: int = 8) -> dict[str, Any]:
    return _post("/meta/query", json={"query": query, "top_k": top_k})
