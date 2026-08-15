"""Tests for the OTel tracer setup.

We verify:
* :func:`get_tracer` returns a working (no-op) tracer even before setup —
  this is what made the forward-design pattern work in earlier phases.
* :func:`setup_tracing` is idempotent: calling it twice is a no-op.
* :func:`shutdown_tracing` cleanly tears the provider down and is idempotent.
"""

from __future__ import annotations

from unittest.mock import patch

from src.observability import tracer as tracer_module
from src.observability.tracer import get_tracer, setup_tracing, shutdown_tracing


def _reset_provider() -> None:
    """Force the module-level provider back to ``None`` so each test runs clean."""
    tracer_module._provider = None  # type: ignore[attr-defined]


def test_get_tracer_returns_noop_before_setup() -> None:
    _reset_provider()
    tracer = get_tracer("test.noop")
    # The OTel SDK's no-op tracer accepts start_as_current_span just like a real one.
    with tracer.start_as_current_span("dummy") as span:
        # No-op span has an invalid span context, but the call must not raise.
        ctx = span.get_span_context()
        assert ctx is not None


def test_setup_tracing_is_idempotent() -> None:
    _reset_provider()
    # Patch the OTLP exporter so the test doesn't try to open a gRPC channel.
    with patch("src.observability.tracer.OTLPSpanExporter"):
        setup_tracing()
        first_provider = tracer_module._provider  # type: ignore[attr-defined]
        setup_tracing()
        second_provider = tracer_module._provider  # type: ignore[attr-defined]

    assert first_provider is second_provider
    shutdown_tracing()


def test_shutdown_tracing_is_idempotent() -> None:
    _reset_provider()
    # Two calls in a row with no setup → both no-ops, no exceptions.
    shutdown_tracing()
    shutdown_tracing()
    assert tracer_module._provider is None  # type: ignore[attr-defined]


def test_setup_then_shutdown_clears_provider() -> None:
    _reset_provider()
    with patch("src.observability.tracer.OTLPSpanExporter"):
        setup_tracing()
        assert tracer_module._provider is not None  # type: ignore[attr-defined]
        shutdown_tracing()
        assert tracer_module._provider is None  # type: ignore[attr-defined]


def test_get_tracer_works_after_setup_with_real_spans() -> None:
    _reset_provider()
    with patch("src.observability.tracer.OTLPSpanExporter"):
        setup_tracing()
        tracer = get_tracer("test.real")
        with tracer.start_as_current_span("real-span") as span:
            ctx = span.get_span_context()
            # With a configured provider, the span context must be valid.
            assert ctx.is_valid
            assert ctx.trace_id != 0
        shutdown_tracing()
