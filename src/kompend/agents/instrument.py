"""OpenTelemetry instrumentation for pydantic-ai agents.

Env-gated — zero behavior change when off.
Enabled via ``KOMP_OTEL_ENABLED=true`` plus an OTLP endpoint.

Example::

    export KOMP_OTEL_ENABLED=true
    export KOMP_OTEL_ENDPOINT=http://localhost:4318
    export KOMP_OTEL_SERVICE_NAME=kompend
    export KOMP_OTEL_HEADERS="Authorization=Bearer token"
    kompend compile

OpenTelemetry SDKs are optional and only required when the feature is
enabled. Install with ``uv sync --extra otel`` or ``uv add "kompend[otel]"``.
When disabled, this module has zero overhead — the tracer provider is never
constructed.

See ``docs/guides/observability.md`` for the full guide.
"""

from __future__ import annotations

import logging
import os

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from pydantic_ai import Agent

from kompend.core.env import EnvVar

logger = logging.getLogger(__name__)

# ── idempotency guard ──────────────────────────────────────────────────────

_instrumented: bool = False


# ── public API ──────────────────────────────────────────────────────────────


def configure_observability() -> None:
    """Configure OpenTelemetry tracing for all pydantic-ai agents.

    Called lazily from :meth:`KompendAgentBase._build_agent` on first agent
    construction. Safe to call multiple times — only configures once.

    Reads the following environment variables (via :class:`EnvVar`):

    - ``KOMP_OTEL_ENABLED`` — ``true|1|yes`` to enable. Any other value (including
      unset) disables tracing with zero overhead.
    - ``KOMP_OTEL_ENDPOINT`` — OTLP HTTP endpoint (default: ``http://localhost:4318``).
      For Langfuse self-hosted: ``http://localhost:3000/api/otel``.
    - ``KOMP_OTEL_SERVICE_NAME`` — service name in the OTel UI (default: ``kompend``).
    - ``KOMP_OTEL_HEADERS`` — optional HTTP headers for the OTLP endpoint
      (``key=value`` format, used for auth tokens).
    """  # see docs/guides/observability.md
    global _instrumented  # noqa: PLW0603

    if _instrumented:
        return

    enabled = os.environ.get(EnvVar.OTEL_ENABLED, "").strip().lower()
    if enabled not in ("1", "true", "yes"):
        logger.debug("OTel tracing disabled (KOMP_OTEL_ENABLED not set or false)")
        return

    endpoint = os.environ.get(EnvVar.OTEL_ENDPOINT, "http://localhost:4318").rstrip("/")
    service_name = os.environ.get(EnvVar.OTEL_SERVICE_NAME, "kompend")
    headers = _parse_headers(os.environ.get(EnvVar.OTEL_HEADERS, ""))

    resource = Resource.create({SERVICE_NAME: service_name})
    exporter = OTLPSpanExporter(
        endpoint=f"{endpoint}/v1/traces",
        headers=headers or None,
    )
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    # Enable pydantic-ai OTel instrumentation (all agents from this point)
    Agent.instrument_all()

    logger.info("OTel tracing enabled -> %s  (service: %s)", endpoint, service_name)
    _instrumented = True


def _parse_headers(raw: str) -> dict[str, str]:
    """Parse ``key=value,key=value`` header string into a dict."""
    headers: dict[str, str] = {}
    for raw_pair in raw.split(","):
        pair = raw_pair.strip()
        if not pair:
            continue
        if "=" in pair:
            key, value = pair.split("=", 1)
            headers[key.strip()] = value.strip()
    return headers
