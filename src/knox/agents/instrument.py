"""OpenTelemetry instrumentation for pydantic-ai agents.

Env-gated — zero behavior change when off.
Enabled via ``KNOX_OTEL_ENABLED=true`` plus an OTLP endpoint.

Example::

    export KNOX_OTEL_ENABLED=true
    export KNOX_OTEL_ENDPOINT=http://localhost:4318
    export KNOX_OTEL_SERVICE_NAME=knox
    export KNOX_OTEL_HEADERS="Authorization=Bearer token"
    knox compile

The OpenTelemetry SDKs are imported at module level: they are guaranteed
present because ``logfire`` (a runtime dependency) pulls in
``opentelemetry-sdk`` and the OTLP HTTP exporter, which we also pin
explicitly in the optional ``otel`` extra. No ``try/except ImportError``
guard is needed.
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

from knox.core.env import EnvVar

logger = logging.getLogger(__name__)

# ── idempotency guard ──────────────────────────────────────────────────────

_instrumented: bool = False


# ── public API ──────────────────────────────────────────────────────────────


def configure_observability() -> None:
    """Configure OpenTelemetry tracing for all pydantic-ai agents.

    Called lazily from :meth:`KnoxAgentBase._build_agent` on first agent
    construction. Safe to call multiple times — only configures once.

    Reads the following environment variables (via :class:`EnvVar`):

    - ``KNOX_OTEL_ENABLED`` — ``true|1|yes`` to enable. Any other value (including
      unset) disables tracing with zero overhead.
    - ``KNOX_OTEL_ENDPOINT`` — OTLP HTTP endpoint (default: ``http://localhost:4318``).
      For Langfuse self-hosted: ``http://localhost:3000/api/otel``.
    - ``KNOX_OTEL_SERVICE_NAME`` — service name in the OTel UI (default: ``knox``).
    - ``KNOX_OTEL_HEADERS`` — optional HTTP headers for the OTLP endpoint
      (``key=value`` format, used for auth tokens).
    """
    global _instrumented  # noqa: PLW0603

    if _instrumented:
        return

    enabled = os.environ.get(EnvVar.OTEL_ENABLED, "").strip().lower()
    if enabled not in ("1", "true", "yes"):
        logger.debug("OTel tracing disabled (KNOX_OTEL_ENABLED not set or false)")
        return

    endpoint = os.environ.get(EnvVar.OTEL_ENDPOINT, "http://localhost:4318").rstrip("/")
    service_name = os.environ.get(EnvVar.OTEL_SERVICE_NAME, "knox")
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
