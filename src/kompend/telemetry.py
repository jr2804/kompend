"""Logfire telemetry configuration.

This module owns ``import logfire`` and nothing else imports it directly. It is
loaded on demand — via :func:`importlib.import_module`, the mechanism the lazy
command group already uses — so a process that never configures telemetry never
pays for it.

``import logfire`` costs roughly 700 ms and pulls in ``requests``. At module scope
in ``cli/app.py`` that cost was paid by ``kompend --version``, ``kompend --help``
and every in-process import of the CLI (tests, MCP), none of which emit
telemetry.
"""

from __future__ import annotations

import os

import logfire

from kompend.core.keys import EnvVar

__all__ = ["configure_logfire"]


def configure_logfire() -> None:
    """Enable Logfire instrumentation once per process.

    ``configure()`` must run before ``instrument_pydantic_ai()``.
    """
    if os.environ.get(EnvVar.LOGFIRE_DISABLED, "").strip().lower() in ("1", "true", "yes"):
        return
    # logfire fans telemetry out to standard OTLP exporters when the ambient
    # OTEL_EXPORTER_OTLP_ENDPOINT env var is set (logfire config.py:1469).
    # Each signal (traces/metrics/logs) has its own switch, all defaulting to
    # enabled. A stale collector endpoint from unrelated shell tooling
    # (e.g. Copilot's local collector, injected per-window into VS Code
    # terminals - never on disk) then receives kompend telemetry and retries
    # against a dead port. kompend has its own collector path
    # (agents/instrument.py, gated on KOMP_OTEL_ENABLED), so ambient OTel vars
    # must not divert logfire telemetry. setdefault keeps the escape hatch:
    # OTEL_TRACES_EXPORTER=otlp (etc.) explicitly re-enables fan-through.
    os.environ.setdefault("OTEL_TRACES_EXPORTER", "none")
    os.environ.setdefault("OTEL_METRICS_EXPORTER", "none")
    os.environ.setdefault("OTEL_LOGS_EXPORTER", "none")
    # Quiet the logfire console span exporter: it prints every instrumented
    # span/log to the terminal (config.py:1206-1223) — "running tool: …",
    # "chat …", per-agent spans — drowning CLI progress output. Passing
    # console=False hard-overrides LOGFIRE_* env vars (config.py:811 only
    # consults the env when console=None), so an explicitly set
    # LOGFIRE_CONSOLE=true is honored instead by passing console=None.
    # Backend telemetry (spans/logs/metrics to OTLP or logfire.dev) is
    # untouched either way.
    console = (
        None  # logfire's own env-driven default applies (console on)
        if os.environ.get("LOGFIRE_CONSOLE", "").strip().lower() in ("1", "true", "yes")
        else False
    )
    logfire.configure(
        service_name="kompend",
        environment="staging",
        send_to_logfire="if-token-present",
        console=console,
    )
    logfire.instrument_pydantic_ai()
