"""Tests for OpenTelemetry instrumentation (:mod:`kompend.agents.instrument`)."""

from __future__ import annotations

import pytest

from kompend.agents import instrument
from kompend.agents.instrument import _parse_headers, configure_observability
from kompend.core.env import EnvVar
from tests.conftest import as_obj


def test_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """No OTel config when KOMP_OTEL_ENABLED is unset or false."""
    monkeypatch.delenv(EnvVar.OTEL_ENABLED, raising=False)
    monkeypatch.setattr("kompend.agents.instrument._instrumented", False)

    configure_observability()

    assert _flag() is False


def test_false_flag_disables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(EnvVar.OTEL_ENABLED, "false")
    monkeypatch.setattr("kompend.agents.instrument._instrumented", False)

    configure_observability()

    assert _flag() is False


def test_enabled_wires_exporter_and_instrumentall(monkeypatch: pytest.MonkeyPatch) -> None:
    """Enabling constructs an OTLP exporter, registers a TracerProvider, and
    calls Agent.instrument_all() with the configured service name/endpoint.
    """
    monkeypatch.setenv(EnvVar.OTEL_ENABLED, "true")
    monkeypatch.setenv(EnvVar.OTEL_ENDPOINT, "http://127.0.0.1:4318")
    monkeypatch.setenv(EnvVar.OTEL_SERVICE_NAME, "kompend-test")
    monkeypatch.setattr("kompend.agents.instrument._instrumented", False)

    calls: dict[str, object] = {}
    instrumented_flag: list[bool] = [False]

    class _FakeProvider:
        resource = type("R", (), {"attributes": {}})()

        def add_span_processor(self, processor: object) -> None:
            calls["processor_added"] = processor

    # Stub the exports so no real tracer provider is installed globally
    monkeypatch.setattr(
        "kompend.agents.instrument.OTLPSpanExporter",
        lambda **kw: calls.setdefault("exporter", kw),
    )
    monkeypatch.setattr("kompend.agents.instrument.BatchSpanProcessor", lambda e: e)
    monkeypatch.setattr("kompend.agents.instrument.TracerProvider", lambda **kw: _FakeProvider())
    monkeypatch.setattr(
        "kompend.agents.instrument.trace",
        type("T", (), {"set_tracer_provider": staticmethod(lambda p: instrumented_flag.__setitem__(0, True))})(),
    )
    monkeypatch.setattr(
        "kompend.agents.instrument.Agent",
        type("A", (), {"instrument_all": staticmethod(lambda: calls.setdefault("instrument_all", True))})(),
    )

    configure_observability()

    assert instrumented_flag[0] is True
    assert calls.get("instrument_all") is True
    exporter = as_obj(calls["exporter"])
    assert exporter["endpoint"] == "http://127.0.0.1:4318/v1/traces"
    assert exporter["headers"] is None
    assert _flag() is True


def test_enabled_with_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(EnvVar.OTEL_ENABLED, "true")
    monkeypatch.setenv(EnvVar.OTEL_ENDPOINT, "http://127.0.0.1:4318/")
    monkeypatch.setenv(EnvVar.OTEL_HEADERS, "Authorization=Bearer tok, X-Foo=bar")
    monkeypatch.setattr("kompend.agents.instrument._instrumented", False)

    exporter_kw: dict[str, object] = {}
    tracker: list[bool] = [False]

    class _FakeProvider:
        resource = type("R", (), {"attributes": {}})()

        def add_span_processor(self, processor: object) -> None:
            pass

    monkeypatch.setattr("kompend.agents.instrument.OTLPSpanExporter", lambda **kw: exporter_kw.update(kw))
    monkeypatch.setattr("kompend.agents.instrument.BatchSpanProcessor", lambda e: e)
    monkeypatch.setattr("kompend.agents.instrument.TracerProvider", lambda **kw: _FakeProvider())
    monkeypatch.setattr(
        "kompend.agents.instrument.trace",
        type("T", (), {"set_tracer_provider": staticmethod(lambda p: tracker.__setitem__(0, True))})(),
    )
    monkeypatch.setattr(
        "kompend.agents.instrument.Agent",
        type("A", (), {"instrument_all": staticmethod(lambda: None)})(),
    )

    configure_observability()

    # endpoint trailing slash stripped; headers parsed and sent
    assert exporter_kw["endpoint"] == "http://127.0.0.1:4318/v1/traces"
    assert exporter_kw["headers"] == {"Authorization": "Bearer tok", "X-Foo": "bar"}
    assert tracker[0] is True


def test_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(EnvVar.OTEL_ENABLED, "true")
    monkeypatch.setenv(EnvVar.OTEL_ENDPOINT, "http://127.0.0.1:1")
    monkeypatch.setattr("kompend.agents.instrument._instrumented", False)
    calls: list[str] = []

    class _FakeProvider:
        resource = type("R", (), {"attributes": {}})()

        def add_span_processor(self, processor: object) -> None:
            pass

    monkeypatch.setattr("kompend.agents.instrument.OTLPSpanExporter", lambda **kw: calls.append("exporter"))
    monkeypatch.setattr("kompend.agents.instrument.BatchSpanProcessor", lambda e: e)
    monkeypatch.setattr("kompend.agents.instrument.TracerProvider", lambda **kw: _FakeProvider())
    monkeypatch.setattr(
        "kompend.agents.instrument.trace",
        type("T", (), {"set_tracer_provider": staticmethod(lambda p: calls.append("provider"))})(),
    )
    monkeypatch.setattr(
        "kompend.agents.instrument.Agent",
        type("A", (), {"instrument_all": staticmethod(lambda: calls.append("instrument"))})(),
    )

    configure_observability()
    configure_observability()

    # only configured once despite two calls
    assert calls.count("exporter") == 1
    assert calls.count("provider") == 1
    assert calls.count("instrument") == 1
    assert _flag() is True


def test_disabled_leaves_no_global_tracer_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """When KOMP_OTEL_ENABLED is false/unset, configure_observability must not
    install a global tracer provider. Smoke check for the off-path.
    """
    monkeypatch.delenv(EnvVar.OTEL_ENABLED, raising=False)
    monkeypatch.setattr("kompend.agents.instrument._instrumented", False)

    calls: list[object] = []
    monkeypatch.setattr(
        "kompend.agents.instrument.trace",
        type("T", (), {"set_tracer_provider": staticmethod(calls.append)})(),
    )
    monkeypatch.setattr(
        "kompend.agents.instrument.Agent",
        type("A", (), {"instrument_all": staticmethod(lambda: calls.append("instrument"))})(),
    )

    configure_observability()

    # No tracer provider installed, no instrumentation, flag stays False
    assert calls == []
    assert _flag() is False


def _flag() -> bool:
    return instrument._instrumented


def test_headers_parse(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _parse_headers("") == {}
    assert _parse_headers("Authorization=Bearer tok") == {"Authorization": "Bearer tok"}
    assert _parse_headers("a=1, b=2") == {"a": "1", "b": "2"}
    # items without '=' are ignored
    assert _parse_headers("Authorization=Bearer tok, junk") == {"Authorization": "Bearer tok"}
