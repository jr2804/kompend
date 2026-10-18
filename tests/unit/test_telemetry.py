"""Tests for logfire telemetry configuration (src/kompend/telemetry.py)."""

from __future__ import annotations

from typing import Any

import pytest

from kompend import telemetry


class _LogfireRecorder:
    """Captures logfire.configure() kwargs without touching global state."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def configure(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)

    def instrument_pydantic_ai(self) -> None:
        return None


class TestConfigureLogfire:
    def test_console_disabled_by_default(self, recorder: _LogfireRecorder, monkeypatch: pytest.MonkeyPatch) -> None:
        telemetry.configure_logfire()
        assert len(recorder.calls) == 1
        call = recorder.calls[0]
        assert call["console"] is False
        # Scope guard: CLI telemetry keeps its identity and opt-in backend.
        assert call["service_name"] == "kompend"
        assert call["send_to_logfire"] == "if-token-present"

    def test_logfire_console_env_opt_back_in(self, recorder: _LogfireRecorder, monkeypatch: pytest.MonkeyPatch) -> None:
        """LOGFIRE_CONSOLE=true explicitly re-enables logfire's own env-driven console."""
        monkeypatch.setenv("LOGFIRE_CONSOLE", "true")
        telemetry.configure_logfire()
        assert recorder.calls[0]["console"] is None

    def test_disabled_env_short_circuits(self, recorder: _LogfireRecorder, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KOMP_LOGFIRE_DISABLED", "1")
        telemetry.configure_logfire()
        assert recorder.calls == []


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> _LogfireRecorder:
    logfire_recorder = _LogfireRecorder()
    monkeypatch.setattr(telemetry.logfire, "configure", logfire_recorder.configure)
    monkeypatch.setattr(telemetry.logfire, "instrument_pydantic_ai", logfire_recorder.instrument_pydantic_ai)
    return logfire_recorder


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KOMP_LOGFIRE_DISABLED", raising=False)
    monkeypatch.delenv("LOGFIRE_CONSOLE", raising=False)
