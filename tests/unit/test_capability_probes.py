"""Tests for the doctor capability probes and the compact-format codecs."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from kompend.core.config._configs import LlmDataFormat
from kompend.diagnostics import capabilities as _capabilities
from kompend.llm.format import decode_from_llm, encode_for_llm

_MODEL = "test/model"


class _FakeRun:
    def __init__(self, output: object) -> None:
        self.output = output


class _FakeAgent:
    """Agent stand-in that replays the outputs the test queued.

    A queued callable is invoked instead of being returned — the tool-use
    probe uses that to pretend the model made the call.
    """

    queued: list[object] = []

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        self._tools: list[Callable[[object], str]] = []

    def tool_plain(self, fn: Callable[[object], str], *, name: str = "", description: str = "") -> Callable[[object], str]:
        self._tools.append(fn)
        return fn

    def run_sync(self, _prompt: str, **_kwargs: object) -> _FakeRun:
        nxt = _FakeAgent.queued.pop(0) if _FakeAgent.queued else ""
        if callable(nxt):
            for tool in self._tools:
                tool(None)
            return _FakeRun("")
        return _FakeRun(nxt)


class TestStructuredOutputProbes:
    def test_native_probe_accepts_a_valid_answer(self, queued: list[object]) -> None:
        queued.append(_capabilities._ProbeAnswer(ok=True, message="hello"))

        probe = _capabilities.probe_native_structured_output(_MODEL, timeout=5.0, retries=1)

        assert probe.ok
        assert probe.name == "structured-output"

    def test_native_probe_rejects_an_invalid_answer(self, queued: list[object]) -> None:
        queued.append(_capabilities._ProbeAnswer(ok=False, message="nope"))

        probe = _capabilities.probe_native_structured_output(_MODEL, timeout=5.0, retries=1)

        assert not probe.ok

    def test_prompted_probe_accepts_a_valid_answer(self, queued: list[object]) -> None:
        queued.append(_capabilities._ProbeAnswer(ok=True, message="hello"))

        probe = _capabilities.probe_prompted_structured_output(_MODEL, timeout=5.0, retries=1)

        assert probe.ok

    def test_raw_json_probe_accepts_json(self, queued: list[object]) -> None:
        queued.append('{"ok": true, "message": "hello"}')

        probe = _capabilities.probe_raw_json(_MODEL, timeout=5.0, retries=1)

        assert probe.ok

    def test_raw_json_probe_rejects_non_json(self, queued: list[object]) -> None:
        queued.append("not json at all")

        probe = _capabilities.probe_raw_json(_MODEL, timeout=5.0, retries=1)

        assert not probe.ok
        assert "json" in probe.detail.lower()

    def test_tool_use_probe_passes_when_a_tool_is_called(self, queued: list[object]) -> None:
        queued.append(lambda: None)  # the fake runs registered tools on a callable

        probe = _capabilities.probe_tool_use(_MODEL, timeout=5.0, retries=1)

        assert probe.ok
        assert probe.name == "tool-use"

    def test_tool_use_probe_fails_without_a_call(self, queued: list[object]) -> None:
        queued.append("plain answer, no tool")

        probe = _capabilities.probe_tool_use(_MODEL, timeout=5.0, retries=1)

        assert not probe.ok
        assert "without calling" in probe.detail


class TestCompactFormatProbe:
    def test_toon_answer_decodes(self, queued: list[object]) -> None:
        queued.append("[2]:\n  - title: alpha\n    score: 4\n  - title: beta\n    score: 5")

        probe = _capabilities.probe_compact_format(_MODEL, LlmDataFormat.TOON, timeout=5.0, retries=1)

        assert probe.ok
        assert "toon" in probe.detail

    def test_undecodable_answer_fails(self, queued: list[object]) -> None:
        queued.append("not a toon document at all")

        probe = _capabilities.probe_compact_format(_MODEL, LlmDataFormat.TOON, timeout=5.0, retries=1)

        assert not probe.ok


class TestMeasureCompactFormats:
    def test_every_format_is_reported(self) -> None:
        probe = _capabilities.measure_compact_formats()

        assert probe.ok
        assert "json" in probe.detail
        assert "toon" in probe.detail
        assert "ison" in probe.detail


class TestFormatRoundTrip:
    @pytest.mark.parametrize("fmt", [LlmDataFormat.JSON, LlmDataFormat.TOON])
    def test_round_trip_keeps_the_records(self, fmt: LlmDataFormat) -> None:
        records = [{"title": "alpha", "score": 4}, {"title": "beta", "score": 5}]

        encoded = encode_for_llm(records, fmt)

        assert decode_from_llm(encoded, fmt) == records

    def test_ison_round_trip_keeps_the_table_rows(self) -> None:
        records = [{"title": "alpha", "score": 4}, {"title": "beta", "score": 5}]

        encoded = encode_for_llm(records, LlmDataFormat.ISON)
        decoded = decode_from_llm(encoded, LlmDataFormat.ISON)

        assert decoded is not None
        assert isinstance(decoded, list)
        first = decoded[0]
        assert isinstance(first, dict)
        assert first["title"] == "alpha"

    def test_broken_text_decodes_to_none(self) -> None:
        assert decode_from_llm("}{ not a format", LlmDataFormat.TOON) is None
        assert decode_from_llm("}{ not a format", LlmDataFormat.ISON) is None
        assert decode_from_llm("", LlmDataFormat.JSON) is None


@pytest.fixture
def queued() -> list[object]:
    _FakeAgent.queued = []
    return _FakeAgent.queued


@pytest.fixture(autouse=True)
def fake_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_capabilities, "Agent", _FakeAgent)
    monkeypatch.setattr(_capabilities, "build_model", lambda name: None)
