"""Regression test: the tool-use probe must actually run, not fail at registration.

`tool_plain` registers a context-free tool. Annotating the probe tool with
`RunContext` made pydantic-ai refuse the schema for EVERY model with
"RunContext annotations can only be used with tools that take context", so the
check could never pass; `_short()` then hid the reason behind a trailing
colon. These two tests exercise the real `Agent` (no fake), so a registration
error surfaces here instead of in a doctor run.
"""

from __future__ import annotations

import pytest
from pydantic_ai.models.test import TestModel

from kompend.diagnostics import capabilities as _capabilities


def test_the_tool_use_probe_registers_a_context_free_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    """Registration is the failure mode: assert the probe reaches the model."""
    monkeypatch.setattr(_capabilities, "build_model", lambda _model: TestModel())
    monkeypatch.setattr(_capabilities, "ModelSettings", lambda **_kwargs: None)

    probe = _capabilities.probe_tool_use("test/model", timeout=5.0, retries=1)

    assert probe.ok, probe.detail
    assert "called a tool" in probe.detail


def test_short_keeps_the_reason_from_the_second_line() -> None:
    """A pydantic-ai UserError puts the actual reason on line two."""
    exc = RuntimeError("Error generating schema for _ping:\n  RunContext annotations can only be used with tools that take context")

    rendered = _capabilities._short(exc)

    assert "RunContext annotations" in rendered
    assert rendered.endswith("take context")
