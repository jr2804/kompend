"""The ACP session asks pydantic-ai to cache the conversation prefix."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from pydantic_ai.capabilities import Caching

from kompend.acp import harness_adapter


def test_session_attaches_the_caching_capability(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    _patch(monkeypatch, prompt_cache=True)

    config = harness_adapter.kompend_session_config(_session(tmp_path))

    assert config.capabilities is not None
    assert any(isinstance(capability, Caching) for capability in config.capabilities)


def test_session_skips_caching_when_disabled(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    _patch(monkeypatch, prompt_cache=False)

    config = harness_adapter.kompend_session_config(_session(tmp_path))

    assert config.capabilities is None


def _session(tmp_path: Any) -> SimpleNamespace:
    return SimpleNamespace(cwd=tmp_path)


def _patch(monkeypatch: pytest.MonkeyPatch, *, prompt_cache: bool) -> None:
    definition = SimpleNamespace(tools=SimpleNamespace(enabled=[]))
    monkeypatch.setattr(harness_adapter, "AgentRegistry", lambda base_dir: SimpleNamespace(active_for=lambda task, overrides: definition))
    monkeypatch.setattr(harness_adapter, "build_toolset", lambda **kwargs: SimpleNamespace())
    llm = SimpleNamespace(prompt_cache=prompt_cache)
    resolved = SimpleNamespace(kompend=SimpleNamespace(compile=SimpleNamespace(llm=llm)))
    monkeypatch.setattr(harness_adapter, "resolve_config", lambda **kwargs: resolved)
