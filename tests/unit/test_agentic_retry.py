"""Classification of failed LLM calls and the generation retry policy."""

from __future__ import annotations

from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import MagicMock

import httpx
import pytest
from pydantic_ai.exceptions import ModelHTTPError

from kompend.agents.base import KompendAgentBase
from kompend.agents.deps import PipelineAgentDeps
from kompend.core.config._configs import LLMConfig
from kompend.core.utils.exceptions import LLMOutputError, LLMTransientError
from kompend.llm.errors import is_transient_error


@pytest.mark.parametrize("status", [408, 409, 425, 429, 500, 502, 503, 504, 529])
def test_retryable_statuses_are_transient(status: int) -> None:
    assert is_transient_error(ModelHTTPError(status, "model", body={})) is True


@pytest.mark.parametrize("status", [400, 401, 404, 422])
def test_request_errors_are_not_transient(status: int) -> None:
    """A 4xx is a property of the request - a "timed out" wording does not change that."""
    assert is_transient_error(ModelHTTPError(status, "model", body={"message": "Request timed out."})) is False


def test_transport_failures_are_transient() -> None:
    assert is_transient_error(httpx.ReadTimeout("slow")) is True
    assert is_transient_error(httpx.ConnectError("refused")) is True
    assert is_transient_error(ConnectionError("reset")) is True
    assert is_transient_error(TimeoutError()) is True


def test_cause_chain_is_followed() -> None:
    wrapper = RuntimeError("agent failed")
    wrapper.__cause__ = httpx.ReadTimeout("slow")
    assert is_transient_error(wrapper) is True


def test_status_attribute_is_respected() -> None:
    class ProviderError(Exception):
        def __init__(self, status_code: int) -> None:
            super().__init__("boom")
            self.status_code = status_code

    assert is_transient_error(ProviderError(503)) is True
    assert is_transient_error(ProviderError(422)) is False


def test_plain_errors_are_not_transient() -> None:
    assert is_transient_error(ValueError("Request timed out.")) is False


class _StubAgent(KompendAgentBase[str, PipelineAgentDeps]):
    """Minimal agent: no template, no tools - the request path only."""

    template_name: ClassVar[str] = ""
    system_template_name: ClassVar[str] = ""

    def render_user_prompt(self, deps: PipelineAgentDeps) -> str:
        return deps.text

    def render_system_prompt(self, deps: PipelineAgentDeps) -> str:
        return ""


def _stub(retries: int = 2) -> tuple[_StubAgent, MagicMock]:
    agent = _StubAgent("test/model", llm=LLMConfig(transient_retries=retries, transient_backoff=0.01))
    fake = MagicMock()
    agent._agent = fake
    return agent, fake


def _deps() -> PipelineAgentDeps:
    return PipelineAgentDeps(text="hello", source_id="test")


def test_transient_failure_is_retried_on_the_agent_base(monkeypatch: pytest.MonkeyPatch) -> None:
    """The retry policy lives in one place - no per-stage backoff loops."""
    monkeypatch.setattr("kompend.agents.base.time.sleep", lambda _seconds: None)
    agent, fake = _stub(retries=2)
    fake.run_sync.side_effect = [httpx.ReadTimeout("slow"), SimpleNamespace(output="ok")]

    assert agent.run(_deps()) == "ok"
    assert fake.run_sync.call_count == 2


def test_transient_budget_is_honoured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("kompend.agents.base.time.sleep", lambda _seconds: None)
    agent, fake = _stub(retries=1)
    fake.run_sync.side_effect = httpx.ReadTimeout("slow")

    with pytest.raises(LLMTransientError):
        agent.run(_deps())
    assert fake.run_sync.call_count == 2


def test_semantic_failure_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("kompend.agents.base.time.sleep", lambda _seconds: None)
    agent, fake = _stub(retries=2)
    fake.run_sync.side_effect = ValueError("bad schema")

    with pytest.raises(LLMOutputError):
        agent.run(_deps())
    assert fake.run_sync.call_count == 1
