"""The ACP agent resolves its model per request step instead of a placeholder.

The server has to start with nothing configured (an editor may configure an
endpoint before it prompts), so a missing model must surface as the house
sentence on the prompt, not as a construction-time ``Unknown model`` crash.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import acp
import pytest
from pydantic_ai import Agent
from pydantic_ai.capabilities import Caching, SelectModel
from pydantic_ai.models import Model, ModelSelectionContext
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.experimental.acp import AcpSession

from kompend.acp import harness_adapter, model_selection
from kompend.acp.model_selection import NO_MODEL_MESSAGE, ConfigModelSelector
from kompend.llm.user_providers import set_user_providers


class TestBuildKompendAgent:
    def test_agent_carries_no_model_of_its_own(self) -> None:
        agent = harness_adapter.build_kompend_agent()

        assert agent.model is None
        assert isinstance(agent, Agent)

    def test_agent_capabilities_select_the_model(self) -> None:
        agent = cast(Any, harness_adapter.build_kompend_agent())

        assert agent.root_capability._can_supply_model
        assert isinstance(agent.root_capability.get_model(), ConfigModelSelector)


class TestConfigModelSelector:
    def test_missing_model_raises_the_house_sentence(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        _patch_config(monkeypatch, default_model=None)

        with pytest.raises(acp.RequestError) as caught:
            ConfigModelSelector(tmp_path)(_context())

        assert caught.value.code == model_selection._INTERNAL_ERROR
        assert str(caught.value) == NO_MODEL_MESSAGE

    def test_the_failure_is_not_pydantic_ais_user_error(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        _patch_config(monkeypatch, default_model=None)

        with pytest.raises(acp.RequestError):
            ConfigModelSelector(tmp_path)(_context())

    def test_configured_model_is_built_and_cached(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        _patch_config(monkeypatch, default_model="test/kompend")
        built: list[str] = []
        sentinel = TestModel()

        def _build(model_id: str) -> Model:
            built.append(model_id)
            return sentinel

        monkeypatch.setattr(model_selection, "build_model", _build)
        selector = ConfigModelSelector(tmp_path)

        first = selector(_context())
        second = selector(_context())

        assert first is sentinel
        assert second is sentinel
        assert built == ["test/kompend"], "the model is built once per session, not once per step"

    def test_invalid_model_id_names_the_config_value(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        _patch_config(monkeypatch, default_model="nope/model")

        def _build(model_id: str) -> Model:
            raise ValueError(f"Unknown provider {model_id!r}.")

        monkeypatch.setattr(model_selection, "build_model", _build)

        with pytest.raises(acp.RequestError) as caught:
            ConfigModelSelector(tmp_path)(_context())

        assert "Unknown provider" in str(caught.value)

    def test_a_failed_attempt_is_retried_on_the_next_step(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Configuring a model after a failed prompt must work without a restart."""
        default_model: str | None = None
        _patch_config(monkeypatch, default_model=lambda: default_model)
        sentinel = TestModel()
        monkeypatch.setattr(model_selection, "build_model", lambda model_id: sentinel)
        selector = ConfigModelSelector(tmp_path)

        with pytest.raises(acp.RequestError):
            selector(_context())

        default_model = "test/kompend"

        assert selector(_context()) is sentinel


class TestKompendSessionConfig:
    def test_selector_is_bound_to_the_session_workspace(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        _patch_session(monkeypatch, prompt_cache=True)

        config = harness_adapter.kompend_session_config(_session(tmp_path))

        assert config.capabilities is not None
        selectors = [c for c in config.capabilities if isinstance(c, SelectModel)]
        assert len(selectors) == 1
        selector = selectors[0].selector
        assert isinstance(selector, ConfigModelSelector)
        assert selector.base_dir == tmp_path

    def test_caching_stays_first_when_enabled(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        _patch_session(monkeypatch, prompt_cache=True)

        capabilities = harness_adapter.kompend_session_config(_session(tmp_path)).capabilities

        assert capabilities is not None
        assert isinstance(capabilities[0], Caching)

    def test_model_selection_survives_caching_being_off(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        _patch_session(monkeypatch, prompt_cache=False)

        capabilities = harness_adapter.kompend_session_config(_session(tmp_path)).capabilities

        assert capabilities is not None
        assert all(isinstance(c, SelectModel) for c in capabilities)


class TestConfiguredEndpoint:
    def test_the_model_points_at_the_configured_provider(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """The built model talks to the endpoint named by ``[providers.*]``."""
        set_user_providers({"kompendtest": {"base_url": "https://endpoint.test/v1", "api_key": "sk-test"}})
        try:
            _patch_config(monkeypatch, default_model="kompendtest/some-model")

            model = ConfigModelSelector(tmp_path)(_context())
            provider = model.provider

            assert provider is not None
            assert provider.base_url.rstrip("/") == "https://endpoint.test/v1"
        finally:
            set_user_providers({})


class TestPromptWithoutAModel:
    async def test_the_house_sentence_reaches_the_caller(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """A prompt with nothing configured fails with the sentence, not UserError."""
        _patch_config(monkeypatch, default_model=None)
        agent = Agent(
            None,
            instructions="test",
            name="no-model",
            capabilities=[SelectModel(ConfigModelSelector(tmp_path))],
        )

        with pytest.raises(acp.RequestError) as caught:
            await agent.run("hello")

        assert str(caught.value) == NO_MODEL_MESSAGE


def _context() -> ModelSelectionContext[None]:
    return cast(ModelSelectionContext[None], SimpleNamespace())


def _session(tmp_path: Path) -> AcpSession:
    return cast(AcpSession, SimpleNamespace(cwd=tmp_path))


def _patch_config(monkeypatch: pytest.MonkeyPatch, *, default_model: Any) -> None:
    """Point the selector's config lookup at a stub with *default_model*.

    A callable value is re-evaluated on every lookup, so a test can change the
    configured model between steps.
    """

    def _resolved(**kwargs: object) -> Any:
        value = default_model() if callable(default_model) else default_model
        llm = SimpleNamespace(default_model=value)
        return SimpleNamespace(kompend=SimpleNamespace(compile=SimpleNamespace(llm=llm)))

    monkeypatch.setattr(model_selection, "resolve_config", _resolved)


def _patch_session(monkeypatch: pytest.MonkeyPatch, *, prompt_cache: bool) -> None:
    definition = SimpleNamespace(tools=SimpleNamespace(enabled=[]))
    monkeypatch.setattr(
        harness_adapter,
        "AgentRegistry",
        lambda base_dir: SimpleNamespace(active_for=lambda task, overrides: definition),
    )
    monkeypatch.setattr(harness_adapter, "build_toolset", lambda **kwargs: SimpleNamespace())
    llm = SimpleNamespace(prompt_cache=prompt_cache)
    resolved = SimpleNamespace(kompend=SimpleNamespace(compile=SimpleNamespace(llm=llm)))
    monkeypatch.setattr(harness_adapter, "resolve_config", lambda **kwargs: resolved)
