"""The ``providers/list|set|disable`` routes reach kompend and steer model selection.

The editor injects its LLM endpoint over ACP. The SDK routes nothing for
``providers/*``, so kompend registers those routes on the router itself; these
tests pin the handler behaviour, the routing, and that a stored endpoint is what
the next model selection builds.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import acp
import pytest
from pydantic_ai.models import Model, ModelSelectionContext
from pydantic_ai.models.test import TestModel

from kompend.acp import endpoint_model
from kompend.acp.acp_agent import build_acp_agent
from kompend.acp.endpoint_model import EndpointOverrideSelector, build_endpoint_model, model_name_of
from kompend.acp.model_selection import NO_MODEL_MESSAGE, ConfigModelSelector
from kompend.acp.provider_store import LlmApiType, ProviderConfig, ProviderCredentials
from kompend.acp.provider_wire import PROVIDER_ID, ProviderWire, build_provider_router


class TestProviderList:
    async def test_nothing_configured_reports_an_empty_current(self) -> None:
        wire = ProviderWire(ProviderCredentials())

        response = await wire.providers_list()

        assert [provider.id for provider in response.providers] == [PROVIDER_ID]
        provider = response.providers[0]
        assert provider.current is None
        assert provider.required is False
        assert provider.supported == [api_type.value for api_type in LlmApiType]

    async def test_a_stored_endpoint_is_reported(self) -> None:
        store = ProviderCredentials()
        store.set(ProviderConfig(api_type=LlmApiType.ANTHROPIC.value, base_url="https://anthropic.test"))

        response = await ProviderWire(store).providers_list()

        current = response.providers[0].current
        assert current is not None
        assert (current.api_type, current.base_url) == ("anthropic", "https://anthropic.test")


class TestProviderSet:
    async def test_the_endpoint_is_stored(self) -> None:
        store = ProviderCredentials()

        await ProviderWire(store).providers_set(
            id=PROVIDER_ID,
            api_type=LlmApiType.OPENAI.value,
            base_url="https://endpoint.test/v1",
            headers={"Authorization": "Bearer sk-test"},
        )

        stored = store.get()
        assert stored is not None
        assert stored.base_url == "https://endpoint.test/v1"
        assert stored.api_key() == "sk-test"

    async def test_an_unknown_provider_id_is_rejected(self) -> None:
        with pytest.raises(acp.RequestError):
            await ProviderWire(ProviderCredentials()).providers_set(
                id="other",
                api_type=LlmApiType.OPENAI.value,
                base_url="https://endpoint.test/v1",
            )

    async def test_a_missing_base_url_is_rejected(self) -> None:
        with pytest.raises(acp.RequestError):
            await ProviderWire(ProviderCredentials()).providers_set(
                id=PROVIDER_ID,
                api_type=LlmApiType.OPENAI.value,
                base_url="",
            )


class TestProviderDisable:
    async def test_disable_drops_the_endpoint(self) -> None:
        store = ProviderCredentials()
        store.set(ProviderConfig(api_type=LlmApiType.OPENAI.value, base_url="https://endpoint.test"))

        await ProviderWire(store).providers_disable(id=PROVIDER_ID)

        assert store.get() is None

    async def test_disable_rejects_an_unknown_id(self) -> None:
        with pytest.raises(acp.RequestError):
            await ProviderWire(ProviderCredentials()).providers_disable(id="other")


class TestRoutes:
    async def test_the_router_serves_the_three_methods(self) -> None:
        store = ProviderCredentials()
        router = build_provider_router(cast(Any, SimpleNamespace()), ProviderWire(store))

        await router("providers/set", {"id": PROVIDER_ID, "apiType": "openai", "baseUrl": "https://x.test"}, False)
        listed = await router("providers/list", {}, False)

        assert listed["providers"][0]["current"] == {"apiType": "openai", "baseUrl": "https://x.test"}
        await router("providers/disable", {"id": PROVIDER_ID}, False)
        assert store.get() is None

    async def test_an_unrouted_provider_method_still_404s(self, monkeypatch: pytest.MonkeyPatch) -> None:
        router = build_provider_router(cast(Any, SimpleNamespace()), ProviderWire(ProviderCredentials()))

        with pytest.raises(acp.RequestError):
            await router("providers/unknown", {}, False)


class TestEndpointModel:
    def test_an_openai_endpoint_becomes_an_openai_model(self) -> None:
        model = build_endpoint_model(
            ProviderConfig(api_type=LlmApiType.OPENAI.value, base_url="https://endpoint.test/v1"),
            "gpt-4o",
        )

        assert model.provider is not None
        assert model.provider.base_url.rstrip("/") == "https://endpoint.test/v1"

    def test_an_anthropic_endpoint_becomes_an_anthropic_model(self) -> None:
        model = build_endpoint_model(
            ProviderConfig(api_type=LlmApiType.ANTHROPIC.value, base_url="https://anthropic.test"),
            "claude-x",
        )

        assert model.provider is not None
        assert model.provider.base_url.rstrip("/") == "https://anthropic.test"

    def test_a_blank_header_still_builds(self) -> None:
        model = build_endpoint_model(
            ProviderConfig(api_type=LlmApiType.OPENAI.value, base_url="https://endpoint.test", headers={}),
            "gpt-4o",
        )

        assert model.provider is not None

    def test_an_unsupported_api_type_is_refused(self) -> None:
        with pytest.raises(ValueError, match="not supported"):
            build_endpoint_model(
                ProviderConfig(api_type=LlmApiType.BEDROCK.value, base_url="https://bedrock.test"),
                "some-model",
            )

    def test_the_model_name_drops_the_provider_prefix(self) -> None:
        assert model_name_of("openai/gpt-4o") == "gpt-4o"
        assert model_name_of("bare-model") == "bare-model"


class TestEndpointOverrideSelector:
    def test_without_an_endpoint_the_config_model_serves(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sentinel = TestModel()
        fallback = _FixedSelector(sentinel)
        selector = EndpointOverrideSelector(
            base_dir=Path(),
            credentials=ProviderCredentials(),
            fallback=cast(Any, fallback),
        )

        assert selector(_context()) is sentinel

    def test_a_stored_endpoint_replaces_the_config_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_config(monkeypatch, default_model="openai/gpt-4o")
        store = ProviderCredentials()
        store.set(ProviderConfig(api_type=LlmApiType.OPENAI.value, base_url="https://injected.test/v1"))
        selector = EndpointOverrideSelector(
            base_dir=Path(),
            credentials=store,
            fallback=ConfigModelSelector(Path()),
        )

        model = selector(_context())

        assert model.provider is not None
        assert model.provider.base_url.rstrip("/") == "https://injected.test/v1"

    def test_a_second_injection_is_honoured_on_the_next_step(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_config(monkeypatch, default_model="openai/gpt-4o")
        store = ProviderCredentials()
        store.set(ProviderConfig(api_type=LlmApiType.OPENAI.value, base_url="https://first.test/v1"))
        selector = EndpointOverrideSelector(Path(), store, ConfigModelSelector(Path()))

        first = selector(_context())
        store.set(ProviderConfig(api_type=LlmApiType.OPENAI.value, base_url="https://second.test/v1"))
        second = selector(_context())

        assert first is not second
        assert second.provider is not None
        assert second.provider.base_url.rstrip("/") == "https://second.test/v1"

    def test_an_endpoint_without_a_configured_model_still_fails_with_the_sentence(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The endpoint says where to send requests, not which model answers."""
        _patch_config(monkeypatch, default_model=None)
        store = ProviderCredentials()
        store.set(ProviderConfig(api_type=LlmApiType.OPENAI.value, base_url="https://injected.test/v1"))
        selector = EndpointOverrideSelector(Path(), store, ConfigModelSelector(Path()))

        with pytest.raises(acp.RequestError) as caught:
            selector(_context())

        assert str(caught.value) == NO_MODEL_MESSAGE


class TestAdvertisedCapability:
    async def test_initialize_advertises_provider_configuration(self) -> None:

        agent = build_acp_agent(cast(Any, SimpleNamespace(name="test", output_type=str)))

        response = await agent.initialize(1)

        assert response.agent_capabilities is not None
        assert response.agent_capabilities.providers is not None


class _FixedSelector:
    def __init__(self, model: Model) -> None:
        self._model = model

    def __call__(self, ctx: ModelSelectionContext[None]) -> Model:
        del ctx
        return self._model


def _context() -> ModelSelectionContext[None]:
    return cast(ModelSelectionContext[None], SimpleNamespace())


def _patch_config(monkeypatch: pytest.MonkeyPatch, *, default_model: str | None) -> None:
    llm = SimpleNamespace(default_model=default_model)
    resolved = SimpleNamespace(kompend=SimpleNamespace(compile=SimpleNamespace(llm=llm)))
    monkeypatch.setattr(endpoint_model, "resolve_config", lambda **kwargs: resolved)
