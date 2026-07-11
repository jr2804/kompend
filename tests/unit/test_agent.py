"""Unit tests for the ACP agent module."""

from __future__ import annotations

import os

import pytest

from knox.agent.provider_store import LLM_PROVIDER_MAP, ProviderConfig, ProviderCredentials
from knox.agent.slash_commands import HELP_TEXT, parse_slash_command


@pytest.fixture(autouse=True)
def _cleanup_env() -> None:
    """Clean up KNOX env vars set by tests."""
    yield
    for key in ("KNOX_LLM_DEFAULT_MODEL", "KNOX_LLM_API_BASE", "KNOX_LLM_API_KEY"):
        os.environ.pop(key, None)


class TestProviderConfig:
    """ProviderConfig dataclass."""

    def test_default_headers_is_empty(self) -> None:
        cfg = ProviderConfig(api_type="openai", base_url="https://api.openai.com/v1")
        assert cfg.headers == {}

    def test_with_headers(self) -> None:
        cfg = ProviderConfig(
            api_type="anthropic",
            base_url="https://api.anthropic.com",
            headers={"x-api-key": "sk-test"},
        )
        assert cfg.headers["x-api-key"] == "sk-test"


class TestProviderCredentials:
    """ProviderCredentials store."""

    def test_default_is_none(self) -> None:
        store = ProviderCredentials()
        assert store.get() is None

    def test_set_and_get(self) -> None:
        store = ProviderCredentials()
        cfg = ProviderConfig(api_type="openai", base_url="https://api.openai.com/v1")
        store.set(cfg)
        retrieved = store.get()
        assert retrieved is not None
        assert retrieved.api_type == "openai"
        assert retrieved.base_url == "https://api.openai.com/v1"

    def test_get_returns_copy(self) -> None:
        store = ProviderCredentials()
        cfg = ProviderConfig(api_type="openai", base_url="https://api.openai.com/v1")
        store.set(cfg)
        retrieved = store.get()
        assert retrieved is not None
        retrieved.api_type = "anthropic"
        # Original should be unchanged
        assert store.get().api_type == "openai"

    def test_disable_clears(self) -> None:
        store = ProviderCredentials()
        cfg = ProviderConfig(api_type="openai", base_url="https://api.openai.com/v1")
        store.set(cfg)
        store.disable()
        assert store.get() is None

    def test_set_reconfigures_env(self) -> None:
        store = ProviderCredentials()
        cfg = ProviderConfig(
            api_type="openai",
            base_url="https://api.openai.com/v1",
            headers={"Authorization": "Bearer sk-test"},
        )
        store.set(cfg)
        assert os.environ.get("KNOX_LLM_DEFAULT_MODEL") == "openai/default"
        assert os.environ.get("KNOX_LLM_API_BASE") == "https://api.openai.com/v1"
        assert os.environ.get("KNOX_LLM_API_KEY") == "sk-test"

    def test_disable_restores_env(self) -> None:
        store = ProviderCredentials()
        cfg = ProviderConfig(api_type="openai", base_url="https://api.openai.com/v1")
        store.set(cfg)
        assert os.environ.get("KNOX_LLM_DEFAULT_MODEL") == "openai/default"
        store.disable()
        assert "KNOX_LLM_DEFAULT_MODEL" not in os.environ
        assert "KNOX_LLM_API_BASE" not in os.environ

    def test_llm_provider_map(self) -> None:
        assert LLM_PROVIDER_MAP["anthropic"] == "anthropic"
        assert LLM_PROVIDER_MAP["openai"] == "openai"
        assert LLM_PROVIDER_MAP["azure"] == "azure"
        assert LLM_PROVIDER_MAP["vertex"] == "vertex_ai"
        assert LLM_PROVIDER_MAP["bedrock"] == "bedrock"


class TestSlashCommands:
    """Slash command parsing."""

    def test_parse_compile(self) -> None:
        cmd = parse_slash_command("/knox compile")
        assert cmd is not None
        assert cmd.name == "compile"
        assert cmd.args == []
        assert cmd.kwargs == {}

    def test_parse_compile_with_flags(self) -> None:
        cmd = parse_slash_command('/knox compile --start-at-tier chunk --force')
        assert cmd is not None
        assert cmd.name == "compile"
        assert cmd.kwargs.get("start-at-tier") == "chunk"
        assert "force" in cmd.kwargs

    def test_parse_query(self) -> None:
        cmd = parse_slash_command('/knox query "What is audio processing?" --tag audio')
        assert cmd is not None
        assert cmd.name == "query"
        assert cmd.args == ["What is audio processing?"]
        assert cmd.kwargs.get("tag") == "audio"

    def test_parse_capture(self) -> None:
        cmd = parse_slash_command("/knox capture Remember to check the spec")
        assert cmd is not None
        assert cmd.name == "capture"
        assert cmd.args == ["Remember", "to", "check", "the", "spec"]

    def test_parse_status(self) -> None:
        cmd = parse_slash_command("/knox status")
        assert cmd is not None
        assert cmd.name == "status"

    def test_parse_help(self) -> None:
        cmd = parse_slash_command("/knox help")
        assert cmd is not None
        assert cmd.name == "help"

    def test_parse_help_with_command(self) -> None:
        cmd = parse_slash_command("/knox help compile")
        assert cmd is not None
        assert cmd.name == "help"
        assert cmd.args == ["compile"]

    def test_non_knox_command_returns_none(self) -> None:
        cmd = parse_slash_command("hello world")
        assert cmd is None

    def test_empty_after_knox_returns_none(self) -> None:
        cmd = parse_slash_command("/knox ")
        assert cmd is None

    def test_help_text_contains_commands(self) -> None:
        assert "/knox compile" in HELP_TEXT
        assert "/knox query" in HELP_TEXT
        assert "/knox capture" in HELP_TEXT
        assert "/knox status" in HELP_TEXT
        assert "/knox help" in HELP_TEXT
