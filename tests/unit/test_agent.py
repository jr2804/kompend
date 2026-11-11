"""Unit tests for the ACP agent module."""

from __future__ import annotations

import json
import os
from collections.abc import Generator
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pytest

from kompend.acp.provider_store import (
    LlmApiType,
    ProviderConfig,
    ProviderCredentials,
)
from kompend.cli.cmds.agents import _agents_install, _agents_remove
from kompend.cli.workflow.pipeline import emit_json_done, make_json_progress_callback
from kompend.core.env import EnvVar


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
        got = store.get()
        assert got is not None
        assert got.api_type == "openai"

    def test_disable_clears(self) -> None:
        store = ProviderCredentials()
        cfg = ProviderConfig(api_type="openai", base_url="https://api.openai.com/v1")
        store.set(cfg)
        store.disable()
        assert store.get() is None

    def test_set_and_get_stores_headers(self) -> None:
        store = ProviderCredentials()
        cfg = ProviderConfig(
            api_type="openai",
            base_url="https://api.openai.com/v1",
            headers={"Authorization": "Bearer sk-test"},
        )
        store.set(cfg)
        retrieved = store.get()
        assert retrieved is not None
        assert retrieved.api_key() == "sk-test"

    def test_set_does_not_touch_env(self) -> None:
        # The store is the only channel: nothing reads KOMP_LLM_API_BASE, so an
        # endpoint written to os.environ would be silently lost.
        store = ProviderCredentials()
        store.set(ProviderConfig(api_type="openai", base_url="https://api.openai.com/v1"))
        assert EnvVar.LLM_API_BASE not in os.environ
        assert EnvVar.LLM_API_KEY not in os.environ
        assert EnvVar.LLM_DEFAULT_MODEL not in os.environ

    def test_api_key_is_none_without_authorization(self) -> None:
        cfg = ProviderConfig(api_type="openai", base_url="https://example.test", headers={"x-other": "1"})
        assert cfg.api_key() is None

    def test_api_key_accepts_a_bare_token(self) -> None:
        cfg = ProviderConfig(
            api_type="openai",
            base_url="https://example.test",
            headers={"authorization": "sk-plain"},
        )
        assert cfg.api_key() == "sk-plain"

    def test_disable_clears_without_touching_env(self) -> None:
        store = ProviderCredentials()
        store.set(ProviderConfig(api_type="openai", base_url="https://api.openai.com/v1"))
        store.disable()
        assert store.get() is None
        assert EnvVar.LLM_API_BASE not in os.environ

    def test_llm_api_type_enum_values(self) -> None:
        # Enum values must match the wire strings clients send on ACP — they are
        # what ``providers/list`` advertises as ``supported``.
        assert LlmApiType.ANTHROPIC.value == "anthropic"
        assert LlmApiType.OPENAI.value == "openai"
        assert LlmApiType.AZURE.value == "azure"
        assert LlmApiType.VERTEX.value == "vertex"
        assert LlmApiType.BEDROCK.value == "bedrock"

    def test_llm_api_type_vertex_alias_resolves(self) -> None:
        # Clients on older versions send the underscore spelling.
        assert LlmApiType("vertex_ai") is LlmApiType.VERTEX


class TestJsonProgressCallback:
    """JSON progress callback for --json-progress mode."""

    def test_progress_writes_jsonl(self) -> None:
        """Each progress event is one JSON line on stdout."""
        cb = make_json_progress_callback()
        fake_stdout = StringIO()
        with patch("sys.stdout", fake_stdout):
            cb("parse", 5, 40, "[5/40] file.pdf")
        lines = [line for line in fake_stdout.getvalue().strip().split("\n") if line]
        assert len(lines) == 1
        event = json.loads(lines[0])
        assert event["event"] == "progress"
        assert event["phase"] == "parse"
        assert event["current"] == 5
        assert event["total"] == 40
        assert event["message"] == "[5/40] file.pdf"

    def test_done_writes_final_line(self) -> None:
        """_emit_json_done writes a single done event."""
        fake_stdout = StringIO()
        with patch("sys.stdout", fake_stdout):
            emit_json_done(123.4, "Compiled wiki written to /tmp/wiki")
        event = json.loads(fake_stdout.getvalue().strip())
        assert event["event"] == "done"
        assert event["duration"] == 123.4
        assert "Compiled wiki" in event["message"]


class TestAgentInstallRemove:
    """kompend agents install/remove CLI actions."""

    def test_install_valid_agent(self, tmp_path: Path) -> None:
        """Install a valid TOML agent into the vault."""
        agent_toml = tmp_path / "my-agent.toml"
        agent_toml.write_text(
            'name = "my-agent"\ndescription = "Custom"\ntask = "query"\n\n[prompt]\nsystem = "x"\n',
            encoding="utf-8",
        )

        vault_dir = tmp_path / ".kompend" / "agents"
        with patch("kompend.cli.cmds.agents._resolve_base_dir", return_value=str(tmp_path)):
            _agents_install(str(agent_toml))

        assert (vault_dir / "my-agent.toml").exists()

    def test_install_invalid_agent_raises(self, tmp_path: Path) -> None:
        """Invalid TOML raises during install."""
        bad_toml = tmp_path / "bad.toml"
        bad_toml.write_text("invalid toml {{{{", encoding="utf-8")

        with pytest.raises(SystemExit) as excinfo, patch("kompend.cli.cmds.agents._resolve_base_dir", return_value=str(tmp_path)):
            _agents_install(str(bad_toml))
        assert excinfo.value.code == 1

    def test_remove_existing_agent(self, tmp_path: Path) -> None:
        """Remove an installed agent from the vault."""
        vault_dir = tmp_path / ".kompend" / "agents"
        vault_dir.mkdir(parents=True)
        (vault_dir / "test-agent.toml").write_text(
            'name = "test-agent"\ndescription = "x"\ntask = "generic"\n\n[prompt]\nsystem = "x"\n',
            encoding="utf-8",
        )

        with patch("kompend.cli.cmds.agents._resolve_base_dir", return_value=str(tmp_path)):
            _agents_remove("test-agent")

        assert not (vault_dir / "test-agent.toml").exists()

    def test_remove_nonexistent_raises(self, tmp_path: Path) -> None:
        """Removing a non-existent agent raises."""
        with pytest.raises(SystemExit) as excinfo, patch("kompend.cli.cmds.agents._resolve_base_dir", return_value=str(tmp_path)):
            _agents_remove("nonexistent")
        assert excinfo.value.code == 1


@pytest.fixture(autouse=True)
def _cleanup_env() -> Generator[None]:
    """Clean up kompend env vars set by tests."""
    yield
    for key in (EnvVar.LLM_DEFAULT_MODEL, EnvVar.LLM_API_BASE, EnvVar.LLM_API_KEY):
        os.environ.pop(key, None)
