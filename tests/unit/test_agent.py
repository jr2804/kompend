"""Unit tests for the ACP agent module."""

from __future__ import annotations

import json
import os
from collections.abc import Generator
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pytest

from knox.acp.provider_store import (
    LLM_PROVIDER_MAP,
    LlmApiType,
    ProviderConfig,
    ProviderCredentials,
)
from knox.cli._pipeline_helpers import emit_json_done, make_json_progress_callback
from knox.cli.cmds.agents import _agents_install, _agents_remove


@pytest.fixture(autouse=True)
def _cleanup_env() -> Generator[None]:
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
        got = store.get()
        assert got is not None
        assert got.api_type == "openai"

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
        # Well-known apiTypes — four are identity, vertex is overridden.
        assert LLM_PROVIDER_MAP["anthropic"] == "anthropic"
        assert LLM_PROVIDER_MAP["openai"] == "openai"
        assert LLM_PROVIDER_MAP["azure"] == "azure"
        assert LLM_PROVIDER_MAP["vertex"] == "vertex_ai"
        assert LLM_PROVIDER_MAP["bedrock"] == "bedrock"

    def test_llm_provider_map_identity_default(self) -> None:
        # Unknown apiTypes fall through to the identity mapping so custom
        # protocols Just Work without registering an entry.
        assert LLM_PROVIDER_MAP["custom"] == "custom"
        assert LLM_PROVIDER_MAP["my-thing"] == "my-thing"

    def test_llm_provider_map_enumerates_well_known(self) -> None:
        # `providers/list` advertises the well-known apiTypes only —
        # unknown keys should not appear via the identity fallback.
        assert sorted(LLM_PROVIDER_MAP.keys()) == sorted(member.value for member in LlmApiType)

    def test_llm_api_type_enum_values(self) -> None:
        # Enum values must match the wire strings clients send on ACP.
        assert LlmApiType.ANTHROPIC.value == "anthropic"
        assert LlmApiType.OPENAI.value == "openai"
        assert LlmApiType.AZURE.value == "azure"
        assert LlmApiType.VERTEX.value == "vertex"
        assert LlmApiType.BEDROCK.value == "bedrock"


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
    """knox agents install/remove CLI actions."""

    def test_install_valid_agent(self, tmp_path: Path) -> None:
        """Install a valid TOML agent into the vault."""
        agent_toml = tmp_path / "my-agent.toml"
        agent_toml.write_text(
            'name = "my-agent"\ndescription = "Custom"\ntask = "query"\n\n[prompt]\nsystem = "x"\n',
            encoding="utf-8",
        )

        vault_dir = tmp_path / ".knox" / "agents"
        with patch("knox.cli.cmds.agents._resolve_base_dir", return_value=str(tmp_path)):
            _agents_install(str(agent_toml))

        assert (vault_dir / "my-agent.toml").exists()

    def test_install_invalid_agent_raises(self, tmp_path: Path) -> None:
        """Invalid TOML raises during install."""
        bad_toml = tmp_path / "bad.toml"
        bad_toml.write_text("invalid toml {{{{", encoding="utf-8")

        with pytest.raises(RuntimeError), patch("knox.cli.cmds.agents._resolve_base_dir", return_value=str(tmp_path)):
            _agents_install(str(bad_toml))

    def test_remove_existing_agent(self, tmp_path: Path) -> None:
        """Remove an installed agent from the vault."""
        vault_dir = tmp_path / ".knox" / "agents"
        vault_dir.mkdir(parents=True)
        (vault_dir / "test-agent.toml").write_text(
            'name = "test-agent"\ndescription = "x"\ntask = "generic"\n\n[prompt]\nsystem = "x"\n',
            encoding="utf-8",
        )

        with patch("knox.cli.cmds.agents._resolve_base_dir", return_value=str(tmp_path)):
            _agents_remove("test-agent")

        assert not (vault_dir / "test-agent.toml").exists()

    def test_remove_nonexistent_raises(self, tmp_path: Path) -> None:
        """Removing a non-existent agent raises."""
        with pytest.raises(RuntimeError), patch("knox.cli.cmds.agents._resolve_base_dir", return_value=str(tmp_path)):
            _agents_remove("nonexistent")
