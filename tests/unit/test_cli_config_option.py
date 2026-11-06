"""The root ``--config`` option is the only config-file flag."""

from __future__ import annotations

from typing import Any

import pytest

from kompend.cli.app import app, register_all_commands
from kompend.cli.cmds.quality.lint import history as lint_history_cmd
from kompend.cli.context import RunContext
from kompend.cli.settings import config_file_from_ctx
from tests.helpers.cli import CliRunner


class TestConfigOptionIsRootOnly:
    def test_root_help_lists_config(self) -> None:
        register_all_commands()

        output = CliRunner().invoke(app, ["--help"]).output

        assert "--config" in output
        assert "-c" in output

    def test_lint_all_does_not_declare_its_own_config(self) -> None:
        assert "--config" not in _option_names("lint", "all")

    def test_sources_list_does_not_declare_its_own_config(self) -> None:
        assert "--config" not in _option_names("sources", "list")

    def test_init_names_its_seed_file_differently(self) -> None:
        """``init`` installs a config - it must not shadow the run-level flag."""
        assert "--config" not in _option_names("init")
        assert "--seed-config" in _option_names("init")


class TestConfigFileFromCtx:
    def test_reads_the_root_value(self) -> None:
        assert config_file_from_ctx(RunContext(config_file="alt.toml")) == "alt.toml"

    def test_missing_value_is_none(self) -> None:
        assert config_file_from_ctx(RunContext()) is None
        assert config_file_from_ctx(None) is None

    def test_a_nested_command_sees_the_root_file(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The value travels through the ambient run context, not a command option."""
        seen: dict[str, Any] = {}

        def _capture(**_kwargs: Any) -> Any:
            seen["config_file"] = config_file_from_ctx()
            raise SystemExit(0)

        monkeypatch.setattr(lint_history_cmd, "vault_paths_from_ctx", _capture)
        register_all_commands()

        result = CliRunner().invoke(app, ["--config", "alt.toml", "lint", "history"])

        assert result.exit_code == 0
        assert seen["config_file"] == "alt.toml"


def _option_names(command: str, *sub: str) -> set[str]:
    """Option names the command itself declares (not the root meta flags)."""
    register_all_commands()
    target = app[command]
    for name in sub:
        target = target[name]
    return {name for argument in target.assemble_argument_collection() for name in argument.names if name.startswith("-")}
