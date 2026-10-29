"""The root ``--config`` option is the only config-file flag."""

from __future__ import annotations

from typing import Any

import pytest
import typer
from typer.main import get_command
from typer.testing import CliRunner

from kompend.cli._shared import config_file_from_ctx
from kompend.cli.app import app, register_all_commands
from kompend.cli.cmds.quality import lint as lint_cmd


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
        ctx = typer.Context(get_command(app), obj={"config_file": "alt.toml"})

        assert config_file_from_ctx(ctx) == "alt.toml"

    def test_missing_value_is_none(self) -> None:
        assert config_file_from_ctx(typer.Context(get_command(app), obj={})) is None
        assert config_file_from_ctx(None) is None

    def test_a_nested_command_sees_the_root_file(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The value travels through the Typer context, not a command option."""
        seen: dict[str, Any] = {}

        def _capture(ctx: Any, **_kwargs: Any) -> Any:
            seen["config_file"] = config_file_from_ctx(ctx)
            raise SystemExit(0)

        monkeypatch.setattr(lint_cmd, "vault_paths_from_ctx", _capture)
        register_all_commands()

        result = CliRunner().invoke(app, ["--config", "alt.toml", "lint", "history"])

        assert result.exit_code == 0
        assert seen["config_file"] == "alt.toml"


def _option_names(command: str, *sub: str) -> set[str]:
    """Option names a command's own ``--help`` advertises."""
    register_all_commands()
    result = CliRunner().invoke(app, [command, *sub, "--help"])
    names: set[str] = set()
    for line in result.output.splitlines():
        for token in line.replace(",", " ").split():
            if token.startswith("-") and not token.startswith("---"):
                names.add(token.rstrip(":"))
    return names
