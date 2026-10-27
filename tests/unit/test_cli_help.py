"""End-to-end CLI help tests — Typer must own the full command tree.

Regression: a previous lazy-registration scheme sniffed ``sys.argv`` to
decide which command modules to import, so ``kompend -w <dir> --help``
rendered help with an EMPTY Commands panel (``--help`` at position 0 was
special-cased, but global flags legitimately precede it). The scheme was
replaced by a Click-native lazy group: Typer handles help in every
position natively while command modules still import on demand.

Help groups subcommands into workflow panels (``rich_help_panel``);
every command in the registry must land in exactly one panel.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest
from typer.testing import CliRunner

from kompend.cli._registry import COMMANDS, PANEL_ORDER, CommandModule, CommandName
from kompend.cli.app import app

runner = CliRunner()


class TestHelpEverywhere:
    def test_bare_help_lists_commands(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "compile" in result.output
        assert "Build pipeline" in result.output

    def test_help_after_global_flag_lists_commands(self) -> None:
        """``kompend -w <dir> --help`` must show the same full help."""
        result = runner.invoke(app, ["-w", "d:/some/wiki", "--help"])
        assert result.exit_code == 0
        assert "compile" in result.output
        assert "Build pipeline" in result.output

    def test_every_command_has_a_workflow_panel(self) -> None:
        """The registry covers every declared command, and every entry declares
        a panel drawn from the panel order — no command without a panel, and no
        panel name outside the workflow order.
        """
        assert set(COMMANDS) == set(CommandName)
        assert {spec.panel for spec in COMMANDS.values()} <= set(PANEL_ORDER)
        assert {spec.module for spec in COMMANDS.values()} == set(CommandModule)

    def test_help_shows_all_workflow_panels(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        for panel in PANEL_ORDER:
            assert panel.value in result.output

    def test_unknown_short_help_rejected_regardless_of_position(self) -> None:
        """``-h`` is not a registered option — same rejection everywhere
        (pre-existing framework behavior, position-independent).
        """
        bare = runner.invoke(app, ["-h"])
        after_flag = runner.invoke(app, ["--wiki-dir", "d:/x", "-h"])
        assert bare.exit_code == after_flag.exit_code != 0
        assert "No such option" in bare.output
        assert "No such option" in after_flag.output

    def test_subcommand_help_unaffected(self) -> None:
        result = runner.invoke(app, ["-w", "d:/some/wiki", "compile", "--help"])
        assert result.exit_code == 0
        assert "compile" in result.output.lower()


class TestCommandHelpSafetyNet:
    """Every registry command must answer ``<cmd> --help`` with exit 0.

    Guards the lazy group, the registry, and every option declaration at
    once: a broken alias (e.g. a missing ``envvar`` member) surfaces here
    the moment a command module or args module drifts.
    """

    @pytest.mark.parametrize("name", list(COMMANDS), ids=lambda n: n.value)
    def test_command_help_exits_zero(self, name: CommandName) -> None:
        result = runner.invoke(app, [name.value, "--help"])
        assert result.exit_code == 0, result.output

    @pytest.mark.parametrize("name", list(COMMANDS), ids=lambda n: n.value)
    def test_registry_short_help_matches_the_command_help(self, name: CommandName) -> None:
        """The registry summary is the text the command itself shows.

        ``--help`` renders from ``CommandSpec.short_help`` without importing
        the command module, so the two must not drift.
        """
        result = runner.invoke(app, [name.value, "--help"])
        assert result.exit_code == 0, result.output
        assert _squash(COMMANDS[name].short_help) in _squash(result.output)


class TestHelpAndVersionStayLight:
    """``--help`` and ``--version`` must not import command modules or config.

    Both are in a fresh interpreter: the point is what a real invocation
    loads, not what an earlier test already imported.
    """

    def test_help_imports_no_command_module(self) -> None:
        exit_code, modules = _probe_cli("--help")
        assert exit_code == 0
        assert [name for name in modules if name.startswith("kompend.cli.cmds")] == []
        assert "logfire" not in modules

    def test_version_does_not_import_the_config_engine(self) -> None:
        exit_code, modules = _probe_cli("--version")
        assert exit_code == 0
        assert "kompend.core.config" not in modules
        assert "logfire" not in modules

    def test_launcher_answers_version_without_the_command_tree(self) -> None:
        probe = (
            "import json, sys\n"
            "sys.argv = ['kompend', '--version']\n"
            "import kompend.cli.launch as launch\n"
            "launch.main()\n"
            "print(json.dumps({'app': 'kompend.cli.app' in sys.modules}))\n"
        )
        completed = subprocess.run(  # noqa: S603 — fixed interpreter, literal probe, no shell
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            check=True,
        )
        lines = completed.stdout.strip().splitlines()
        assert lines[0].startswith(("20", "0.0.0")), lines[0]
        assert json.loads(lines[-1]) == {"app": False}


def _squash(text: str) -> str:
    """Collapse whitespace so wrapped help lines compare equal."""
    return " ".join(text.split())


def _probe_cli(*args: str) -> tuple[int, list[str]]:
    """Run the CLI in a fresh interpreter; return its exit code and module list."""
    probe = (
        "import json, sys\n"
        "from typer.testing import CliRunner\n"
        "import kompend.cli.app as app_module\n"
        "result = CliRunner().invoke(app_module.app, sys.argv[1:])\n"
        "print(json.dumps({'exit': result.exit_code, 'modules': sorted(sys.modules)}))\n"
    )
    completed = subprocess.run(  # noqa: S603 — fixed interpreter, literal probe, no shell
        [sys.executable, "-c", probe, *args],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    return int(payload["exit"]), list(payload["modules"])
