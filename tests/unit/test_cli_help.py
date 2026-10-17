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
