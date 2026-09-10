"""Tests for lazy command registration (``_register_for_argv``).

Regression: ``knox -w <dir> --help`` rendered help with an EMPTY Commands
panel because the help-detection only checked ``argv[0]``. Global flags
legitimately precede ``--help`` (``knox -w <dir> --help``), so the check
must scan the whole argv.
"""

from __future__ import annotations

from unittest.mock import patch

from knox.cli.app import _register_for_argv


class TestRegisterForArgv:
    def test_bare_help_loads_all(self) -> None:
        with patch("knox.cli.app.register_all_commands") as m:
            _register_for_argv(["--help"])
            m.assert_called_once()

    def test_help_after_global_flag_loads_all(self) -> None:
        """``knox -w <dir> --help`` must load all commands (p863 roundup)."""
        with patch("knox.cli.app.register_all_commands") as m:
            _register_for_argv(["-w", "d:/some/wiki", "--help"])
            m.assert_called_once()

    def test_short_help_after_global_flag_loads_all(self) -> None:
        with patch("knox.cli.app.register_all_commands") as m:
            _register_for_argv(["--wiki-dir", "x", "-h"])
            m.assert_called_once()

    def test_empty_argv_loads_all(self) -> None:
        with patch("knox.cli.app.register_all_commands") as m:
            _register_for_argv([])
            m.assert_called_once()

    def test_known_subcommand_loads_only_it(self) -> None:
        with patch("knox.cli.app.register_all_commands") as m, patch("knox.cli.app.register_command") as r:
            _register_for_argv(["compile"])
            m.assert_not_called()
            r.assert_called_once_with("compile")

    def test_global_flag_only_registers_nothing(self) -> None:
        with patch("knox.cli.app.register_all_commands") as m, patch("knox.cli.app.register_command") as r:
            _register_for_argv(["--version"])
            m.assert_not_called()
            r.assert_not_called()
