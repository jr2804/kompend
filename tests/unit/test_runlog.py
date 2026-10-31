"""Unit tests for the run-log command label.

The label is a pure function over already-parsed command tokens supplied
by the CLI layer — the run log never reads ``sys.argv`` (the previous
argv-walk that hand-stripped ``--wiki-dir``/``-w`` is gone).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kompend.cli.app import app
from kompend.core.config import resolve_config
from kompend.core.config._configs import DisplayConfig
from kompend.core.reporting.runlog import _command_label
from tests.helpers.cli import CliRunner


class TestCommandLabel:
    def test_plain_command(self) -> None:
        assert _command_label(["compile"], DisplayConfig()) == "compile"

    def test_nested_command_keeps_full_path(self) -> None:
        assert _command_label(["attachments", "add", "s", "f"], DisplayConfig()) == "attachments add s f"

    def test_truncates_to_preview_limit(self) -> None:
        assert _command_label(["lint", "a", "b", "c", "d"], DisplayConfig()) == "lint a b c"

    def test_empty_tokens_labels_none(self) -> None:
        assert _command_label([], DisplayConfig()) == "(none)"

    def test_subcommand_args_included_verbatim(self) -> None:
        assert _command_label(["compile", "--foo"], DisplayConfig()) == "compile --foo"


class TestRunLogNeverFails:
    """The run log is written for every command — a config problem must not abort one.

    The run log reads its display limits from the *merged* config, never from
    the validated model: a vault whose config does not validate yet (a typo in
    a key) must still run ``kompend config path``.
    """

    def test_invalid_config_does_not_break_the_command(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / "kompend.toml").write_text("[compile]\nchunk_size = 128\n")

        result = CliRunner().invoke(app, ["config", "path"])

        assert result.exit_code == 0, result.output

    def test_display_limits_come_from_the_config(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / "kompend.toml").write_text("[display]\ncli_arg_preview = 2\n")

        cfg = resolve_config()

        assert cfg.get("display.cli_arg_preview") == 2


class TestCallbackTokens:
    """What the CLI launcher hands the run log.

    The Cyclopts meta app receives the raw tokens *before* dispatch, so the run
    log finally records which command ran. (Under Typer those tokens were always
    empty: ``TyperGroup.invoke`` cleared them before running the group callback,
    which is why this class used to pin the defect as ``xfail``.)
    """

    def test_callback_still_resolves_the_wiki_dir(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The run-level options the launcher owns are read correctly.

        ``--help`` short-circuits before the launcher runs (Cyclopts handles it
        during meta resolution), so this uses a real command.
        """
        captured = self._capture_tokens(monkeypatch)
        result = CliRunner().invoke(app, ["-w", "some/wiki", "list-parsers"])
        assert result.exit_code == 0, result.output
        assert captured["wiki_dir"] == "some/wiki"

    @pytest.mark.parametrize(
        ("argv", "expected"),
        [
            (["attachments", "add", "s", "f"], ["attachments", "add", "s", "f"]),
            (["config", "path"], ["config", "path"]),
            (["-w", "some/wiki", "list-parsers"], ["list-parsers"]),
        ],
    )
    def test_records_the_subcommand_and_its_arguments(
        self,
        monkeypatch: pytest.MonkeyPatch,
        argv: list[str],
        expected: list[str],
    ) -> None:
        captured = self._capture_tokens(monkeypatch)
        CliRunner().invoke(app, argv)
        assert captured["cmd_tokens"] == expected, f"argv={argv!r} produced {captured['cmd_tokens']!r}"

    @staticmethod
    def _capture_tokens(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
        captured: dict[str, object] = {}

        def _capture(wiki_dir: str | None = None, *, cmd_tokens: list[str] | None = None) -> object:
            captured["wiki_dir"] = wiki_dir
            captured["cmd_tokens"] = cmd_tokens
            return None

        monkeypatch.setattr("kompend.cli.app.log_run_start", _capture)
        return captured
