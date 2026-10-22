"""Unit tests for the run-log command label.

The label is a pure function over already-parsed command tokens supplied
by the CLI layer — the run log never reads ``sys.argv`` (the previous
argv-walk that hand-stripped ``--wiki-dir``/``-w`` is gone).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from kompend.cli.app import app
from kompend.core.config import resolve_config
from kompend.core.config._configs import DisplayConfig
from kompend.core.reporting.runlog import _command_label


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
    """What the CLI callback hands the run log.

    KNOWN DEFECT, pinned here rather than hidden: ``cmd_tokens`` is always empty,
    so the run log's command label is always ``(none)`` and the audit trail never
    records which command ran. ``TyperGroup.parse_args`` populates
    ``ctx._protected_args``/``ctx.args``, then ``TyperGroup.invoke`` reads both and
    clears them *before* it runs the group callback — so by the time ``_callback``
    reads them they are empty. These are ``xfail(strict=True)``: they document the
    defect and will fail loudly once it is fixed, prompting their removal.
    """

    def test_callback_still_resolves_the_wiki_dir(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The group options the callback *does* own are read correctly."""
        captured = self._capture_tokens(monkeypatch)
        result = CliRunner().invoke(app, ["-w", "some/wiki", "compile", "--help"])
        assert result.exit_code == 0
        assert captured["wiki_dir"] == "some/wiki"

    @pytest.mark.xfail(strict=True, reason="run-log command label is always empty (see class docstring)")
    @pytest.mark.parametrize(
        ("argv", "expected"),
        [
            (["compile", "--help"], ["compile", "--help"]),
            (["--help"], []),
            (["attachments", "add", "s", "f"], ["attachments", "add", "s", "f"]),
            (["lint", "--help"], ["lint", "--help"]),
            (["-w", "some/wiki", "compile", "--help"], ["compile", "--help"]),
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
