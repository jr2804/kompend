"""Unit tests for the run-log command label.

The label is a pure function over already-parsed command tokens supplied
by the CLI layer — the run log never reads ``sys.argv`` (the previous
argv-walk that hand-stripped ``--wiki-dir``/``-w`` is gone).
"""

from __future__ import annotations

from kompend.core.reporting.runlog import _command_label


class TestCommandLabel:
    def test_plain_command(self) -> None:
        assert _command_label(["compile"]) == "compile"

    def test_nested_command_keeps_full_path(self) -> None:
        assert _command_label(["attachments", "add", "s", "f"]) == "attachments add s f"

    def test_truncates_to_preview_limit(self) -> None:
        assert _command_label(["lint", "a", "b", "c", "d"]) == "lint a b c"

    def test_empty_tokens_labels_none(self) -> None:
        assert _command_label([]) == "(none)"

    def test_subcommand_args_included_verbatim(self) -> None:
        assert _command_label(["compile", "--foo"]) == "compile --foo"
