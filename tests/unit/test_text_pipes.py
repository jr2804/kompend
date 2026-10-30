"""Tests for :mod:`kompend.core.text.pipes` (generation contract).

Regression anchor: unescaped prose pipes in list-item paragraph
continuations are the input class that sent docling's markdown backend into
an infinite CPU spin — kompend-generated markdown must never emit one.
"""

from __future__ import annotations

import pytest

from kompend.core.text.pipes import PipeEscapeError, assert_pipes_escaped, escape_prose_pipes


def test_escapes_bare_prose_pipe() -> None:
    assert escape_prose_pipes("the |delay| term") == "the \\|delay\\| term"


def test_real_world_fixture_lines() -> None:
    # Shape distilled from distilled-20261018T225611-pairprocess.md.
    text = "- Timing check:\n  the delay difference |delay[i] - delay[i-2]| stays bounded\n  and |start| is well-defined.\n"
    out = escape_prose_pipes(text)
    assert "\\|delay[i] - delay[i-2]\\|" in out
    assert "\\|start\\|" in out
    assert_pipes_escaped(out)


def test_idempotent() -> None:
    once = escape_prose_pipes("a | b | c")
    assert escape_prose_pipes(once) == once


def test_no_pipes_unchanged() -> None:
    text = "plain paragraph\n\n- item\n"
    assert escape_prose_pipes(text) == text


def test_already_escaped_pipe_untouched() -> None:
    assert escape_prose_pipes("a \\| b") == "a \\| b"


def test_even_backslash_run_is_bare_pipe() -> None:
    # Two backslashes form an escaped backslash; the pipe is bare.
    assert escape_prose_pipes("a \\\\| b") == "a \\\\\\| b"


def test_genuine_table_untouched() -> None:
    text = "| a | b |\n| --- | --- |\n| 1 | 2 |\n\nprose | pipe\n"
    out = escape_prose_pipes(text)
    lines = out.split("\n")
    assert lines[0] == "| a | b |"
    assert lines[1] == "| --- | --- |"
    assert lines[2] == "| 1 | 2 |"
    assert lines[4] == "prose \\| pipe"


def test_delimiter_without_matching_header_is_prose() -> None:
    # Cell counts differ -> not a table -> both lines escaped.
    out = escape_prose_pipes("| a | b | c |\n| --- | --- |")
    assert out == "\\| a \\| b \\| c \\|\n\\| --- \\| --- \\|"


def test_fenced_code_block_untouched() -> None:
    text = "```python\nx = a | b\n```\n"
    assert escape_prose_pipes(text) == text


def test_tilde_fence_untouched() -> None:
    text = "~~~\na | b\n~~~\nprose | pipe\n"
    out = escape_prose_pipes(text)
    assert "a | b" in out
    assert out.endswith("prose \\| pipe\n")


def test_inline_code_span_untouched() -> None:
    assert escape_prose_pipes("run `cut -d| -f2` now") == "run `cut -d| -f2` now"


def test_math_span_untouched() -> None:
    assert escape_prose_pipes("$|x|$ and $$|a - b|$$") == "$|x|$ and $$|a - b|$$"


def test_wikilink_alias_untouched() -> None:
    assert escape_prose_pipes("see [[Node Title|alias]] here") == "see [[Node Title|alias]] here"


def test_assert_passes_on_escaped_text() -> None:
    assert_pipes_escaped("clean\n\nescaped \\| pipe\n")


def test_assert_raises_on_unescaped_pipe() -> None:
    with pytest.raises(PipeEscapeError, match="line 3"):
        assert_pipes_escaped("clean\n\nclean | dirty\n")


def test_assert_ignores_protected_regions() -> None:
    assert_pipes_escaped("| h |\n| --- |\n| `a|b` |\n")
