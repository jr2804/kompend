"""Tests for content windowing — the shared splitter, the write-source tool, the write paths.

A budget may size one call; it may never shorten the content that is
processed. These tests pin that: every window path reassembles the
original text, and no path crops a source.
"""

from __future__ import annotations

from typing import cast

import pytest
from pydantic_ai import RunContext

from kompend.agents.types import WriteDeps
from kompend.cli.cmds.knowledge import capture as capture_cmd
from kompend.compilation.write_tools import read_source
from kompend.core.text.windows import split_text_windows
from kompend.mcp.tools import pipeline as mcp_pipeline


class _Ctx:
    def __init__(self, deps: WriteDeps) -> None:
        self.deps = deps


class TestSplitTextWindows:
    def test_short_text_is_one_window(self) -> None:
        assert split_text_windows("aa\nbb\ncc", 100) == ["aa\nbb\ncc"]

    def test_long_text_is_split_without_loss(self) -> None:
        text = "\n".join("x" * 10 for _ in range(20))
        windows = split_text_windows(text, 25)
        assert len(windows) > 1
        assert all(len(w) <= 25 for w in windows)
        assert "".join(windows) == text

    def test_oversized_line_is_hard_split(self) -> None:
        windows = split_text_windows("y" * 100, 30)
        assert [len(w) for w in windows] == [30, 30, 30, 10]
        assert "".join(windows) == "y" * 100

    def test_no_limit_returns_the_whole_text(self) -> None:
        text = "z" * 500
        assert split_text_windows(text, 0) == [text]

    def test_empty_text_yields_no_window(self) -> None:
        assert split_text_windows("", 100) == []


class TestReadSourceWindows:
    async def test_source_shorter_than_the_window_is_returned_whole(self) -> None:
        out = await read_source(_ctx(_deps("short text", char_max=100)), "doc")
        assert out == "short text"

    async def test_long_source_is_served_in_windows(self) -> None:
        text = "a" * 250
        out = await read_source(_ctx(_deps(text, char_max=100)), "doc")
        assert "a" * 100 in out
        assert "offset=100" in out

    async def test_paging_recovers_the_whole_source(self) -> None:
        text = "\n".join(f"line {i}" for i in range(40))
        limit = 50
        collected = ""
        offset = 0
        for _ in range(20):
            reply = await read_source(_ctx(_deps(text, char_max=limit)), "doc", offset=offset)
            body = reply.split("\n\n[end of window", 1)[0]
            collected += body
            if "offset=" not in reply:
                break
            offset = int(reply.rsplit("offset=", 1)[1].split(")", 1)[0])
        assert collected == text

    async def test_no_window_limit_returns_the_whole_source(self) -> None:
        text = "b" * 500
        out = await read_source(_ctx(_deps(text, char_max=0)), "doc")
        assert out == text

    async def test_offset_past_the_end_returns_nothing(self) -> None:
        out = await read_source(_ctx(_deps("short", char_max=10)), "doc", offset=99)
        assert out == ""

    async def test_unknown_source_is_reported(self) -> None:
        out = await read_source(_ctx(_deps("text", char_max=10)), "other")
        assert "Unknown source" in out


class TestWritePageSourcesAreNotCropped:
    def test_cli_write_page_keeps_the_full_source(self, monkeypatch: pytest.MonkeyPatch) -> None:
        text = "t" * 20_000
        seen: dict[str, object] = {}

        class _Agent:
            def run(self, deps: WriteDeps) -> str:
                seen["deps"] = deps
                return "written"

        monkeypatch.setattr(capture_cmd, "make_agentic_write_agent", lambda **kwargs: _Agent())

        body = capture_cmd._synthesize_write_page(
            page_title="T",
            source_texts=[("doc", text)],
            related_titles=[],
            model="some/model",
            timeout=10.0,
            source_char_max=4000,
        )

        assert body.startswith("written")
        deps = cast(WriteDeps, seen["deps"])
        assert deps.source_texts == [("doc", text)]
        assert deps.source_char_max == 4000

    def test_mcp_write_page_covers_every_source_character(self) -> None:
        text = "u" * 20_000
        blocks = mcp_pipeline._source_blocks([("doc", text)], 4000)

        assert len(blocks) == 5
        recovered = "".join(block.split("\n\n", 1)[1] for block in blocks)
        assert recovered == text


def _ctx(deps: WriteDeps) -> RunContext[WriteDeps]:
    return cast(RunContext[WriteDeps], _Ctx(deps))


def _deps(text: str, *, char_max: int = 0) -> WriteDeps:
    return WriteDeps(
        page_title="Test page",
        source_texts=[("doc", text)],
        source_char_max=char_max,
    )
