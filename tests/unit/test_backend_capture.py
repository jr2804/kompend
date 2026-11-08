"""knox-drz1: backend stdout/stderr banners land in the run log, attributed.

Contract under test: a backend that ``print()``s (or writes to stderr)
during a parse has those lines captured and logged as INFO records naming
the backend and the source file, the terminal receives nothing, writes from
unregistered threads pass through unchanged, and chatty backends are capped
per parse with the remainder summarised. The parse result itself is never
affected.
"""

from __future__ import annotations

import io
import logging
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from kompend.core.models import ParsedDocument
from kompend.core.pipeline.backend_capture import (
    DEFAULT_MAX_BANNER_LINES,
    BackendBanners,
    backend_name,
    capture_backend_output,
    log_backend_banners,
)
from kompend.core.pipeline.bounded_parse import DeadlineWorker
from kompend.core.pipeline.stages.parse_sources import _parse_source
from kompend.core.reporting.ledger import stage_ledger
from kompend.preprocessing.host_memory import HeavyParseGuard
from kompend.preprocessing.parsers import DocumentParser

_LOGGER = "kompend.core.pipeline.backend_capture"


class _PrintingBackend:
    """Picklable backend that prints a banner before parsing."""

    def backend_identity(self) -> str:
        return "banner-backend"

    def parse(self, target: Path) -> ParsedDocument:  # noqa: ARG002 — target unused
        print("weights loaded")  # noqa: T201 — the backend under test
        return ParsedDocument(source_file=target, elements=[], title="doc")


class _PrintingChain:
    """Parser facade matching the pieces ``_parse_source`` touches."""

    def backend_identity(self) -> str:
        return "chain"

    def route(self, source_file: Path) -> _PrintingBackend:  # noqa: ARG002 — any backend
        return _PrintingBackend()


def test_a_printed_banner_is_captured_with_attribution(caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    terminal_out, terminal_err = io.StringIO(), io.StringIO()
    monkeypatch.setattr("sys.stdout", terminal_out)
    monkeypatch.setattr("sys.stderr", terminal_err)

    banners = BackendBanners(source="sources/stuck.md", backend="docling")
    with capture_backend_output(banners):
        print("Loading weights 745")  # noqa: T201 — the backend under test
        print("conversion finished", file=sys.stderr)  # noqa: T201 — the backend under test
    # Read the terminal streams NOW: logging after the block may legitimately
    # write to a handler that resolves sys.stderr lazily (a handler another
    # test installed), and that is not the capture's doing.
    printed_out, printed_err = terminal_out.getvalue(), terminal_err.getvalue()
    with caplog.at_level(logging.INFO, logger=_LOGGER):
        log_backend_banners(banners)

    assert banners.lines == ["Loading weights 745", "conversion finished"]
    assert [r.getMessage() for r in _records(caplog)] == [
        "backend banner [docling] sources/stuck.md: Loading weights 745",
        "backend banner [docling] sources/stuck.md: conversion finished",
    ]
    # Replace, not tee: the terminal received none of it.
    assert printed_out == ""
    assert printed_err == ""


def test_unregistered_threads_pass_through(monkeypatch: pytest.MonkeyPatch) -> None:
    terminal = io.StringIO()
    monkeypatch.setattr("sys.stdout", terminal)

    banners = BackendBanners(source="sources/a.md", backend="docling")
    with capture_backend_output(banners):
        stranger = threading.Thread(target=print, args=("from another thread",))
        stranger.start()
        stranger.join()

    assert banners.lines == []
    assert terminal.getvalue() == "from another thread\n"


def test_chatty_backends_are_capped_with_a_summary(caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    terminal = io.StringIO()
    monkeypatch.setattr("sys.stdout", terminal)

    banners = BackendBanners(source="s.md", backend="transformers", max_lines=3)
    with capture_backend_output(banners):
        for index in range(5):
            print(f"layer {index}")  # noqa: T201 — the backend under test
    with caplog.at_level(logging.INFO, logger=_LOGGER):
        log_backend_banners(banners)

    messages = [r.getMessage() for r in _records(caplog)]
    assert messages[:3] == [f"backend banner [transformers] s.md: layer {i}" for i in range(3)]
    assert messages[3] == "backend banner [transformers] s.md: … and 2 more banner line(s) over the per-parse cap"
    assert len(messages) == 4


def test_carriage_return_progress_becomes_one_line() -> None:
    banners = BackendBanners(source="s.md", backend="tqdm-ish", max_lines=DEFAULT_MAX_BANNER_LINES)
    with capture_backend_output(banners):
        sys.stdout.write("17%\r18%\r19%\n")  # noqa: T201 — the backend under test
    assert banners.lines == ["19%"]


def test_crlf_is_a_line_terminator_not_a_progress_overwrite() -> None:
    banners = BackendBanners(source="s.md", backend="windows-ish", max_lines=DEFAULT_MAX_BANNER_LINES)
    with capture_backend_output(banners):
        sys.stdout.write("first line\r\nsecond line\r\n")  # noqa: T201 — the backend under test
    assert banners.lines == ["first line", "second line"]


def test_backend_name_prefers_the_identity_hook() -> None:
    class _Named:
        def backend_identity(self) -> str:
            return "docling-x"

    class _Broken:
        def backend_identity(self) -> str:
            raise RuntimeError("no identity")

    class _Plain:
        pass

    assert backend_name(_Named()) == "docling-x"
    assert backend_name(_Plain()) == "_Plain"
    assert backend_name(_Broken()) == "_Broken"


def test_parse_source_logs_the_banner_before_the_ledger_row(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    source = tmp_path / "stuck.md"
    source.write_text("hello")
    caplog.set_level(logging.INFO)

    with stage_ledger(tmp_path / "led.db") as ledger:
        parsed = _parse_source(
            _pipeline(tmp_path),
            source,
            use_cache=False,
            content_hash="",
            heavy_guard=HeavyParseGuard(),
            ledger=ledger,
            parser_identity=None,
        )

    assert parsed is not None
    banner = [r.getMessage() for r in _records(caplog)]
    assert banner == ["backend banner [banner-backend] sources/stuck.md: weights loaded"]


def _pipeline(tmp_path: Path) -> Any:
    return SimpleNamespace(
        parser=cast("DocumentParser", _PrintingChain()),
        base_directory=tmp_path,
        rel_file=lambda file: f"sources/{Path(file).name}",
        cache_directory=lambda: tmp_path / "cache",
        config=SimpleNamespace(output=SimpleNamespace(rumdl_config=None, flavor=None)),
    )


def test_the_in_process_deadline_path_attributes_banners(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    worker = DeadlineWorker(0.0)
    target = tmp_path / "doc.md"
    caplog.set_level(logging.INFO)

    doc = worker.parse(_PrintingBackend(), target, source="sources/doc.md", backend="docling")

    assert doc.title == "doc"
    assert [r.getMessage() for r in _records(caplog)] == ["backend banner [docling] sources/doc.md: weights loaded"]


def test_the_deadline_worker_ships_banners_to_the_parent(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    worker = DeadlineWorker(30.0)
    target = tmp_path / "doc.md"
    target.write_text("hello")
    caplog.set_level(logging.INFO)
    deadline = time.monotonic() + 60.0

    doc = worker.parse(_PrintingBackend(), target, source="sources/doc.md", backend="docling")
    worker.close()

    assert doc.title == "doc"
    assert time.monotonic() < deadline  # sanity: the spawn must not hang the suite
    assert [r.getMessage() for r in _records(caplog)] == ["backend banner [docling] sources/doc.md: weights loaded"]


def _records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.name == _LOGGER]
