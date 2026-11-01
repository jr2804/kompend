"""The per-file parse deadline: pass-through, timeout, and worker reuse."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest

from kompend.core.pipeline.bounded_parse import DeadlineWorker, ParseDeadlineExceeded


class _FakeParser:
    """Minimal picklable stand-in for a parser: only ``parse`` is used."""

    def __init__(self, *, delay: float = 0.0, error: BaseException | None = None) -> None:
        self.delay = delay
        self.error = error

    def parse(self, target: Path) -> Any:
        if self.delay:
            time.sleep(self.delay)
        if self.error is not None:
            raise self.error
        return f"parsed:{Path(target).name}"


class TestInProcess:
    def test_zero_timeout_parses_in_process(self, tmp_path: Path) -> None:
        with DeadlineWorker(0.0) as worker:
            assert worker.parse(_FakeParser(), tmp_path / "doc.md") == "parsed:doc.md"

    def test_in_process_errors_propagate_unchanged(self, tmp_path: Path) -> None:
        parser = _FakeParser(error=ValueError("bad page"))

        with DeadlineWorker(0.0) as worker, pytest.raises(ValueError, match="bad page"):
            worker.parse(parser, tmp_path / "doc.md")


class TestDeadline:
    def test_a_hung_parse_is_killed_and_reported(self, tmp_path: Path) -> None:
        target = tmp_path / "stuck.pdf"

        with DeadlineWorker(5.0) as worker, pytest.raises(ParseDeadlineExceeded) as excinfo:
            worker.parse(_FakeParser(delay=120.0), target)

        assert excinfo.value.timeout_s == 5.0
        assert excinfo.value.source_file == target
        assert "parse_timeout_s" in str(excinfo.value)

    def test_the_worker_survives_a_deadline_failure(self, tmp_path: Path) -> None:
        """A killed worker must be replaced, not leave the rest of the run broken."""
        with DeadlineWorker(5.0) as worker:
            with pytest.raises(ParseDeadlineExceeded):
                worker.parse(_FakeParser(delay=120.0), tmp_path / "stuck.pdf")

            assert worker.parse(_FakeParser(), tmp_path / "next.md") == "parsed:next.md"

    def test_a_failing_parse_surfaces_its_type_and_message(self, tmp_path: Path) -> None:
        parser = _FakeParser(error=RuntimeError("all parsecraft passes failed"))

        with DeadlineWorker(120.0) as worker, pytest.raises(RuntimeError, match="RuntimeError: all parsecraft passes failed"):
            worker.parse(parser, tmp_path / "doc.md")

    def test_a_finished_parse_returns_the_document(self, tmp_path: Path) -> None:
        with DeadlineWorker(120.0) as worker:
            assert worker.parse(_FakeParser(), tmp_path / "doc.md") == "parsed:doc.md"
