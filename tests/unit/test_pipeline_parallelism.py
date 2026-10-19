"""Tests for the interpreter parallelism report (parallelism.py)."""

from __future__ import annotations

import sys

import pytest

from kompend.core.pipeline import parallelism
from kompend.core.pipeline.parallelism import describe_parallelism, gil_enabled


class TestGilEnabled:
    def test_reports_the_running_interpreter(self) -> None:
        """Whatever this interpreter is, the answer must be a real bool."""
        assert isinstance(gil_enabled(), bool)

    def test_missing_attribute_is_treated_as_a_gil_build(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Only a free-threaded build omits the GIL, so absent means enabled."""
        monkeypatch.delattr(sys, "_is_gil_enabled", raising=False)

        assert gil_enabled() is True


class TestDescribeParallelism:
    def test_gil_build_warns_about_cpu_bound_work(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(parallelism, "gil_enabled", lambda: True)

        message = describe_parallelism(4)

        assert "4 workers" in message
        assert "GIL-enabled" in message
        assert "3.14t" in message

    def test_free_threaded_build_reports_real_parallelism(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(parallelism, "gil_enabled", lambda: False)

        message = describe_parallelism(8)

        assert "8 workers" in message
        assert "free-threaded" in message
        assert "GIL-enabled" not in message
