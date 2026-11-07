"""Per-module log levels: taming third-party loggers one name at a time."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from kompend.logging import (
    THIRD_PARTY_LOG_LEVELS,
    attach_run_log,
    detach_run_log,
    route_third_party_loggers,
    set_module_levels,
    set_verbosity,
)


class TestSetModuleLevels:
    """``set_module_levels`` overrides named loggers and resets cleanly."""

    def test_applies_names_case_insensitively(self) -> None:
        set_module_levels({"docling": "debug", "transformers": "WARNING"})
        try:
            assert logging.getLogger("docling").level == logging.DEBUG
            assert logging.getLogger("transformers").level == logging.WARNING
        finally:
            set_module_levels({})

    def test_accepts_integer_levels(self) -> None:
        set_module_levels({"httpx": logging.ERROR})
        try:
            assert logging.getLogger("httpx").level == logging.ERROR
        finally:
            set_module_levels({})

    def test_replacing_the_mapping_resets_previous_overrides(self) -> None:
        set_module_levels({"docling": "DEBUG"})
        assert logging.getLogger("docling").level == logging.DEBUG

        set_module_levels({"transformers": "ERROR"})

        assert logging.getLogger("docling").level == logging.NOTSET
        assert logging.getLogger("transformers").level == logging.ERROR
        set_module_levels({})

    def test_unknown_name_raises_without_partial_application(self) -> None:
        with pytest.raises(ValueError, match=r"Invalid logging level 'LOUD' for module 'docling'"):
            set_module_levels({"docling": "LOUD", "transformers": "DEBUG"})

        assert logging.getLogger("transformers").level == logging.NOTSET


class TestThirdPartyRouting:
    """Default routing for the libraries that dominate parse-time chatter."""

    def test_defaults_reach_the_run_log_not_the_terminal(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        set_verbosity(logging.WARNING)
        route_third_party_loggers()
        attach_run_log("t1", tmp_path)
        try:
            logging.getLogger("docling").debug("layout model detail")
        finally:
            detach_run_log(failed=True, reason="test dump")
            set_module_levels({})

        text = (tmp_path / "run-t1.log").read_text(encoding="utf-8")
        assert "layout model detail" in text
        assert "layout model detail" not in capsys.readouterr().err

    def test_operator_overrides_win_over_defaults(self) -> None:
        route_third_party_loggers({"transformers": "WARNING"})
        try:
            assert logging.getLogger("transformers").level == logging.WARNING
            assert logging.getLogger("docling").level == logging.DEBUG
            assert logging.getLogger("httpx").level == logging.DEBUG
        finally:
            set_module_levels({})

    def test_covers_the_named_libraries(self) -> None:
        assert set(THIRD_PARTY_LOG_LEVELS) == {"docling", "transformers", "httpx", "ollama"}
