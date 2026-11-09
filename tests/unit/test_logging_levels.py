"""Per-module log levels: taming third-party loggers one name at a time."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from rich.logging import RichHandler

from kompend.core.config import DEFAULTS, resolve_config
from kompend.logging import (
    THIRD_PARTY_LOG_LEVELS,
    apply_logging_config,
    attach_run_log,
    configure_logging,
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
        assert {"docling", "transformers", "httpx", "ollama"} <= set(THIRD_PARTY_LOG_LEVELS)
        assert THIRD_PARTY_LOG_LEVELS["docling"] == "DEBUG"
        assert THIRD_PARTY_LOG_LEVELS["onnxruntime"] == "CRITICAL"


class TestApplyLoggingConfig:
    """``[logging] modules`` resolves through the config engine (knox-nnm6)."""

    def test_the_key_exists_in_defaults(self) -> None:
        assert DEFAULTS["logging"]["modules"] == {}

    def test_the_key_resolves_and_reaches_the_logger_levels(self, tmp_path: Path) -> None:
        (tmp_path / "kompend.toml").write_text(
            '[logging]\nmodules = { docling = "WARNING", transformers = "ERROR" }\n',
            encoding="utf-8",
        )

        cfg = resolve_config(config_file=tmp_path / "kompend.toml", base_dir=tmp_path)

        assert cfg.get("logging.modules") == {"docling": "WARNING", "transformers": "ERROR"}

        apply_logging_config(cfg)
        try:
            assert logging.getLogger("docling").level == logging.WARNING
            assert logging.getLogger("transformers").level == logging.ERROR
            # Untouched names keep the built-in routing.
            assert logging.getLogger("httpx").level == logging.DEBUG
        finally:
            set_module_levels({})

    def test_empty_mapping_keeps_the_builtin_routing(self) -> None:
        apply_logging_config(None)
        try:
            assert logging.getLogger("docling").level == logging.DEBUG
        finally:
            set_module_levels({})


def test_the_terminal_handler_does_not_read_config_brackets_as_markup() -> None:
    """A log message names its config key as the TOML table it is.

    With Rich markup on, ``[ingest] parse_min_free_commit_mb`` reached the user
    as ``lower  parse_min_free_commit_mb`` — the table name silently eaten.
    """
    configure_logging()
    handler = logging.getLogger().handlers[0]
    assert isinstance(handler, RichHandler)
    assert handler.markup is False
