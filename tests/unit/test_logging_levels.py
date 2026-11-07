"""Per-module log levels: taming third-party loggers one name at a time."""

from __future__ import annotations

import logging

import pytest

from kompend.logging import set_module_levels


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
