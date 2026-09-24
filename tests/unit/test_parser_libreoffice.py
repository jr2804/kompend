"""Lazy LibreOffice discovery for docling (``preprocessing.parsers.libreoffice``)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from knox.preprocessing.parsers import libreoffice
from knox.preprocessing.parsers.libreoffice import (
    ENV_VAR,
    ensure_libreoffice_env,
    needs_libreoffice,
)


class TestNeedsLibreoffice:
    def test_ole_formats(self) -> None:
        assert needs_libreoffice(Path("a.doc"))
        assert needs_libreoffice(Path("A.RTF"))

    def test_others(self) -> None:
        assert not needs_libreoffice(Path("a.pdf"))
        assert not needs_libreoffice(Path("a.docx"))


class TestResolutionOrder:
    def test_explicit_existing_value_is_left_untouched(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        custom = tmp_path / "soffice-custom"
        custom.write_text("", encoding="utf-8")
        monkeypatch.setenv(ENV_VAR, str(custom))

        assert ensure_libreoffice_env() == custom
        assert os.environ[ENV_VAR] == str(custom)

    def test_explicit_missing_value_is_not_overwritten(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # A user's explicit (broken) value wins over our discovery, per setdefault.
        monkeypatch.setenv(ENV_VAR, str(tmp_path / "nope"))
        monkeypatch.setattr(libreoffice, "_from_path", lambda: Path("C:/discovered/soffice.exe"))

        assert ensure_libreoffice_env() is None
        assert os.environ[ENV_VAR] == str(tmp_path / "nope")

    def test_path_hit_is_installed_via_setdefault(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        located = tmp_path / "soffice"
        located.write_text("", encoding="utf-8")
        monkeypatch.setattr(libreoffice, "_from_path", lambda: located)

        assert ensure_libreoffice_env() == located
        assert os.environ[ENV_VAR] == str(located)

    def test_platform_default_used_when_path_misses(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        found = tmp_path / "Program Files" / "LibreOffice" / "program" / "soffice.exe"
        monkeypatch.setattr(libreoffice, "_from_path", lambda: None)
        monkeypatch.setattr(libreoffice, "_from_platform_default", lambda: found)

        assert ensure_libreoffice_env() == found
        assert os.environ[ENV_VAR] == str(found)

    def test_nothing_found_returns_none_and_sets_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(libreoffice, "_from_path", lambda: None)
        monkeypatch.setattr(libreoffice, "_from_platform_default", lambda: None)

        assert ensure_libreoffice_env() is None
        assert ENV_VAR not in os.environ


class TestPlatformDefault:
    def test_windows_program_files(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        soffice = tmp_path / "LibreOffice" / "program" / "soffice.exe"
        soffice.parent.mkdir(parents=True)
        soffice.write_text("", encoding="utf-8")
        monkeypatch.setenv("PROGRAMFILES", str(tmp_path))
        monkeypatch.delenv("PROGRAMFILES(X86)", raising=False)

        assert libreoffice._from_platform_default() == soffice

    def test_none_when_no_root_has_it(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PROGRAMFILES", str(tmp_path))
        monkeypatch.delenv("PROGRAMFILES(X86)", raising=False)

        assert libreoffice._from_platform_default() is None


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV_VAR, raising=False)
