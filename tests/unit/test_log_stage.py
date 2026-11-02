"""log_stage must stay quiet when there is no bundle to append to."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from kompend.core.paths import VaultPaths
from kompend.output.pages.log import log_stage


class TestLogStageWithoutABundle:
    def test_a_missing_output_dir_is_not_a_warning(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        vault = VaultPaths(tmp_path)

        with caplog.at_level(logging.DEBUG, logger="kompend.output.pages.log"):
            log_stage(vault, action="Ingest", detail="102 document(s)")

        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert "skipping the Ingest log entry" in caplog.text
        assert not vault.output_dir.exists()

    def test_an_existing_bundle_receives_the_entry(self, tmp_path: Path) -> None:
        vault = VaultPaths(tmp_path)
        vault.output_dir.mkdir()

        log_stage(vault, action="Ingest", detail="102 document(s)")

        written = (vault.output_dir / "log.md").read_text(encoding="utf-8")
        assert "Ingest" in written
        assert "102 document(s)" in written
