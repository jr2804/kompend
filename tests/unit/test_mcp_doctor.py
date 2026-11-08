"""The MCP doctor must resolve the recorded embed model the way the CLI does."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

doctor_module = importlib.import_module("kompend.mcp.tools.quality.doctor")
doctor = doctor_module.doctor


def test_mcp_doctor_passes_the_recorded_embed_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    vault = _vault(tmp_path)
    captured: dict[str, object] = {}

    def _run_doctor(**kwargs: object) -> list[object]:
        captured.update(kwargs)
        return []

    monkeypatch.setattr(doctor_module, "run_doctor", _run_doctor)
    monkeypatch.setattr(doctor_module, "recorded_embedding_model", lambda _db: "recorded-digest")

    result = doctor(wiki_dir=str(vault), json_output=True)

    assert result["status"] == "ok"
    assert captured["embed_model"] == "test/embed"
    assert captured["recorded_embed_model"] == "recorded-digest"


def _vault(tmp_path: Path) -> Path:
    vault = tmp_path / "vault"
    (vault / ".kompend").mkdir(parents=True)
    (vault / ".kompend" / "config.toml").write_text(
        '[compile.embedding]\nmodel = "test/embed"\n',
        encoding="utf-8",
    )
    return vault
