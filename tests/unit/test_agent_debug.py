"""Tests for per-LLM-call debug dumps (``KNOX_DEBUG_LLM_DIR``)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from knox.agents.debug import save_debug_prompt


class _FakeAgent:
    stage = "compile"
    current_source_id = "src-1"


def _dump(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, content: str) -> Path:
    monkeypatch.setenv("KNOX_DEBUG_LLM_DIR", str(tmp_path))
    save_debug_prompt(
        _FakeAgent(),
        "ollama/qwen3",
        [{"role": "user", "content": content}],
        {},
        '{"ok": true}',
        "stop",
    )
    files = list(tmp_path.iterdir())
    assert len(files) == 1
    return files[0]


def test_save_debug_prompt_writes_valid_json(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    f = _dump(monkeypatch, tmp_path, "plain text")
    data = json.loads(f.read_text(encoding="utf-8"))
    assert data["stage"] == "compile"
    assert data["messages"][0]["content"] == "plain text"


def test_save_debug_prompt_survives_lone_surrogates(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Regression: lone surrogates in messages must not abort the dump.

    Before the fix, ``write_text(..., encoding="utf-8")`` raised
    ``UnicodeEncodeError: surrogates not allowed`` and left a zero-byte file
    behind, logging "Failed to save debug prompt".
    """
    f = _dump(monkeypatch, tmp_path, "bad \ud83d text")
    assert f.stat().st_size > 0
    data = json.loads(f.read_text(encoding="utf-8"))
    # backslashreplace writes \ud83d as a JSON escape — round-trips losslessly.
    assert data["messages"][0]["content"] == "bad \ud83d text"
