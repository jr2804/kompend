"""One house sentence for "no LLM model configured".

Six surfaces used to word the same condition six ways — some naming the env
var, some the config key, some neither — so a reader had to work out which hint
applied to the command they had just run. The sentence now lives once in
:mod:`kompend.llm.messages`, names both the key and the command, and every
surface raises it.

The guards here are: the literal exists exactly once in ``src/kompend``, and
each surface (CLI, MCP) reports the sentence verbatim. The ACP surface is
pinned by ``tests/unit/test_acp_model_selection.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kompend.cli.app import app
from kompend.core.models.references import Author, Reference
from kompend.core.pipeline.references import ReferenceStore
from kompend.llm import messages as messages_module
from kompend.llm.messages import NO_MODEL_CONFIGURED_MESSAGE
from kompend.mcp.tools.pipeline import wiki_organize
from kompend.mcp.tools.query import query_wiki
from tests.helpers.cli import CliRunner

_OWNER = Path(str(messages_module.__file__)).resolve()
_SRC_ROOT = _OWNER.parents[1]
_SENTENCE_START = "No LLM model configured"


class TestTheCliSurface:
    def test_references_fetch_reports_the_sentence(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        (tmp_path / ".kompend").mkdir(parents=True, exist_ok=True)
        (tmp_path / ".kompend" / "config.toml").write_text("", encoding="utf-8")
        (tmp_path / "sources").mkdir(exist_ok=True)
        (tmp_path / "wiki").mkdir(exist_ok=True)
        store = ReferenceStore(tmp_path / ".kompend" / "references.json")
        store.add_source(
            "sources/paper.pdf",
            [
                Reference(
                    id="jones2019",
                    type="article-journal",
                    title="Graph Networks",
                    author=[Author(family="Jones", given="A")],
                    issued="2019",
                ),
            ],
        )
        store.save()
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("KOMP_LLM_DEFAULT_MODEL", "")

        result = CliRunner().invoke(app, ["references", "fetch"])

        assert result.exit_code == 1
        assert NO_MODEL_CONFIGURED_MESSAGE in result.output


class TestTheMcpSurface:
    def test_organize_reports_the_sentence(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        vault = _vault_with_one_page(tmp_path)
        monkeypatch.setenv("KOMP_LLM_DEFAULT_MODEL", "")

        result = wiki_organize(wiki_dir=str(vault))

        assert result["status"] == "error"
        assert result["message"] == NO_MODEL_CONFIGURED_MESSAGE

    def test_query_reports_the_sentence_before_its_hint(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        vault = _vault_with_one_page(tmp_path)
        monkeypatch.setenv("KOMP_LLM_DEFAULT_MODEL", "")

        result = query_wiki("transformers attention", wiki_dir=str(vault))

        assert result["status"] == "error", result
        assert str(result["message"]).startswith(NO_MODEL_CONFIGURED_MESSAGE)
        assert "--raw" in str(result["message"])


def test_the_sentence_names_the_key_and_the_command() -> None:
    """Both halves of the fix stay in the sentence, not in a caller's suffix."""
    assert NO_MODEL_CONFIGURED_MESSAGE == "No LLM model configured. Set compile.llm.default_model or run 'kompend config set'."
    assert "compile.llm.default_model" in NO_MODEL_CONFIGURED_MESSAGE
    assert "kompend config set" in NO_MODEL_CONFIGURED_MESSAGE


def test_the_sentence_literal_lives_in_exactly_one_source_file() -> None:
    """A second copy is how the six wordings happened the first time."""
    offenders = [
        f"{path.relative_to(_SRC_ROOT.parent).as_posix()}:{line_no}"
        for path in _SRC_ROOT.rglob("*.py")
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if _SENTENCE_START in line and path.resolve() != _OWNER
    ]

    assert offenders == [], f"the house sentence must come from kompend.llm.messages, found copies in: {offenders}"


def _vault_with_one_page(tmp_path: Path) -> Path:
    vault = tmp_path / "vault"
    (vault / ".kompend").mkdir(parents=True)
    (vault / ".kompend" / "config.toml").write_text("", encoding="utf-8")
    (vault / "wiki" / "concepts").mkdir(parents=True)
    (vault / "wiki" / "index.md").write_text("# Wiki Index\n", encoding="utf-8")
    (vault / "wiki" / "concepts" / "transformers.md").write_text(
        "---\ntype: Concept\ntitle: Transformers\n---\n\nTransformers use self-attention mechanisms.\n",
        encoding="utf-8",
    )
    return vault
