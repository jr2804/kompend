"""Tests for the capture and distill modules."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, Mock

import pytest
import yaml

from kompend.agents.capture_models import CaptureOutput
from kompend.agents.deps import DistillDeps
from kompend.agents.distill import DistillAgent
from kompend.capture.capture import (
    _merge_tags,
    capture,
)
from kompend.capture.consolidate import consolidate, find_duplicates
from kompend.capture.distill import (
    CapturedNote,
    _group_by_tag,
    _parse_frontmatter,
    distill,
)
from kompend.capture.learn import learn
from kompend.cli.cmds.knowledge.capture import _build_references_section
from kompend.core.pipeline.ingest import extract_text_from_cached
from kompend.core.pipeline.parse_cache import write_document
from kompend.core.store import vault_db_path
from kompend.output.markdown import MarkdownWriter
from kompend.output.models import Document


class TestParseFrontmatter:
    def test_with_frontmatter(self) -> None:
        text = "---\ntitle: My Note\ntags:\n  - python\n  - testing\n---\n\nSome content here."
        meta, body = _parse_frontmatter(text)
        assert meta["title"] == "My Note"
        assert meta["tags"] == ["python", "testing"]
        assert body.strip() == "Some content here."

    def test_without_frontmatter(self) -> None:
        text = "Just plain markdown."
        meta, body = _parse_frontmatter(text)
        assert meta == {}
        assert body == "Just plain markdown."

    def test_malformed_frontmatter(self) -> None:
        text = "---\nnot valid yaml: [unclosed\n---\nbody"
        meta, body = _parse_frontmatter(text)
        assert meta == {}
        assert body == text


class TestGroupByTag:
    def test_groups_by_first_tag(self) -> None:
        notes = [
            CapturedNote(title="a", content="x", tags=["python", "test"]),
            CapturedNote(title="b", content="y", tags=["python"]),
            CapturedNote(title="c", content="z", tags=["rust"]),
        ]
        groups = _group_by_tag(notes)
        assert set(groups.keys()) == {"python", "rust"}
        assert len(groups["python"]) == 2
        assert len(groups["rust"]) == 1

    def test_untagged_goes_to_general(self) -> None:
        notes = [CapturedNote(title="a", content="x", tags=[])]
        groups = _group_by_tag(notes)
        assert "general" in groups
        assert len(groups["general"]) == 1


class TestDistill:
    def test_missing_captured_dir_returns_empty(self, tmp_path: Path) -> None:
        sources_dir = tmp_path / "sources"
        sources_dir.mkdir()
        result = distill(sources_dir)
        assert result == []

    def test_empty_captured_dir_returns_empty(self, tmp_path: Path) -> None:
        sources_dir = tmp_path / "sources"
        (sources_dir / "_captured").mkdir(parents=True)
        result = distill(sources_dir)
        assert result == []

    def test_fallback_without_llm(self, tmp_path: Path) -> None:
        sources_dir = tmp_path / "sources"
        captured_dir = sources_dir / "_captured"
        captured_dir.mkdir(parents=True)
        (captured_dir / "note-1.md").write_text(
            "---\ntitle: Decision A\ntags:\n  - arch\n---\n\nUse QKV projections.\n",
            encoding="utf-8",
        )
        (captured_dir / "note-2.md").write_text(
            "---\ntitle: Discovery B\ntags:\n  - arch\n---\n\nRoPE works well.\n",
            encoding="utf-8",
        )

        paths = distill(sources_dir)
        assert len(paths) == 1
        output = paths[0].read_text(encoding="utf-8")
        assert "type: distilled" in output
        assert "Decision A" in output
        assert "Discovery B" in output
        assert "note-1.md" in output
        assert "note-2.md" in output

    def test_with_mock_llm(self, tmp_path: Path) -> None:
        sources_dir = tmp_path / "sources"
        captured_dir = sources_dir / "_captured"
        captured_dir.mkdir(parents=True)
        (captured_dir / "note-1.md").write_text(
            "---\ntitle: Note A\ntags:\n  - ml\n---\n\nSome content.\n",
            encoding="utf-8",
        )

        mock_agent = MagicMock()
        mock_agent.run.return_value = MagicMock(
            title="Synthesized ML Page",
            content="## Overview\n\nIntegrated content.",
            tags=["ml", "synthesized"],
        )
        mock_agent.timeout = 120

        paths = distill(sources_dir, distill_agent=mock_agent)
        assert len(paths) == 1
        output = paths[0].read_text(encoding="utf-8")
        assert "Synthesized ML Page" in output
        assert "Integrated content" in output
        assert "synthesized" in output
        assert mock_agent.run.called

    def test_llm_failure_falls_back(self, tmp_path: Path) -> None:
        sources_dir = tmp_path / "sources"
        captured_dir = sources_dir / "_captured"
        captured_dir.mkdir(parents=True)
        (captured_dir / "note-1.md").write_text(
            "---\ntitle: Note A\ntags:\n  - test\n---\n\nContent A.\n",
            encoding="utf-8",
        )

        mock_result = MagicMock()
        mock_result.json_data = None
        mock_agent = MagicMock()
        mock_agent.run.side_effect = Exception("LLM failed")
        mock_agent.timeout = 120

        paths = distill(sources_dir, distill_agent=mock_agent)
        assert len(paths) == 1
        output = paths[0].read_text(encoding="utf-8")
        assert "Content A" in output

    def test_multiple_tags_produce_multiple_pages(self, tmp_path: Path) -> None:
        sources_dir = tmp_path / "sources"
        captured_dir = sources_dir / "_captured"
        captured_dir.mkdir(parents=True)
        for tag in ("alpha", "beta"):
            (captured_dir / f"note-{tag}.md").write_text(
                f"---\ntitle: Note {tag}\ntags:\n  - {tag}\n---\n\nContent {tag}.\n",
                encoding="utf-8",
            )

        paths = distill(sources_dir)
        assert len(paths) == 2

    def test_agent_renders_template_with_distill_deps(self) -> None:
        """Regression: DistillAgent.run() raised ValueError because the
        template requires ``notes``/``group_tag`` vars that weren't on deps.
        """
        agent = DistillAgent("test-model")
        deps = DistillDeps(
            notes=[{"title": "T", "content": "body", "source": "x.md"}],
            group_tag="ml",
            text="go",
            source_id="repro",
        )
        up = agent.render_user_prompt(deps)
        assert "body" in up
        assert "ml" in up
        sp = agent.render_system_prompt(deps)
        assert sp == ""


class TestCapture:
    def test_no_llm_applies_user_tags(self, tmp_path: Path) -> None:
        paths = capture(
            "The transformer uses d_model=512 and h=8 heads with RoPE positional encodings for the attention mechanism.",
            sources_dir=tmp_path,
            tags=["architecture", "decoder"],
        )
        assert len(paths) == 1
        content = paths[0].read_text(encoding="utf-8")
        assert "architecture" in content
        assert "decoder" in content

    def test_merge_tags_user_first_dedup(self) -> None:
        merged = _merge_tags(["alpha", "Beta"], ["beta", "gamma"])
        assert merged == ["alpha", "Beta", "gamma"]

    def test_merge_tags_empty(self) -> None:
        assert _merge_tags([], []) == []


class TestWriteCmd:
    """knox-zkfd: write resolves sources through cache, writes to write/ subfolder."""

    def test_write_reads_cache_not_raw_bytes(self, tmp_path: Path) -> None:
        """When a parse-cache entry exists, write_cmd extracts clean text, not raw PDF bytes."""
        # Seed a parse-cache entry for a fake PDF into the store.
        cache_dir = tmp_path / ".kompend" / "cache"
        cache_dir.mkdir(parents=True)
        write_document(
            vault_db_path(cache_dir),
            "paper",
            {"elements": [{"content": "This is clean extracted text from the PDF."}]},
        )
        cache_file = cache_dir / "paper.json"
        text = extract_text_from_cached(cache_file)
        assert "clean extracted text" in text
        assert "%PDF" not in text

    def test_write_default_frontmatter(self, tmp_path: Path) -> None:
        """Kompend-rxod+zkfd: the write page carries kind, compiled_at, sources."""
        compiled_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        fm = {
            "title": "My Page",
            "kind": "page",
            "compiled_at": compiled_at,
            "sources": ["paper"],
        }
        doc = Document(title="My Page", body="Content.", frontmatter=fm, node_type="page")
        writer = MarkdownWriter()
        rendered = writer.renderer.render(doc)
        assert "kind: page" in rendered
        assert "compiled_at:" in rendered
        assert "sources:" in rendered

    def test_references_section_built(self) -> None:
        """_build_references_section lists sources and related wikilinks."""
        section = _build_references_section(
            [("paper_a", "text"), ("paper_b", "text")],
            related_titles=["Listening Effort", "Better Ear"],
        )
        assert "paper_a" in section
        assert "paper_b" in section
        assert "[[Listening Effort]]" in section
        assert "[[Better Ear]]" in section


class TestLearn:
    """learn() writes to sources/_learned/ with tags + attributes."""

    def test_writes_to_learned_dir(self, tmp_path: Path) -> None:
        path = learn("The transformer uses d_model=512", sources_dir=tmp_path)
        assert path.parent.name == "_learned"
        assert path.exists()

    def test_attributes_in_frontmatter(self, tmp_path: Path) -> None:
        path = learn(
            "The transformer uses d_model=512 and h=8 heads.",
            sources_dir=tmp_path,
            tags=["architecture"],
            attributes={"version": "2", "status": "confirmed"},
        )
        content = path.read_text(encoding="utf-8")
        assert "okf_version" in content
        meta = yaml.safe_load(content[4:].split("\n---\n", 1)[0])
        # Quoted, so a numeric-looking attribute stays the string it was passed
        # as. The previous hand-built frontmatter wrote it bare, and YAML read
        # it back as the integer 2.
        assert meta["attributes"] == {"version": "2", "status": "confirmed"}
        assert meta["tags"] == ["architecture"]


class TestCaptureSufficiency:
    """capture() rejects inputs too short to extract from."""

    def test_empty_input_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="empty"):
            capture("   \n  ", sources_dir=tmp_path)

    def test_short_input_captured_without_llm(self, tmp_path: Path) -> None:
        # Without an LLM, short input is stored as a raw note (no arbitrary
        # length gate — the agent is the judge of "nonsense" when present).
        paths = capture("short", sources_dir=tmp_path)
        assert len(paths) == 1

    def test_agent_refusal_writes_nothing(self, tmp_path: Path) -> None:

        agent = Mock()
        agent.run.return_value = CaptureOutput(items=[], refused=True, refusal_reason="gibberish")
        paths = capture("asdf qwerty zxcv", sources_dir=tmp_path, capture_agent=agent)
        assert paths == []

    def test_agent_empty_items_writes_nothing(self, tmp_path: Path) -> None:

        agent = Mock()
        agent.run.return_value = CaptureOutput(items=[])
        paths = capture("Just a greeting, nothing else here.", sources_dir=tmp_path, capture_agent=agent)
        assert paths == []

    def test_attributes_in_frontmatter(self, tmp_path: Path) -> None:
        paths = capture(
            "The transformer uses d_model=512 and h=8 heads with RoPE positional encodings for the attention mechanism.",
            sources_dir=tmp_path,
            attributes={"version": "2"},
        )
        content = paths[0].read_text(encoding="utf-8")
        assert "okf_version" in content
        meta = yaml.safe_load(content[4:].split("\n---\n", 1)[0])
        assert meta["attributes"] == {"version": "2"}


class TestOkfFrontmatter:
    """Every source-corpus writer emits OKF 0.2 frontmatter.

    The source notes are part of the same corpus as the compiled wiki, so they
    declare the spec version they were written against; a reader cannot
    otherwise tell a pre-migration note from a conformant one.
    """

    def test_learn_declares_okf_version(self, tmp_path: Path) -> None:
        meta = self._meta(learn("A fact worth keeping.", sources_dir=tmp_path))
        assert meta["okf_version"] == "0.2"
        assert meta["type"] == "learn"

    def test_capture_declares_okf_version(self, tmp_path: Path) -> None:
        meta = self._meta(capture("Something captured.", sources_dir=tmp_path)[0])
        assert meta["okf_version"] == "0.2"
        assert meta["type"] == "note"

    def test_special_characters_do_not_break_the_yaml(self, tmp_path: Path) -> None:
        """A colon in a title or tag used to produce invalid frontmatter.

        The writers interpolated values into hand-built YAML; anything with a
        colon parsed as a nested mapping and a newline split the block.
        """
        meta = self._meta(
            capture(
                "Ratio: 3:1 is the documented setting.",
                sources_dir=tmp_path,
                tags=["tag: with colon"],
                attributes={"note": "value: with colon"},
            )[0]
        )
        assert meta["tags"] == ["tag: with colon"]
        assert meta["attributes"]["note"] == "value: with colon"
        assert isinstance(meta["title"], str)

    def test_existing_corpus_notes_still_read(self, tmp_path: Path) -> None:
        """Pre-migration notes carry no okf_version; readers must not require one."""
        path = tmp_path / "capture-old.md"
        path.write_text(
            "---\ntype: note\ntitle: Old\ncaptured_at: 20260101T000000\ntags:\n  - legacy\nattributes:\n  source: capture\n---\n\nBody.\n",
            encoding="utf-8",
        )
        meta = self._meta(path)
        assert "okf_version" not in meta
        assert meta["tags"] == ["legacy"]
        assert meta["captured_at"] == "20260101T000000"

    @staticmethod
    def _meta(path: Path) -> dict:
        content = path.read_text(encoding="utf-8")
        assert content.startswith("---\n")
        return yaml.safe_load(content[4:].split("\n---\n", 1)[0])


class TestDistillUpdateAndArchive:
    """distill() updates existing per-group pages and archives originals."""

    def test_repeated_distill_updates_same_page(self, tmp_path: Path) -> None:
        self._make_note(tmp_path, "note-a.md", "alpha", "First note about alpha.")
        first = distill(tmp_path, archive=False)
        assert len(first) == 1
        page = first[0]
        # Second run with a new note in the same group must UPDATE the same file.
        self._make_note(tmp_path, "note-b.md", "alpha", "Second note about alpha.")
        second = distill(tmp_path, archive=False)
        assert len(second) == 1
        assert second[0] == page  # same path, not a new timestamped file

    def test_archive_moves_originals(self, tmp_path: Path) -> None:
        self._make_note(tmp_path, "note-a.md", "alpha", "First note about alpha.")
        distill(tmp_path, archive=True)
        assert not (tmp_path / "_captured" / "note-a.md").exists()
        assert (tmp_path / "_archive" / "capture" / "note-a.md").exists()

    def test_no_archive_keeps_originals(self, tmp_path: Path) -> None:
        self._make_note(tmp_path, "note-a.md", "alpha", "First note about alpha.")
        distill(tmp_path, archive=False)
        assert (tmp_path / "_captured" / "note-a.md").exists()

    def _make_note(self, sources_dir: Path, name: str, tag: str, body: str) -> Path:
        captured = sources_dir / "_captured"
        captured.mkdir(parents=True, exist_ok=True)
        p = captured / name
        p.write_text(
            f"---\ntitle: {name}\ntags:\n  - {tag}\n---\n\n{body}",
            encoding="utf-8",
        )
        return p


class TestConsolidate:
    """consolidate() finds near-duplicates and archives them."""

    def test_finds_identical_duplicates(self, tmp_path: Path) -> None:
        body = "The PESQ algorithm aligns utterances before scoring."
        self._write(tmp_path, "_captured", "a.md", body)
        self._write(tmp_path, "_captured", "b.md", body)
        groups = find_duplicates(
            [
                CapturedNote(title="a", content=body, source_file=tmp_path / "_captured" / "a.md"),
                CapturedNote(title="b", content=body, source_file=tmp_path / "_captured" / "b.md"),
            ]
        )
        assert len(groups) == 1
        assert len(groups[0].duplicates) == 1

    def test_consolidate_archives_duplicates(self, tmp_path: Path) -> None:
        body = "The PESQ algorithm aligns utterances before scoring."
        self._write(tmp_path, "_captured", "a.md", body)
        self._write(tmp_path, "_captured", "b.md", body)
        groups = consolidate(tmp_path)
        assert len(groups) == 1
        assert not (tmp_path / "_captured" / "b.md").exists()
        assert (tmp_path / "_archive" / "capture" / "b.md").exists()

    def test_dry_run_moves_nothing(self, tmp_path: Path) -> None:
        body = "The PESQ algorithm aligns utterances before scoring."
        self._write(tmp_path, "_captured", "a.md", body)
        self._write(tmp_path, "_captured", "b.md", body)
        groups = consolidate(tmp_path, dry_run=True)
        assert len(groups) == 1
        assert (tmp_path / "_captured" / "b.md").exists()

    def _write(self, sources_dir: Path, sub: str, name: str, body: str) -> Path:
        d = sources_dir / sub
        d.mkdir(parents=True, exist_ok=True)
        p = d / name
        p.write_text(f"---\ntitle: {name}\n---\n\n{body}", encoding="utf-8")
        return p
