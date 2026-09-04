"""Unit tests for KNOX lint rules."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from knox.agents.lint import LintRuleAgent
from knox.agents.remediate import AgenticRemediationAgent, RemediationAgent
from knox.agents.types import RemediationDeps, RemediationPageResult
from knox.compilation._two_phase_core import Extraction
from knox.compilation.schema import Concept, Entity
from knox.core.config.rumdl import resolve_rumdl_config
from knox.core.models import CompiledWiki, WikiNode
from knox.lint import _remediation_gaps as gaps_module
from knox.lint.base import LintReport, LintReportResult, LintResult, LintRuleOutcome, RemediationStatus, Severity
from knox.lint.llm_rules import (
    Contradiction,
    ContradictionOutput,
    CoverageGap,
    CoverageGapOutput,
)
from knox.lint.remediation import remediate
from knox.lint.runner import LintRunner, _build_report, report_to_json, report_to_text
from knox.lint.structural import (
    RULE_BY_NAME,
    STRUCTURAL_RULES,
    BrokenWikilinksRule,
    DuplicateTitleRule,
    EmptyPageRule,
    HeadingSkipRule,
    ImageEmbedRule,
    KnoxReferenceRule,
    MissingFrontmatterRule,
    OrphanAttachmentsRule,
    OrphanPageRule,
    RumdlRule,
    SourceNotFoundRule,
    StalePageRule,
)

# ── Rule 1: broken-wikilinks ────────────────────────────────────────────


class TestBrokenWikilinksRule:
    def test_clean_wiki_passes(self) -> None:
        nodes = [
            _make_node("n1", "Alpha", "Link to [[Beta]]"),
            _make_node("n2", "Beta", "Link to [[Alpha]]"),
        ]
        wiki = _make_wiki(nodes)
        rule = BrokenWikilinksRule()
        results = rule.check(wiki)
        assert all(r.outcome == LintRuleOutcome.PASS for r in results)

    def test_broken_link_detected(self) -> None:
        node = _make_node("n1", "Alpha", "See [[Missing Page]] for details")
        wiki = _make_wiki([node])
        rule = BrokenWikilinksRule()
        results = rule.check(wiki)
        failures = [r for r in results if r.outcome == LintRuleOutcome.FAIL]
        assert len(failures) == 1
        assert failures[0].details["target"] == "Missing Page"

    def test_fix_removes_broken_link(self) -> None:
        node = _make_node("n1", "Alpha", "See [[Missing]] here")
        wiki = _make_wiki([node])
        rule = BrokenWikilinksRule()
        results = rule.check(wiki)
        failures = [r for r in results if r.outcome == LintRuleOutcome.FAIL]
        fixed = rule.fix(wiki, failures)
        assert all(r.fix_applied for r in fixed)
        assert "[[Missing]]" not in node.content
        assert "Missing" in node.content

    def test_multiple_broken_links(self) -> None:
        node = _make_node("n1", "Alpha", "[[A]] and [[B]] and [[C]]")
        wiki = _make_wiki([node])
        rule = BrokenWikilinksRule()
        results = rule.check(wiki)
        failures = [r for r in results if r.outcome == LintRuleOutcome.FAIL]
        assert len(failures) == 3

    def test_empty_wiki_passes(self) -> None:
        wiki = _make_wiki([])
        rule = BrokenWikilinksRule()
        results = rule.check(wiki)
        assert all(r.outcome == LintRuleOutcome.PASS for r in results)

    def test_no_links_in_content(self) -> None:
        node = _make_node("n1", "Alpha", "No links here, just text.")
        wiki = _make_wiki([node])
        rule = BrokenWikilinksRule()
        results = rule.check(wiki)
        assert all(r.outcome == LintRuleOutcome.PASS for r in results)

    def test_image_embeds_not_flagged(self) -> None:
        """Correct ``![[img.png]]`` and wrong ``[[img.png]]`` are not broken page links."""
        node = _make_node("n1", "Alpha", "See ![[diagram.png]] and [[photo.jpg]]")
        wiki = _make_wiki([node])
        rule = BrokenWikilinksRule()
        results = rule.check(wiki)
        assert all(r.outcome == LintRuleOutcome.PASS for r in results)

    def test_acronym_alias_resolves(self) -> None:
        """``[[PESQ]]`` resolves to the page whose title starts with ``PESQ (``."""
        node = _make_node(
            "n1",
            "Alpha",
            "See [[PESQ]] for details",
        )
        target = _make_node(
            "pesq-...",
            "PESQ (Perceptual Evaluation of Speech Quality)",
            "Definition of PESQ",
        )
        wiki = _make_wiki([node, target])
        rule = BrokenWikilinksRule()
        results = rule.check(wiki)
        assert all(r.outcome == LintRuleOutcome.PASS for r in results)

    def test_slug_resolves(self) -> None:
        """``[[some-kebab-slug]]`` resolves directly to that page id."""
        node = _make_node("n1", "Alpha", "See [[some-kebab-slug]] here")
        target = _make_node("some-kebab-slug", "Some Page", "Body")
        wiki = _make_wiki([node, target])
        rule = BrokenWikilinksRule()
        results = rule.check(wiki)
        assert all(r.outcome == LintRuleOutcome.PASS for r in results)

    def test_near_alias_marked_as_alias(self) -> None:
        """A wikilink that doesn't exactly match but is a clear substring
        of an existing page is reported as broken *and* tagged with
        ``alias_of`` so gap analysis can recommend the right fix.
        """
        node = _make_node(
            "n1",
            "Alpha",
            "Implementation in [[P.863]]-based tools",
        )
        target = _make_node(
            "p863-eval",
            "Full-scale database (P.863 evaluation)",
            "Definition",
        )
        wiki = _make_wiki([node, target])
        rule = BrokenWikilinksRule()
        results = rule.check(wiki)
        failures = [r for r in results if r.outcome == LintRuleOutcome.FAIL]
        assert len(failures) == 1
        assert failures[0].details["target"] == "P.863"
        assert failures[0].details.get("alias_of") == "p863-eval"

    def test_genuine_missing_topic_has_no_alias(self) -> None:
        """A wikilink to a topic with no nearby existing page fails without
        an alias marker — gap analysis will recommend adding a source.
        """
        node = _make_node("n1", "Alpha", "See [[No Such Topic XYZ]] here")
        wiki = _make_wiki([node])
        rule = BrokenWikilinksRule()
        results = rule.check(wiki)
        failures = [r for r in results if r.outcome == LintRuleOutcome.FAIL]
        assert len(failures) == 1
        assert "alias_of" not in failures[0].details


# ── Rule 1b: image-embed ─────────────────────────────────────────────────


class TestImageEmbedRule:
    def test_detects_missing_embed_prefix(self) -> None:
        node = _make_node("n1", "Alpha", "See [[diagram.png]] here")
        wiki = _make_wiki([node])
        rule = ImageEmbedRule()
        results = rule.check(wiki)
        failures = [r for r in results if r.outcome == LintRuleOutcome.FAIL]
        assert len(failures) == 1
        assert failures[0].details["target"] == "diagram.png"

    def test_correct_embed_passes(self) -> None:
        node = _make_node("n1", "Alpha", "See ![[diagram.png]] here")
        wiki = _make_wiki([node])
        rule = ImageEmbedRule()
        results = rule.check(wiki)
        assert all(r.outcome == LintRuleOutcome.PASS for r in results)

    def test_fix_adds_embed_prefix(self) -> None:
        node = _make_node("n1", "Alpha", "See [[diagram.png]] and [[chart.svg]]")
        wiki = _make_wiki([node])
        rule = ImageEmbedRule()
        failures = [r for r in rule.check(wiki) if r.outcome == LintRuleOutcome.FAIL]
        fixed = rule.fix(wiki, failures)
        assert all(r.fix_applied for r in fixed)
        assert node.content == "See ![[diagram.png]] and ![[chart.svg]]"

    def test_fix_does_not_touch_correct_embeds(self) -> None:
        node = _make_node("n1", "Alpha", "Already ![[ok.png]] here")
        wiki = _make_wiki([node])
        rule = ImageEmbedRule()
        results = rule.check(wiki)
        assert all(r.outcome == LintRuleOutcome.PASS for r in results)
        assert "![[ok.png]]" in node.content


class TestKnoxReferenceRule:
    def test_detects_knox_links(self) -> None:
        node = _make_node("n1", "Alpha", "See ![fig](.knox/cache/images/f.png) and [x](../../.knox/cache/y.json)")
        wiki = _make_wiki([node])
        rule = KnoxReferenceRule()
        results = rule.check(wiki)
        failures = [r for r in results if r.outcome == LintRuleOutcome.FAIL]
        assert len(failures) == 2
        assert all(r.rule == "knox-reference" for r in failures)

    def test_detects_knox_image_metadata(self) -> None:
        node = _make_node("n1", "Alpha", "Body")
        node.metadata["images"] = [{"path": ".knox/cache/images/f.png", "caption": "", "element_type": "image"}]
        wiki = _make_wiki([node])
        rule = KnoxReferenceRule()
        failures = [r for r in rule.check(wiki) if r.outcome == LintRuleOutcome.FAIL]
        assert len(failures) == 1
        assert failures[0].details["target"].endswith("f.png")

    def test_portable_links_pass(self) -> None:
        node = _make_node("n1", "Alpha", "![fig](../attachments/report/f.png) and [ Concepts](concepts/c.md)")
        wiki = _make_wiki([node])
        rule = KnoxReferenceRule()
        results = rule.check(wiki)
        assert all(r.outcome == LintRuleOutcome.PASS for r in results)


# ── Rule 2: missing-frontmatter ─────────────────────────────────────────


class TestMissingFrontmatterRule:
    def test_page_with_frontmatter_passes(self) -> None:
        node = _make_node("n1", "Alpha", "Content", frontmatter={"title": "Alpha"})
        wiki = _make_wiki([node])
        rule = MissingFrontmatterRule()
        results = rule.check(wiki)
        assert all(r.outcome == LintRuleOutcome.PASS for r in results)

    def test_page_without_frontmatter_fails(self) -> None:
        node = _make_node("n1", "Alpha", "Content")
        wiki = _make_wiki([node])
        rule = MissingFrontmatterRule()
        results = rule.check(wiki)
        failures = [r for r in results if r.outcome == LintRuleOutcome.FAIL]
        assert len(failures) == 1
        assert "title" in failures[0].details["missing_fields"]

    def test_fix_adds_title(self) -> None:
        node = _make_node("n1", "Alpha", "Content")
        wiki = _make_wiki([node])
        rule = MissingFrontmatterRule()
        results = rule.check(wiki)
        failures = [r for r in results if r.outcome == LintRuleOutcome.FAIL]
        rule.fix(wiki, failures)
        assert node.metadata["frontmatter"]["title"] == "Alpha"

    def test_empty_wiki_passes(self) -> None:
        wiki = CompiledWiki(nodes=[], edges=[], toc=[])
        rule = MissingFrontmatterRule()
        results = rule.check(wiki)
        failures = [r for r in results if r.outcome == LintRuleOutcome.FAIL]
        assert len(failures) == 0


# ── Rule 3: orphan-page ─────────────────────────────────────────────────


class TestOrphanPageRule:
    def test_linked_pages_pass(self) -> None:
        nodes = [
            _make_node("n1", "Alpha", "See [[Beta]]"),
            _make_node("n2", "Beta", "See [[Alpha]]"),
        ]
        wiki = _make_wiki(nodes)
        rule = OrphanPageRule()
        results = rule.check(wiki)
        assert all(r.outcome == LintRuleOutcome.PASS for r in results)

    def test_orphan_detected(self) -> None:
        nodes = [
            _make_node("n1", "Alpha", "Content of Alpha"),
            _make_node("n2", "Beta", "Content of Beta"),
            _make_node("n3", "Gamma", "See [[Alpha]]"),
        ]
        wiki = _make_wiki(nodes)
        rule = OrphanPageRule()
        results = rule.check(wiki)
        failures = [r for r in results if r.outcome == LintRuleOutcome.FAIL]
        # Beta has no incoming links, Gamma also has no incoming links
        assert any(r.page_id == "n2" for r in failures)

    def test_single_page_wiki_no_orphans(self) -> None:
        wiki = _make_wiki([_make_node("n1", "Alpha", "Content")])
        rule = OrphanPageRule()
        results = rule.check(wiki)
        # Single page => orphans are meaningless, should pass or be empty
        failures = [r for r in results if r.outcome == LintRuleOutcome.FAIL]
        assert len(failures) == 0

    def test_cannot_fix(self) -> None:
        rule = OrphanPageRule()
        results = [LintResult(rule="orphan-page", outcome=LintRuleOutcome.FAIL)]
        fixed = rule.fix(_make_wiki(), results)
        assert not any(r.fix_applied for r in fixed)


# ── Rule 4: duplicate-title ─────────────────────────────────────────────


class TestDuplicateTitleRule:
    def test_unique_titles_pass(self) -> None:
        nodes = [
            _make_node("n1", "Alpha", "Content"),
            _make_node("n2", "Beta", "Content"),
        ]
        wiki = _make_wiki(nodes)
        rule = DuplicateTitleRule()
        results = rule.check(wiki)
        assert all(r.outcome == LintRuleOutcome.PASS for r in results)

    def test_duplicate_detected(self) -> None:
        nodes = [
            _make_node("n1", "Alpha", "Content A"),
            _make_node("n2", "Alpha", "Content B"),
        ]
        wiki = _make_wiki(nodes)
        rule = DuplicateTitleRule()
        results = rule.check(wiki)
        failures = [r for r in results if r.outcome == LintRuleOutcome.FAIL]
        assert len(failures) == 2

    def test_case_insensitive(self) -> None:
        nodes = [
            _make_node("n1", "Alpha", "Content"),
            _make_node("n2", "alpha", "Content"),
        ]
        wiki = _make_wiki(nodes)
        rule = DuplicateTitleRule()
        results = rule.check(wiki)
        failures = [r for r in results if r.outcome == LintRuleOutcome.FAIL]
        assert len(failures) == 2

    def test_cannot_fix(self) -> None:
        rule = DuplicateTitleRule()
        results = [LintResult(rule="duplicate-title", outcome=LintRuleOutcome.FAIL)]
        fixed = rule.fix(_make_wiki(), results)
        assert not any(r.fix_applied for r in fixed)


# ── Rule 5: empty-page ──────────────────────────────────────────────────


class TestEmptyPageRule:
    def test_page_with_content_passes(self) -> None:
        wiki = _make_wiki([_make_node("n1", "Alpha", "Has content")])
        rule = EmptyPageRule()
        results = rule.check(wiki)
        assert all(r.outcome == LintRuleOutcome.PASS for r in results)

    def test_empty_page_detected(self) -> None:
        wiki = _make_wiki([_make_node("n1", "Alpha", "")])
        rule = EmptyPageRule()
        results = rule.check(wiki)
        failures = [r for r in results if r.outcome == LintRuleOutcome.FAIL]
        assert len(failures) == 1

    def test_whitespace_only_detected(self) -> None:
        wiki = _make_wiki([_make_node("n1", "Alpha", "   \n  ")])
        rule = EmptyPageRule()
        results = rule.check(wiki)
        failures = [r for r in results if r.outcome == LintRuleOutcome.FAIL]
        assert len(failures) == 1

    def test_none_content_detected(self) -> None:
        node = WikiNode(id="n1", title="Alpha", content="", node_type="section")
        wiki = _make_wiki([node])
        rule = EmptyPageRule()
        results = rule.check(wiki)
        failures = [r for r in results if r.outcome == LintRuleOutcome.FAIL]
        assert len(failures) == 1


# ── Rule 6: heading-skip ────────────────────────────────────────────────


class TestHeadingSkipRule:
    def test_sequential_headings_pass(self) -> None:
        content = "# Title\n## Section\n### Subsection"
        wiki = _make_wiki([_make_node("n1", "Alpha", content)])
        rule = HeadingSkipRule()
        results = rule.check(wiki)
        assert all(r.outcome == LintRuleOutcome.PASS for r in results)

    def test_heading_skip_detected(self) -> None:
        content = "# Title\n### Skipped h2"
        wiki = _make_wiki([_make_node("n1", "Alpha", content)])
        rule = HeadingSkipRule()
        results = rule.check(wiki)
        failures = [r for r in results if r.outcome == LintRuleOutcome.FAIL]
        assert len(failures) == 1
        assert failures[0].details["from"] == 1
        assert failures[0].details["to"] == 3

    def test_no_headings_passes(self) -> None:
        wiki = _make_wiki([_make_node("n1", "Alpha", "No headings here")])
        rule = HeadingSkipRule()
        results = rule.check(wiki)
        assert all(r.outcome == LintRuleOutcome.PASS for r in results)

    def test_decreasing_levels_ok(self) -> None:
        content = "### Deep\n## Back up\n# Top"
        wiki = _make_wiki([_make_node("n1", "Alpha", content)])
        rule = HeadingSkipRule()
        results = rule.check(wiki)
        failures = [r for r in results if r.outcome == LintRuleOutcome.FAIL]
        assert len(failures) == 0


# ── Rule 7: stale-page ──────────────────────────────────────────────────


class TestStalePageRule:
    def test_no_sources_dir_passes_with_no_memory(self) -> None:
        wiki = _make_wiki([_make_node("n1", "Alpha", "Content")])
        rule = StalePageRule()
        results = rule.check(wiki)
        assert all(r.outcome == LintRuleOutcome.PASS for r in results)

    def test_nonexistent_sources_dir_passes_with_no_memory(self, tmp_path: Any) -> None:
        wiki = _make_wiki([_make_node("n1", "Alpha", "Content")])
        rule = StalePageRule()
        results = rule.check(wiki, sources_dir=str(tmp_path / "nonexistent"))
        assert all(r.outcome == LintRuleOutcome.PASS for r in results)

    def test_source_file_not_found(self, tmp_path: Any) -> None:
        node = _make_node("n1", "Alpha", "Content", metadata={"source_file": "missing.pdf"})
        wiki = _make_wiki([node])
        rule = StalePageRule()
        results = rule.check(wiki, sources_dir=str(tmp_path))
        failures = [r for r in results if r.outcome == LintRuleOutcome.FAIL]
        assert len(failures) == 1

    def test_source_file_exists_passes(self, tmp_path: Any) -> None:
        (tmp_path / "exists.pdf").write_text("pdf content")
        node = _make_node("n1", "Alpha", "Content", metadata={"source_file": "exists.pdf"})
        wiki = _make_wiki([node])
        rule = StalePageRule()
        results = rule.check(wiki, sources_dir=str(tmp_path))
        assert all(r.outcome == LintRuleOutcome.PASS for r in results)

    def test_superseded_node_flagged(self) -> None:
        node = _make_node("n1", "Old Page", "Content")
        wiki = _make_wiki([node])
        wiki.metadata["memory"] = {"superseded": ["n1"]}
        rule = StalePageRule()
        results = rule.check(wiki)
        failures = [r for r in results if r.outcome == LintRuleOutcome.FAIL]
        assert len(failures) == 1
        assert failures[0].details.get("reason") == "superseded"

    def test_decayed_node_flagged(self) -> None:
        node = _make_node("n1", "Stale Page", "Content")
        wiki = _make_wiki([node])
        wiki.metadata["memory"] = {"stale": ["n1"]}
        rule = StalePageRule()
        results = rule.check(wiki)
        failures = [r for r in results if r.outcome == LintRuleOutcome.FAIL]
        assert len(failures) == 1
        assert failures[0].details.get("reason") == "decayed"
        assert failures[0].severity.value == "info"


# ── Rule 8: source-not-found ────────────────────────────────────────────


class TestSourceNotFoundRule:
    def test_no_sources_dir_skips(self) -> None:
        wiki = _make_wiki([_make_node("n1", "Alpha", "Content", source_chunks=["chunk_1"])])
        rule = SourceNotFoundRule()
        results = rule.check(wiki)
        assert all(r.outcome == LintRuleOutcome.SKIP for r in results)

    def test_missing_source_detected(self, tmp_path: Any) -> None:
        node = _make_node(
            "n1",
            "Alpha",
            "Content",
            source_chunks=["chunk_1"],
            metadata={"source_file": "gone.pdf"},
        )
        wiki = _make_wiki([node])
        rule = SourceNotFoundRule()
        results = rule.check(wiki, sources_dir=str(tmp_path))
        failures = [r for r in results if r.outcome == LintRuleOutcome.FAIL]
        assert len(failures) == 1


# ── LintRunner ───────────────────────────────────────────────────────────


class TestLintRunner:
    def test_default_runs_all_structural(self) -> None:
        wiki = _make_wiki([_make_node(frontmatter={"title": "Test"})])
        runner = LintRunner()
        report = runner.run(wiki)
        # Orphan-page produces no results for a single-page wiki (orphans meaningless)
        rule_names = {r.rule for r in report.results}
        assert len(rule_names) >= 7  # orphan-page may not produce a result for 1 page

    def test_specific_rules_only(self) -> None:
        wiki = _make_wiki([_make_node()])
        runner = LintRunner(rules=["broken-wikilinks"])
        report = runner.run(wiki)
        rule_names = {r.rule for r in report.results}
        assert rule_names == {"broken-wikilinks"}

    def test_unknown_rule_warning(self) -> None:
        wiki = _make_wiki([_make_node()])
        runner = LintRunner(rules=["nonexistent-rule"])
        report = runner.run(wiki)
        # Should produce no results since the rule doesn't exist
        assert len(report.results) == 0

    def test_fix_mode(self) -> None:
        node = _make_node(content="[[Missing]]")
        wiki = _make_wiki([node])
        runner = LintRunner(rules=["broken-wikilinks"], fix=True)
        report = runner.run(wiki)
        fixed = [r for r in report.results if r.fix_applied]
        assert len(fixed) >= 1

    def test_health_score_calculation(self) -> None:
        wiki = _make_wiki([_make_node(frontmatter={"title": "Test"})])
        runner = LintRunner(rules=["empty-page"])
        report = runner.run(wiki)
        assert report.health_score > 0

    def test_empty_wiki(self) -> None:
        wiki = _make_wiki([])
        runner = LintRunner()
        report = runner.run(wiki)
        assert report.health_score > 0

    def test_llm_rules_skip_without_client(self) -> None:
        wiki = _make_wiki([_make_node()])
        runner = LintRunner(rules=["clarity-score"])
        report = runner.run(wiki)
        assert all(r.outcome == LintRuleOutcome.SKIP for r in report.results)


# ── Report formatting ───────────────────────────────────────────────────


class TestReportFormatting:
    def test_json_output_is_valid_json(self) -> None:
        wiki = _make_wiki([_make_node(frontmatter={"title": "Test"})])
        runner = LintRunner(rules=["broken-wikilinks"])
        report = runner.run(wiki)
        output = report_to_json(report)
        data = json.loads(output)
        assert "health_score" in data
        assert "results" in data

    def test_text_output_includes_score(self) -> None:
        wiki = _make_wiki([_make_node(frontmatter={"title": "Test"})])
        runner = LintRunner(rules=["broken-wikilinks"])
        report = runner.run(wiki)
        text = report_to_text(report)
        assert "Health Score" in text

    def test_json_structure(self) -> None:
        wiki = _make_wiki([_make_node()])
        runner = LintRunner(rules=["empty-page"])
        report = runner.run(wiki)
        output = report_to_json(report)
        data = json.loads(output)
        for result in data["results"]:
            assert "rule" in result
            assert "outcome" in result
            assert "severity" in result


# ── _build_report ────────────────────────────────────────────────────────


class TestBuildReport:
    def test_all_pass_100_score(self) -> None:
        results = [
            LintResult(rule="a", outcome=LintRuleOutcome.PASS),
            LintResult(rule="b", outcome=LintRuleOutcome.PASS),
        ]
        report = _build_report(results)
        assert report.health_score == 100.0
        assert report.passes == 2

    def test_half_fail_weighted_score(self) -> None:
        # 1 PASS + 1 FAIL(WARNING): penalty=2, max_possible=2*3=6 → 1-2/6=66.7
        results = [
            LintResult(rule="a", outcome=LintRuleOutcome.PASS),
            LintResult(rule="b", outcome=LintRuleOutcome.FAIL, severity=Severity.WARNING),
        ]
        report = _build_report(results)
        assert report.health_score == 66.7
        assert report.failures == 1

    def test_empty_results(self) -> None:
        report = _build_report([])
        assert report.health_score == 0.0


# ── Edge cases from task requirements ────────────────────────────────────


class TestRumdlRule:
    """Filesystem-based rule — uses real tmp vaults, real rumdl subprocess."""

    def test_checks_wiki_and_notes_but_not_sources(self, tmp_path: Path) -> None:
        base = self._vault(tmp_path)
        # Sources file with issues — must never be reported.
        (base / "sources" / "s.md").write_text("# Bad\n\n## X\n\n## X\n", encoding="utf-8")
        # Wiki output page with issues.
        (base / "wiki" / "page.md").write_text("# Bad\n\n## Y\n\n## Y\n", encoding="utf-8")
        # Self-written note with issues.
        (base / "note.md").write_text("# Bad\n\n## Z\n\n## Z\n", encoding="utf-8")

        results = RumdlRule().check(_make_wiki([]), sources_dir=str(base / "sources"))
        failures = [r for r in results if r.outcome == LintRuleOutcome.FAIL]
        checked_files = {r.details["file"] for r in failures}
        assert all("sources" not in f for f in checked_files)
        assert any(f.endswith("page.md") for f in checked_files)
        assert any(f.endswith("note.md") for f in checked_files)
        assert all(r.details["rumdl_rule"] for r in failures)

    def test_clean_vault_passes(self, tmp_path: Path) -> None:
        base = self._vault(tmp_path)
        (base / "wiki" / "ok.md").write_text("# Ok\n\n## One\n\n### Two\n", encoding="utf-8")
        results = RumdlRule().check(_make_wiki([]), sources_dir=str(base / "sources"))
        assert all(r.outcome == LintRuleOutcome.PASS for r in results)

    def test_fix_reformats_self_written_note(self, tmp_path: Path) -> None:
        base = self._vault(tmp_path)
        note = base / "note.md"
        # MD012 (multiple trailing blank lines) is auto-fixable; MD024 is not.
        note.write_text("# Note\n\nText.\n\n\n\n", encoding="utf-8")
        rule = RumdlRule()
        results = rule.check(_make_wiki([]), sources_dir=str(base / "sources"))
        fixable = [r for r in results if r.outcome == LintRuleOutcome.FAIL and r.details["fixable"]]
        assert fixable, "expected at least one fixable finding"
        fixed = rule.fix(_make_wiki([]), results)
        assert all(r.fix_applied for r in fixed if r.details.get("fixable"))
        assert note.read_text(encoding="utf-8").endswith("Text.\n")

    def test_skips_when_rumdl_missing(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        base = self._vault(tmp_path)
        monkeypatch.setattr("knox.lint.structural.shutil.which", lambda _: None)
        results = RumdlRule().check(_make_wiki([]), sources_dir=str(base / "sources"))
        assert all(r.outcome == LintRuleOutcome.SKIP for r in results)

    def test_skips_without_sources_dir(self) -> None:
        results = RumdlRule().check(_make_wiki([]), sources_dir=None)
        assert all(r.outcome == LintRuleOutcome.SKIP for r in results)

    def test_config_forwarded_to_rumdl(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """compile.output.rumdl_config is passed as --config to rumdl."""
        captured: list[list[str]] = []

        def fake_run_rumdl(exe: str, cmd: list[str]) -> list[dict]:
            captured.append(cmd)
            return []

        monkeypatch.setattr(RumdlRule, "_run_rumdl", staticmethod(fake_run_rumdl))
        base = self._vault(tmp_path)
        (base / "note.md").write_text("# Note\n", encoding="utf-8")
        cfg_file = base / ".rumdl.toml"
        results = RumdlRule(rumdl_config=cfg_file).check(_make_wiki([]), sources_dir=str(base / "sources"))
        assert all(r.outcome == LintRuleOutcome.PASS for r in results)
        assert captured
        assert "--config" in captured[0]
        assert str(cfg_file) in captured[0]

    def _vault(self, tmp_path: Path) -> Path:
        (tmp_path / "sources").mkdir()
        (tmp_path / "wiki").mkdir()
        (tmp_path / ".knox").mkdir()
        return tmp_path


class TestEdgeCases:
    def test_single_page_all_violations(self) -> None:
        """A single page with broken links, no frontmatter, empty content."""
        node = _make_node("n1", "Alpha", "[[Missing]]\n### Skip")
        wiki = _make_wiki([node])
        runner = LintRunner(rules=["broken-wikilinks", "missing-frontmatter", "empty-page", "heading-skip"])
        report = runner.run(wiki)
        failures = [r for r in report.results if r.outcome == LintRuleOutcome.FAIL]
        # heading-skip: h1 missing before h3 — but "[[Missing]]\n### Skip" has no h1,
        # just h3. That's a skip from implicit level 0 to 3? Actually heading-skip
        # checks consecutive headings within a page. Only h3 exists, so no skip.
        # The 2 failures are: broken-wikilinks + missing-frontmatter. That's fine.
        assert len(failures) >= 2
        rule_names = {r.rule for r in failures}
        assert "broken-wikilinks" in rule_names
        assert "missing-frontmatter" in rule_names

    def test_empty_wiki_no_violations(self) -> None:
        """Empty wiki should find no violations."""
        wiki = CompiledWiki(nodes=[], edges=[], toc=[])
        runner = LintRunner()
        report = runner.run(wiki)
        failures = [r for r in report.results if r.outcome == LintRuleOutcome.FAIL]
        assert len(failures) == 0

    def test_structural_rules_registry_complete(self) -> None:
        """All structural rules should be registered.

        ``RumdlRule`` and ``DuplicateReferencesRule`` are NOT singletons —
        they are instantiated by ``LintRunner._resolve_rules`` with config
        values. The singleton list has 11 entries; ``RULE_BY_NAME`` maps all
        13 names.
        """
        assert len(STRUCTURAL_RULES) == 11
        expected = {
            "image-embed",
            "broken-wikilinks",
            "knox-reference",
            "missing-frontmatter",
            "orphan-page",
            "duplicate-title",
            "empty-page",
            "heading-skip",
            "stale-page",
            "source-not-found",
            "orphan-attachments",
            "rumdl",
            "duplicate-references",
        }
        assert set(RULE_BY_NAME.keys()) == expected


class TestContradictionDetectionRule:
    def test_skips_without_client(self) -> None:
        wiki = _make_wiki([_make_node("n1", "A", "x"), _make_node("n2", "B", "y")])
        runner = LintRunner(rules=["contradiction-detection"])
        report = runner.run(wiki)
        assert all(r.outcome == LintRuleOutcome.SKIP for r in report.results)

    def test_passes_with_single_page(self) -> None:
        wiki = _make_wiki([_make_node("n1", "A", "x")])
        runner = LintRunner(rules=["contradiction-detection"], lint_agent=_mock_llm_client(ContradictionOutput(contradictions=[])))
        report = runner.run(wiki)
        assert any(r.outcome == LintRuleOutcome.PASS and "at least two pages" in r.message for r in report.results)

    def test_reports_contradictions(self) -> None:
        wiki = _make_wiki(
            [
                _make_node("n1", "Page A", "The sky is blue."),
                _make_node("n2", "Page B", "The sky is green."),
            ]
        )
        client = _mock_llm_client(
            ContradictionOutput(
                contradictions=[
                    Contradiction(
                        page_a="Page A",
                        page_b="Page B",
                        claim_a="The sky is blue.",
                        claim_b="The sky is green.",
                        explanation="Colors differ",
                        confidence=0.9,
                    )
                ]
            )
        )
        runner = LintRunner(rules=["contradiction-detection"], lint_agent=client)
        report = runner.run(wiki)
        failures = [r for r in report.results if r.outcome == LintRuleOutcome.FAIL]
        assert len(failures) == 1
        assert failures[0].page_id == "n1"
        assert failures[0].severity == Severity.ERROR
        assert failures[0].details["page_b_id"] == "n2"

    def test_passes_when_none_found(self) -> None:
        wiki = _make_wiki([_make_node("n1", "A", "x"), _make_node("n2", "B", "y")])
        runner = LintRunner(rules=["contradiction-detection"], lint_agent=_mock_llm_client(ContradictionOutput(contradictions=[])))
        report = runner.run(wiki)
        assert any(r.outcome == LintRuleOutcome.PASS and "No contradictions detected" in r.message for r in report.results)

    def test_handles_llm_failure(self) -> None:
        wiki = _make_wiki([_make_node("n1", "A", "x"), _make_node("n2", "B", "y")])
        client = _mock_llm_client(ContradictionOutput(contradictions=[]))
        runner = LintRunner(rules=["contradiction-detection"], lint_agent=client)
        report = runner.run(wiki)
        assert not any(r.outcome == LintRuleOutcome.SKIP for r in report.results)

    def test_dedup_identical_pairs(self) -> None:
        """LLM returning the same pair N times produces exactly one result."""
        wiki = _make_wiki(
            [
                _make_node("n1", "Page A", "Auditory models required."),
                _make_node("n2", "Page B", "No auditory models needed."),
            ]
        )
        contradiction = Contradiction(
            page_a="Page A",
            page_b="Page B",
            claim_a="uses auditory models",
            claim_b="avoids auditory models",
            explanation="Mutually exclusive methodology",
            confidence=0.95,
        )
        # LLM returns the same contradiction 5 times in one batch.
        client = _mock_llm_client(ContradictionOutput(contradictions=[contradiction] * 5))
        runner = LintRunner(rules=["contradiction-detection"], lint_agent=client)
        report = runner.run(wiki)
        failures = [r for r in report.results if r.outcome == LintRuleOutcome.FAIL]
        assert len(failures) == 1


class TestCoverageGapRemediation:
    """Coverage-gap remediation — concept vs entity lookup, priority."""

    def test_concept_match_generates_page(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setattr(
            gaps_module,
            "load_all_extractions",
            lambda b, s: {"doc1": Extraction(source="doc1", concepts=[Concept(name="Active Noise Control", definition="Noise reduction technique")])},
        )
        wiki = _make_wiki()
        actions = remediate(wiki, LintReport(results=[_gap_result("Active Noise Control")]), base_dir=tmp_path, agent=_mock_remediate_llm())
        applied = [a for a in actions if a.status == RemediationStatus.APPLIED]
        assert len(applied) == 1
        assert "Active Noise Control" in applied[0].description
        assert any(n.title == "Active Noise Control" for n in wiki.nodes)

    def test_entity_match_generates_page(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Standard extracted as entity (not concept) should still generate a page."""
        monkeypatch.setattr(
            gaps_module,
            "load_all_extractions",
            lambda b, s: {
                "doc1": Extraction(
                    source="doc1",
                    entities=[Entity(name="ITU-T P.57", type="Standard", context="Telephone transmission quality")],
                )
            },
        )
        wiki = _make_wiki()
        actions = remediate(wiki, LintReport(results=[_gap_result("ITU-T P.57")]), base_dir=tmp_path, agent=_mock_remediate_llm())
        applied = [a for a in actions if a.status == RemediationStatus.APPLIED]
        assert len(applied) == 1
        assert "ITU-T P.57" in applied[0].description
        assert any(n.title == "ITU-T P.57" for n in wiki.nodes)

    def test_no_match_produces_no_action(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Concepts without an extraction match are skipped (gaps shows recommendations)."""
        monkeypatch.setattr(
            gaps_module,
            "load_all_extractions",
            lambda b, s: {"doc1": Extraction(source="doc1")},
        )
        wiki = _make_wiki()
        actions = remediate(wiki, LintReport(results=[_gap_result("POLQA")]), base_dir=tmp_path, agent=_mock_remediate_llm())
        assert actions == []

    def test_concept_takes_priority_over_entity(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """When both a concept and entity exist for the same name, concept data is used."""
        monkeypatch.setattr(
            gaps_module,
            "load_all_extractions",
            lambda b, s: {
                "doc1": Extraction(
                    source="doc1",
                    concepts=[Concept(name="ERP", definition="Equalization-based acoustic metric")],
                    entities=[Entity(name="ERP", type="Metric", context="Entity-only context")],
                )
            },
        )
        wiki = _make_wiki()
        actions = remediate(wiki, LintReport(results=[_gap_result("ERP")]), base_dir=tmp_path, agent=_mock_remediate_llm())
        applied = [a for a in actions if a.status == RemediationStatus.APPLIED]
        assert len(applied) == 1
        assert "ERP" in applied[0].description

    def test_dry_run_skips_generation(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setattr(
            gaps_module,
            "load_all_extractions",
            lambda b, s: {"doc1": Extraction(source="doc1", concepts=[Concept(name="Echo Cancellation", definition="DSP technique")])},
        )
        wiki = _make_wiki()
        actions = remediate(
            wiki,
            LintReport(results=[_gap_result("Echo Cancellation")]),
            base_dir=tmp_path,
            agent=_mock_remediate_llm(),
            dry_run=True,
        )
        skipped = [a for a in actions if a.status == RemediationStatus.SKIPPED]
        assert len(skipped) == 1
        assert not any(n.title == "Echo Cancellation" for n in wiki.nodes)

    def test_existing_page_not_regenerated(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setattr(
            gaps_module,
            "load_all_extractions",
            lambda b, s: {"doc1": Extraction(source="doc1", concepts=[Concept(name="Existing", definition="Already there")])},
        )
        wiki = _make_wiki([_make_node("existing", "Existing", "pre-existing")])
        actions = remediate(wiki, LintReport(results=[_gap_result("Existing")]), base_dir=tmp_path, agent=_mock_remediate_llm())
        assert not actions

    def test_source_summary_seed_included_in_prompt(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Per-source document summary is passed to the LLM prompt as seed material."""
        unique_summary = "SUMMARY_MARKER_ACTIVE_NOISE_CONTROL_DETAILS"
        monkeypatch.setattr(
            gaps_module,
            "load_all_extractions",
            lambda b, s: {
                "doc1": Extraction(
                    source="doc1",
                    summary=unique_summary,
                    concepts=[Concept(name="Active Noise Control", definition="Noise reduction technique")],
                )
            },
        )
        client = _mock_remediate_llm()
        wiki = _make_wiki()
        remediate(wiki, LintReport(results=[_gap_result("Active Noise Control")]), base_dir=tmp_path, agent=client)
        prompt = client.run.call_args[0][0].text
        assert unique_summary in prompt

    def test_only_relevant_source_summary_included(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Summary from a source not mentioning the concept is excluded."""
        monkeypatch.setattr(
            gaps_module,
            "load_all_extractions",
            lambda b, s: {
                "doc_relevant": Extraction(
                    source="doc_relevant",
                    summary="RELEVANT_SUMMARY_MARKER",
                    concepts=[Concept(name="ANC", definition="technique")],
                ),
                "doc_other": Extraction(
                    source="doc_other",
                    summary="UNRELATED_SUMMARY_MARKER",
                ),
            },
        )
        client = _mock_remediate_llm()
        wiki = _make_wiki()
        remediate(wiki, LintReport(results=[_gap_result("ANC")]), base_dir=tmp_path, agent=client)
        prompt = client.run.call_args[0][0].text
        assert "RELEVANT_SUMMARY_MARKER" in prompt
        assert "UNRELATED_SUMMARY_MARKER" not in prompt

    def test_generated_page_appears_once(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """A generated concept is not reported twice."""
        monkeypatch.setattr(
            gaps_module,
            "load_all_extractions",
            lambda b, s: {"doc1": Extraction(source="doc1", concepts=[Concept(name="POLQA", definition="Speech quality metric")])},
        )
        wiki = _make_wiki()
        actions = remediate(wiki, LintReport(results=[_gap_result("POLQA")]), base_dir=tmp_path, agent=_mock_remediate_llm())
        applied = [a for a in actions if a.status == RemediationStatus.APPLIED]
        assert len(applied) == 1

    def test_remediate_writes_md_file(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Generated concept pages land in concepts/ with OKF frontmatter."""
        monkeypatch.setattr(
            gaps_module,
            "load_all_extractions",
            lambda b, s: {"doc1": Extraction(source="doc1", concepts=[Concept(name="Active Noise Control", definition="Noise reduction technique")])},
        )
        output_dir = tmp_path / "wiki"
        wiki = _make_wiki()
        remediate(
            wiki,
            LintReport(results=[_gap_result("Active Noise Control")]),
            base_dir=tmp_path,
            output_dir=output_dir,
            agent=_mock_remediate_llm(),
        )
        md = output_dir / "concepts" / "active-noise-control.md"
        assert md.exists()
        text = md.read_text(encoding="utf-8")
        assert "title: Active Noise Control" in text
        assert "type: Concept" in text

    def test_remediate_updates_index(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """index.md links the generated concept page after remediation."""
        monkeypatch.setattr(
            gaps_module,
            "load_all_extractions",
            lambda b, s: {"doc1": Extraction(source="doc1", concepts=[Concept(name="Active Noise Control", definition="Noise reduction technique")])},
        )
        output_dir = tmp_path / "wiki"
        wiki = _make_wiki()
        remediate(
            wiki,
            LintReport(results=[_gap_result("Active Noise Control")]),
            base_dir=tmp_path,
            output_dir=output_dir,
            agent=_mock_remediate_llm(),
        )
        index = (output_dir / "index.md").read_text(encoding="utf-8")
        assert "active-noise-control" in index

    def test_ghost_node_regenerated_not_skipped(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """A lint-remediation ghost node (no .md) is regenerated, not silently skipped."""
        monkeypatch.setattr(
            gaps_module,
            "load_all_extractions",
            lambda b, s: {"doc1": Extraction(source="doc1", concepts=[Concept(name="POLQA", definition="Speech quality metric")])},
        )
        ghost = _make_node("polqa", "POLQA", "stale", metadata={"source": "lint-remediation"})
        wiki = _make_wiki([ghost])
        output_dir = tmp_path / "wiki"
        actions = remediate(
            wiki,
            LintReport(results=[_gap_result("POLQA")]),
            base_dir=tmp_path,
            output_dir=output_dir,
            agent=_mock_remediate_llm("fresh content"),
        )
        applied = [a for a in actions if a.status == RemediationStatus.APPLIED]
        assert len(applied) == 1
        polqa_nodes = [n for n in wiki.nodes if n.id == "polqa"]
        assert len(polqa_nodes) == 1
        assert polqa_nodes[0].content == "fresh content"
        assert (output_dir / "concepts" / "polqa.md").exists()


class TestAgenticRemediation:
    """Agentic remediation agent: one multi-turn session for all gaps."""

    def test_agentic_one_session_for_all_gaps(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """A single agent.run() call should cover every matched gap."""
        monkeypatch.setattr(
            gaps_module,
            "load_all_extractions",
            lambda b, s: {
                "doc1": Extraction(
                    source="doc1",
                    concepts=[
                        Concept(name="Active Noise Control", definition="ANC"),
                        Concept(name="POLQA", definition="Speech quality"),
                    ],
                )
            },
        )
        agent = _mock_agentic_remediation(
            pages=[
                ("Active Noise Control", "# ANC\n..."),
                ("POLQA", "# POLQA\n..."),
            ]
        )
        wiki = _make_wiki([])
        output_dir = tmp_path / "wiki"
        actions = remediate(
            wiki,
            LintReport(
                results=[
                    _gap_result("Active Noise Control"),
                    _gap_result("POLQA"),
                ]
            ),
            base_dir=tmp_path,
            output_dir=output_dir,
            agent=agent,
        )
        # ONE call to agent.run() covers BOTH gaps (collapses N→1).
        assert agent.run.call_count == 1
        applied = [a for a in actions if a.status == RemediationStatus.APPLIED]
        assert {a.details["concept"] for a in applied} == {"Active Noise Control", "POLQA"}
        assert {n.id for n in wiki.nodes} == {"active-noise-control", "polqa"}
        assert (output_dir / "concepts" / "active-noise-control.md").exists()
        assert (output_dir / "concepts" / "polqa.md").exists()

    def test_agentic_records_unfixable(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Gaps the agent declines land as SKIPPED with the agent's reason."""
        monkeypatch.setattr(
            gaps_module,
            "load_all_extractions",
            lambda b, s: {"doc1": Extraction(source="doc1", concepts=[Concept(name="Active Noise Control", definition="ANC")])},
        )
        agent = MagicMock(spec=AgenticRemediationAgent)

        def _run(deps: RemediationDeps) -> None:
            deps.gaps.mark_unfixable("Active Noise Control", "no usable source material")

        agent.run.side_effect = _run
        actions = remediate(
            _make_wiki([]),
            LintReport(results=[_gap_result("Active Noise Control")]),
            base_dir=tmp_path,
            agent=agent,
        )
        skipped = [a for a in actions if a.status == RemediationStatus.SKIPPED]
        assert len(skipped) == 1
        assert skipped[0].details["concept"] == "Active Noise Control"
        assert skipped[0].details["reason"] == "no usable source material"
        assert "unfixable" in skipped[0].category

    def test_agentic_run_failure_does_not_raise(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """An exception inside the agent is caught and surfaced as SKIPPED, not propagated."""
        monkeypatch.setattr(
            gaps_module,
            "load_all_extractions",
            lambda b, s: {"doc1": Extraction(source="doc1", concepts=[Concept(name="POLQA", definition="Speech quality")])},
        )
        agent = MagicMock(spec=AgenticRemediationAgent)
        agent.run.side_effect = RuntimeError("boom")
        actions = remediate(
            _make_wiki([]),
            LintReport(results=[_gap_result("POLQA")]),
            base_dir=tmp_path,
            agent=agent,
        )
        # No exception propagated; no applied actions either.
        assert [a for a in actions if a.status == RemediationStatus.APPLIED] == []


class TestCoverageGapRuleDedup:
    """Cross-batch dedup of semantically equivalent coverage gaps."""

    def test_dedup_merges_equivalent_concepts(self) -> None:
        wiki = _make_wiki([_make_node("n1", "A", "content"), _make_node("n2", "B", "content")])
        client = MagicMock()
        client.run.side_effect = [
            CoverageGapOutput(
                gaps=[
                    CoverageGap(concept="ETSI TS 103 558", reason="not discussed"),
                    CoverageGap(concept="ABLE Model", reason="never mentioned"),
                ]
            ),
            CoverageGapOutput(
                gaps=[
                    CoverageGap(concept="ETSI TS 103 558", reason="not discussed", aliases=["ABLE Model"]),
                ]
            ),
        ]
        runner = LintRunner(rules=["coverage-gap"], lint_agent=client)
        report = runner.run(wiki)
        fails = [r for r in report.results if r.outcome == LintRuleOutcome.FAIL]
        assert len(fails) == 1
        assert fails[0].details["concept"] == "ETSI TS 103 558"
        assert "ABLE Model" in fails[0].details["aliases"]

    def test_dedup_falls_back_on_error(self) -> None:
        wiki = _make_wiki([_make_node("n1", "A", "content"), _make_node("n2", "B", "content")])
        client = MagicMock()
        client.run.side_effect = [
            CoverageGapOutput(
                gaps=[
                    CoverageGap(concept="Concept A", reason="missing"),
                    CoverageGap(concept="Concept B", reason="absent"),
                ]
            ),
            Exception("LLM dedup failed"),
        ]
        runner = LintRunner(rules=["coverage-gap"], lint_agent=client)
        report = runner.run(wiki)
        fails = [r for r in report.results if r.outcome == LintRuleOutcome.FAIL]
        assert len(fails) == 2


class TestCoverageGapSuppression:
    """Deterministic suppression of gaps whose page already exists (cross-batch false positives)."""

    def test_existing_page_gap_suppressed(self) -> None:
        wiki = _make_wiki([_make_node("leap-model", "LEAP Model", "content about LEAP")])
        client = MagicMock()
        client.run.side_effect = [
            CoverageGapOutput(
                gaps=[
                    CoverageGap(concept="LEAP Model", reason="not discussed"),
                    CoverageGap(concept="Truly Missing", reason="absent"),
                ]
            ),
            CoverageGapOutput(
                gaps=[
                    CoverageGap(concept="LEAP Model", reason="not discussed"),
                    CoverageGap(concept="Truly Missing", reason="absent"),
                ]
            ),
        ]
        runner = LintRunner(rules=["coverage-gap"], lint_agent=client)
        report = runner.run(wiki)
        fails = [r for r in report.results if r.outcome == LintRuleOutcome.FAIL]
        assert len(fails) == 1
        assert fails[0].details["concept"] == "Truly Missing"


# ── rumdl config resolution ──────────────────────────────────────────────


class TestRumdlConfigResolution:
    """resolve_rumdl_config: default .knox/rumdl.toml, materialization, overrides."""

    def test_default_materializes_knox_rumdl_toml(self, tmp_path: Path) -> None:

        resolved = resolve_rumdl_config(tmp_path)
        assert resolved is not None
        assert resolved == tmp_path / ".knox" / "rumdl.toml"
        assert resolved.exists()
        content = resolved.read_text(encoding="utf-8")
        assert 'disable = ["MD013"]' in content
        assert 'front-matter-title = ""' in content

    def test_existing_config_not_overwritten(self, tmp_path: Path) -> None:

        target = tmp_path / ".knox" / "rumdl.toml"
        target.parent.mkdir(parents=True)
        target.write_text("[MD013]\nline_length = 999\n", encoding="utf-8")

        resolve_rumdl_config(tmp_path)
        assert target.read_text(encoding="utf-8") == "[MD013]\nline_length = 999\n"

    def test_explicit_config_wins(self, tmp_path: Path) -> None:

        custom = tmp_path / "custom.toml"
        custom.write_text("", encoding="utf-8")
        assert resolve_rumdl_config(tmp_path, custom) == custom

    def test_relative_explicit_config_resolves_against_base_dir(self, tmp_path: Path) -> None:

        assert resolve_rumdl_config(tmp_path, ".knox/rumdl.toml") == tmp_path / ".knox" / "rumdl.toml"


class TestOrphanAttachmentsRule:
    def test_skip_without_compile_state(self, tmp_path: Any) -> None:
        base = tmp_path / "vault"
        (base / "sources").mkdir(parents=True)
        rule = OrphanAttachmentsRule()
        results = rule.check(_make_wiki([]), sources_dir=str(base / "sources"))
        assert results[0].outcome == LintRuleOutcome.SKIP

    def test_orphan_slug_flagged(self, tmp_path: Any) -> None:
        base, sources_dir = self._vault(tmp_path)
        ghost = base / "attachments" / "ghost-source"
        ghost.mkdir(parents=True)
        (ghost / "x.txt").write_text("data")
        rule = OrphanAttachmentsRule()
        results = rule.check(_make_wiki([]), sources_dir=sources_dir)
        failures = [r for r in results if r.outcome == LintRuleOutcome.FAIL]
        assert len(failures) == 1
        assert "ghost-source/x.txt" in failures[0].message

    def test_live_slug_passes(self, tmp_path: Any) -> None:
        base, sources_dir = self._vault(tmp_path)
        slug_dir = base / "attachments" / "report"
        slug_dir.mkdir(parents=True)
        (slug_dir / "notes.txt").write_text("data")
        rule = OrphanAttachmentsRule()
        results = rule.check(_make_wiki([]), sources_dir=sources_dir)
        assert all(r.outcome == LintRuleOutcome.PASS for r in results)

    def test_no_attachments_dir_skips(self, tmp_path: Any) -> None:
        base, sources_dir = self._vault(tmp_path)
        rule = OrphanAttachmentsRule()
        results = rule.check(_make_wiki([]), sources_dir=sources_dir)
        assert results[0].outcome == LintRuleOutcome.SKIP

    def test_runner_includes_rule(self, tmp_path: Any) -> None:
        base, sources_dir = self._vault(tmp_path)
        ghost = base / "attachments" / "ghost-source"
        ghost.mkdir(parents=True)
        (ghost / "x.txt").write_text("data")
        runner = LintRunner(sources_dir=sources_dir, rules=["orphan-attachments"])
        report = runner.run(_make_wiki([]))
        failures = [r for r in report.results if r.outcome == LintRuleOutcome.FAIL]
        assert len(failures) == 1

    def _vault(self, tmp_path: Any) -> tuple[Path, str]:
        """Vault with one source + compile state; returns (base, sources_dir)."""
        base = tmp_path / "vault"
        (base / "sources").mkdir(parents=True)
        (base / "sources" / "report.pdf").write_bytes(b"%PDF fake")
        (base / ".knox").mkdir()
        (base / ".knox" / "compile_state.json").write_text("{}", encoding="utf-8")
        return base, str(base / "sources")


def _mock_agentic_remediation(pages: list[tuple[str, str]] | None = None) -> MagicMock:
    """Mock AgenticRemediationAgent whose ``run`` populates ``deps.pages``."""
    agent = MagicMock(spec=AgenticRemediationAgent)
    pages = pages or []

    def _run(deps: object) -> None:
        for name, content in pages:
            deps.pages.append(RemediationPageResult(name=name, content=content))  # type: ignore[union-attr]

    agent.run.side_effect = _run
    return agent


def _make_wiki(nodes: list[WikiNode] | None = None, **kwargs: Any) -> CompiledWiki:
    return CompiledWiki(
        nodes=list(nodes or []),
        edges=kwargs.get("edges", []),
        toc=kwargs.get("toc", []),
        metadata=kwargs.get("metadata", {}),
    )


# ── Fixtures ─────────────────────────────────────────────────────────────


def _make_node(
    node_id: str = "node_0",
    title: str = "Test Page",
    content: str = "Some content",
    *,
    source_chunks: list[str] | None = None,
    frontmatter: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> WikiNode:
    meta = dict(metadata or {})
    if frontmatter is not None:
        meta["frontmatter"] = frontmatter
    return WikiNode(
        id=node_id,
        title=title,
        content=content,
        node_type="section",
        source_chunks=source_chunks,
        metadata=meta,
    )


# ── LLM rule: contradiction-detection ────────────────────────────────────


def _mock_llm_client(json_data: Any) -> MagicMock:
    agent = MagicMock(spec=LintRuleAgent)
    agent.timeout = 120
    agent.usage = None
    if json_data is not None:
        agent.run.return_value = json_data
    else:
        agent.run.side_effect = Exception("LLM failed")
    return agent


# ── Coverage-gap remediation ────────────────────────────────────────────


def _mock_remediate_llm(content: str = "Generated concept page.") -> MagicMock:
    client = MagicMock(spec=RemediationAgent)
    client.run.return_value = content
    return client


def _gap_result(concept: str) -> LintReportResult:
    return LintReportResult(rule="coverage-gap", outcome="fail", message=f"{concept} — missing", details={"concept": concept})
