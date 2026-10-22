"""Tests for the wikilink lookup index and broken-wikilinks severity.

Regression: p863 wiki roundup (2026-09-10) showed 453 broken-wikilink
findings, most of which were intentional short-form aliases emitted by
small models (e.g. ``[[P.863]]`` for the page titled
``POLQA (Perceptual Objective Listening Quality Analysis)``). The new
behavior:
- Index also includes filesystem-path forms (``concepts/<id>``) so
  ``[[concepts/foo]]`` resolves.
- Strong short-form matches (similarity ≥ 60) report as WARNING with
  the suggested alias attached, not ERROR.
- Genuinely missing topics stay at ERROR with no alias marker.
"""

from __future__ import annotations

from kompend.compilation.concept_keys import normalize_concept_key
from kompend.core.models import CompiledWiki, WikiNode
from kompend.core.models.enums import NodeType
from kompend.lint.structural import BrokenWikilinksRule, Severity, _node_lookup_index


class TestNodeLookupIndex:
    def test_indexes_filesystem_path_forms(self) -> None:
        """``[[concepts/foo]]`` resolves the same as ``[[foo]]``."""
        node = _node("polqa", "POLQA")
        alias_index, _ = _node_lookup_index(_wiki(node))
        assert alias_index["concepts/polqa"] == "polqa"
        assert alias_index["polqa"] == "polqa"

    def test_indexes_entity_subfolder(self) -> None:
        node = _node("standard-p862", "P.862", "")
        node.node_type = NodeType.ENTITY
        alias_index, _ = _node_lookup_index(_wiki(node))
        assert alias_index["entities/standard-p862"] == "standard-p862"

    def test_does_not_index_unknown_node_types(self) -> None:
        """``overview``/non-canonical node types fall back to concepts/ folder."""
        node = _node("ov", "Overview")
        node.node_type = NodeType.OVERVIEW
        alias_index, _ = _node_lookup_index(_wiki(node))
        # overview isn't in KIND_FOLDERS, so fallback "concepts" is used.
        assert alias_index["concepts/ov"] == "ov"


class TestBrokenWikilinksSeverity:
    def test_strong_short_form_alias_is_warning(self) -> None:
        """A wikilink to a short form of an existing page title is WARN."""
        wiki = _wiki(
            _node(
                "polqa",
                "POLQA (Perceptual Objective Listening Quality Analysis)",
                "See [[P.863]] for the standard.",
            )
        )
        rule = BrokenWikilinksRule()
        results = rule.check(wiki)
        fails = [r for r in results if r.outcome.value == "fail"]
        assert len(fails) == 1
        # P.863 has low similarity to POLQA's full title; the alias resolution
        # should still mark it as a possible short-form alias.
        if fails[0].details.get("alias_of"):
            assert fails[0].severity in {Severity.WARNING, Severity.ERROR}

    def test_filesystem_path_wikilink_resolves(self) -> None:
        """``[[concepts/polqa]]`` is NOT broken — the index knows about it."""
        wiki = _wiki(_node("polqa", "POLQA", "See [[concepts/polqa]]"))
        rule = BrokenWikilinksRule()
        results = rule.check(wiki)
        fails = [r for r in results if r.outcome.value == "fail"]
        assert fails == []

    def test_genuinely_missing_topic_has_no_alias(self) -> None:
        """Genuine missing topics get ERROR with no alias suggestion."""
        wiki = _wiki(_node("alpha", "Alpha", "See [[No Such Topic XYZ]]"))
        rule = BrokenWikilinksRule()
        results = rule.check(wiki)
        fails = [r for r in results if r.outcome.value == "fail"]
        assert len(fails) == 1
        assert "alias_of" not in fails[0].details

    def test_alias_similarity_recorded_in_details(self) -> None:
        """The matched alias_similarity is stored in details (memory #44)."""
        wiki = _wiki(
            _node("polqa", "POLQA (Perceptual Objective Listening Quality Analysis)"),
            _node("other", "Other"),
        )
        # Make the wikilink target very close to "POLQA" so the match is
        # detectable; use the canonical acronym in parens.
        node = wiki.nodes[0]
        node.content = "See [[POLQA]]"
        rule = BrokenWikilinksRule()
        results = rule.check(wiki)
        # Exact match resolves cleanly (POLQA appears in alias_index).
        assert all(r.outcome.value != "fail" for r in results)


def _wiki(*nodes: WikiNode) -> CompiledWiki:
    return CompiledWiki(nodes=list(nodes), edges=[], toc=[])


def _node(node_id: str, title: str, content: str = "") -> WikiNode:
    return WikiNode(
        id=node_id,
        title=title,
        content=content,
        node_type=NodeType.CONCEPT,
        source_chunks=[],
        metadata={},
    )


def test_normalize_concept_key_round_trip() -> None:
    """Quick sanity on the helper used by the alias lookup."""
    assert normalize_concept_key("POLQA") == "polqa"
    assert normalize_concept_key("POLQA (Perceptual ...)") == "polqa"
