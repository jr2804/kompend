"""Tests for wikilink alias learning + known-title prompt injection.

Covers beads issues knox-2kai (alias learning) and knox-oumw (long-form
wikilink prompts): :mod:`knox.compilation.flows.generation`.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from knox.compilation.flows.generation import build_known_titles, learn_wikilink_aliases
from knox.core.cache.wiki import write_wiki_cache
from knox.core.models import CompiledWiki, WikiNode
from knox.core.models.enums import MetadataField
from knox.core.templates import get_template_loader

# ── learn_wikilink_aliases (knox-2kai) ──────────────────────────────────


def test_learn_alias_from_substring_match() -> None:
    """A wikilink like [[P.863]] that fuzzy-matches an existing page is
    learned as alias -> canonical title.
    """
    nodes = [
        _node("n1", "EVS Codec", "Evaluated by [[P.863]]-based tools"),
        _node("p863-eval", "Full-scale database (P.863 evaluation)", "Definition"),
    ]
    aliases = learn_wikilink_aliases(nodes)
    assert aliases == {"P.863": "Full-scale database (P.863 evaluation)"}


def test_learn_skips_exact_matches() -> None:
    """Wikilinks that resolve exactly teach nothing — no noise in the map."""
    nodes = [
        _node("n1", "Alpha", "See [[PESQ (Perceptual Evaluation of Speech Quality)]]"),
        _node("pesq", "PESQ (Perceptual Evaluation of Speech Quality)", "Definition"),
    ]
    assert learn_wikilink_aliases(nodes) == {}


def test_learn_skips_genuinely_missing() -> None:
    """Targets with no nearby page are not aliased (they stay 'missing')."""
    nodes = [_node("n1", "Alpha", "See [[No Such Topic XYZ]]")]
    assert learn_wikilink_aliases(nodes) == {}


def test_learn_empty_nodes() -> None:
    assert learn_wikilink_aliases([]) == {}


def test_learn_dedupes_repeated_targets() -> None:
    """The same alias referenced from several pages is learned once."""
    nodes = [
        _node("n1", "Alpha", "[[P.863]] and again [[P.863]]"),
        _node("n2", "Beta", "Also [[P.863]]"),
        _node("p863-eval", "Full-scale database (P.863 evaluation)", "Definition"),
    ]
    aliases = learn_wikilink_aliases(nodes)
    assert list(aliases) == ["P.863"]


def _node(nid: str, title: str, content: str) -> WikiNode:
    return WikiNode(id=nid, title=title, content=content, node_type="entity")


def test_build_known_titles_from_concepts(tmp_path: Path) -> None:
    """Merged concept names are passed through in order."""
    compiler = _fake_compiler(tmp_path)
    concepts = [{"name": "POLQA"}, {"name": "PESQ"}]
    titles = build_known_titles(compiler, concepts)  # type: ignore[arg-type]
    assert titles == ["POLQA", "PESQ"]


def test_build_known_titles_merges_previous_aliases(tmp_path: Path) -> None:
    """Canonical titles learned by the previous compile (stored in the
    on-disk wiki cache metadata) are appended, deduplicated.
    """
    previous = CompiledWiki(
        nodes=[],
        edges=[],
        toc=[],
        metadata={MetadataField.WIKILINK_ALIASES: {"P.863": "Full-scale database (P.863 evaluation)"}},
    )
    cache_dir = tmp_path / ".knox"
    cache_dir.mkdir()
    write_wiki_cache(previous, tmp_path)

    compiler = _fake_compiler(tmp_path)
    titles = build_known_titles(compiler, [{"name": "POLQA"}])  # type: ignore[arg-type]
    assert titles == ["POLQA", "Full-scale database (P.863 evaluation)"]


def test_build_known_titles_survives_missing_cache(tmp_path: Path) -> None:
    """No previous cache (first compile) — only concept names, no error."""
    compiler = _fake_compiler(tmp_path)
    titles = build_known_titles(compiler, [{"name": "POLQA"}])  # type: ignore[arg-type]
    assert titles == ["POLQA"]


# ── build_known_titles (knox-2kai + knox-oumw wiring) ───────────────────


def _fake_compiler(base_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(_config=SimpleNamespace(paths=SimpleNamespace(base_dir=base_dir)))


# ── Prompt template (knox-oumw) ─────────────────────────────────────────


def test_generation_prompt_renders_known_titles() -> None:
    """The prompt template lists canonical titles verbatim when provided."""
    loader = get_template_loader()
    rendered = loader.render(
        "compilation/generation_prompt.md.j2",
        concepts_json="[]",
        content="source text",
        pinned_facts=[],
        known_titles=["POLQA (Perceptual Objective Listening Quality Analysis)"],
    )
    assert "KNOWN PAGE TITLES" in rendered
    assert "- POLQA (Perceptual Objective Listening Quality Analysis)" in rendered


def test_generation_prompt_omits_known_titles_when_empty() -> None:
    """Without known titles the section is absent (backward compatible)."""
    loader = get_template_loader()
    rendered = loader.render(
        "compilation/generation_prompt.md.j2",
        concepts_json="[]",
        content="source text",
        pinned_facts=[],
        known_titles=[],
    )
    assert "KNOWN PAGE TITLES" not in rendered


def test_generation_system_instructs_exact_titles() -> None:
    """The system prompt tells the LLM to use exact full titles in wikilinks."""
    loader = get_template_loader()
    system = loader.get_body("compilation/generation_system.md.j2")
    assert "EXACT full title" in system
