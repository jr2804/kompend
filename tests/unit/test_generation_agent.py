"""Tests for the generation agents and their prompt/kind plumbing.

Covers the agentic generation agent's tool registration plus the
generation deps and page-kind helpers. No LLM is contacted.
"""

from __future__ import annotations

from kompend.agents.generation import make_agentic_generation_agent
from kompend.compilation.flows.generation import _allowed_kind, _categories_block, _resolve_page_kind
from kompend.compilation.schema import GenerationPage


class TestGenerationAgentConstruction:
    def test_agentic_agent_registers_gen_tools(self) -> None:
        """The agentic generation agent must expose read_segment/fetch_concepts
        and list_known_titles (the titles the prompt does not inline).

        Regression: the ``tools`` field was never passed to the pydantic-ai
        ``Agent`` constructor, so the model had no tools and generated pages
        from memory instead of fetching the concept list (leaving every
        assigned concept uncovered).
        """
        agent = make_agentic_generation_agent(model_string="ollama/gpt-oss:20b")
        built = agent._build_agent()
        assert sorted(built._function_toolset.tools.keys()) == ["fetch_concepts", "list_known_titles", "read_segment"]


class TestPageKinds:
    """Category taxonomy: mandatory kinds, configured extras, coercion."""

    def test_generation_page_defaults_to_concept(self) -> None:
        page = GenerationPage(title="T", content="c")
        assert page.kind == "concept"

    def test_generation_page_accepts_any_kind(self) -> None:
        # The schema is permissive; strict validation happens at node creation.
        assert GenerationPage(title="T", content="c", kind="entity").kind == "entity"

    def test_allowed_kind_accepts_mandatory_and_extras(self) -> None:
        assert _allowed_kind("concept", {}) == "concept"
        assert _allowed_kind("entity", {}) == "entity"
        assert _allowed_kind("summary", {}) == "summary"
        assert _allowed_kind("comparison", {}) == "comparison"
        assert _allowed_kind("procedure", {"procedure": "how-to"}) == "procedure"

    def test_allowed_kind_falls_back_to_concept(self) -> None:
        assert _allowed_kind("garbage", {}) == "concept"
        assert _allowed_kind("procedure", {}) == "concept"  # extra not configured

    def test_categories_block_renders_instructions(self) -> None:
        block = _categories_block({"procedure": "how-to pages.", "case-study": "worked example."})
        assert '- "procedure" — how-to pages' in block
        assert '- "case-study" — worked example' in block

    def test_categories_block_empty_for_no_extras(self) -> None:
        assert _categories_block({}) == ""


class TestResolvePageKind:
    """Pre-assigned kinds are enforced; model latitude kept everywhere else."""

    def test_preassigned_entity_wins_over_model_label(self) -> None:
        preassigned = {"itu t": "entity"}
        # Whatever the model emitted, a pre-assigned page keeps kind=entity.
        assert _resolve_page_kind("concept", "ITU-T", preassigned, {}) == "entity"
        assert _resolve_page_kind("procedure", "itu t", preassigned, {"procedure": "x"}) == "entity"

    def test_preassigned_matches_by_normalized_key(self) -> None:
        # Title formatting does not matter; normalize_concept_key folds it.
        assert _resolve_page_kind("summary", "ITU T", {"itu t": "entity"}, {}) == "entity"

    def test_non_preassigned_honours_model_kind(self) -> None:
        assert _resolve_page_kind("summary", "Overview", {}, {}) == "summary"
        assert _resolve_page_kind("procedure", "How-to", {}, {"procedure": "x"}) == "procedure"

    def test_non_preassigned_invalid_kind_falls_back(self) -> None:
        assert _resolve_page_kind("garbage", "Random", {}, {}) == "concept"
