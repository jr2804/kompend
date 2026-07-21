"""Pydantic models for LLM JSON output schemas.

Each model mirrors the JSON schema documented in the corresponding Jinja2
template under ``templates/compilation/``.  These serve three purposes:

1. **Documentation** — the expected LLM output shape lives in one place,
   not scattered across prompt templates and dict.get() call sites.
2. **Validation** — ``model_validate(data)`` catches wrong types and missing
   fields that ``isinstance(data, dict)`` misses.
3. **Foundation** — enables future integration with structured-output
   frameworks like ``instructor`` that require Pydantic models.

Usage::

    from knox.compilation.schema import ExtractionOutput

    result = agent.run(deps)  # PydanticAI agent with output_type=ExtractionOutput
    assert isinstance(result, ExtractionOutput)
    # result.summary, result.entities[0].name, etc.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator

from knox.core.models.enums import ExtractionField
from knox.core.utils.exceptions import LLMOutputError

__all__ = [
    "Concept",
    "Entity",
    "ExtractionOutput",
    "Finding",
    "GenerationOutput",
    "GenerationPage",
    "LLMOutputError",
    "ValidationError",
    "WicerDiagnoseOutput",
    "WicerEvalResult",
    "WicerEvaluateOutput",
    "WicerProbe",
    "WicerProbeOutput",
]


# ── Extraction (extraction_prompt.md.j2) ─────────────────────────────────


class Entity(BaseModel):
    """Named entity extracted from a source document."""

    name: str
    type: str = ""  # Author, Model, Dataset, Metric, Algorithm, System, Organization, Standard, Term
    context: str = ""


class Concept(BaseModel):
    """Key technical concept extracted from a source document."""

    name: str
    definition: str = ""
    importance: str = ""


class Finding(BaseModel):
    """Concrete result or claim from a source document."""

    claim: str
    evidence: str = ""


class ExtractionOutput(BaseModel):
    """LLM output schema for ``compilation/extraction_prompt.md.j2``.

    The extraction prompt asks the LLM to analyse a source text and return
    a JSON object with summary, entities, concepts, findings, and tags.
    """

    summary: str = ""
    entities: list[Entity] = Field(default_factory=list)
    concepts: list[Concept] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


# ── Generation (generation_prompt.md.j2) ─────────────────────────────────


class GenerationPage(BaseModel):
    """A single generated wiki page from the two-phase generation step."""

    title: str
    slug: str = ""
    content: str = ""
    provenance: list[str] = Field(default_factory=list)
    related: list[str] = Field(default_factory=list)

    @field_validator("title", "slug", "content", mode="before")
    @classmethod
    def coerce_none_to_str(cls, v: Any) -> str:
        """Coerce ``None`` to empty string for required string fields.

        Small models sometimes return ``None`` for title/slug/content
        instead of an empty string. Pydantic rejects ``None`` for ``str``
        fields; coerce to ``""`` to avoid a validation failure.
        """
        if v is None:
            return ""
        return v

    @field_validator("provenance", mode="before")
    @classmethod
    def coerce_provenance(cls, v: Any) -> list[str]:
        """Filter non-string items from provenance.

        Small models sometimes put dicts (page metadata) in the provenance
        field. Keep only string items; discard dicts/lists/None.
        """
        if not isinstance(v, list):
            return v
        return [item for item in v if isinstance(item, str)]

    @field_validator("related", mode="before")
    @classmethod
    def coerce_related(cls, v: Any) -> list[str]:
        """Coerce ``list[dict]`` to ``list[str]`` by extracting the ``slug`` key.

        Small models frequently return related pages as ``[{title, slug}, ...]``
        instead of ``[slug, ...]``.  Extract the slug when a dict is encountered;
        pass through strings unchanged.
        """
        if not isinstance(v, list):
            return v
        result: list[str] = []
        for item in v:
            if isinstance(item, str):
                result.append(item)
            elif isinstance(item, dict):
                slug = item.get(ExtractionField.SLUG, "") or ""
                if slug:
                    result.append(slug)
        return result


class GenerationOutput(BaseModel):
    """LLM output schema for ``compilation/generation_prompt.md.j2``.

    The generation prompt asks the LLM to produce one wiki page per concept,
    returned as ``{"pages": [...]}``.
    """

    pages: list[GenerationPage] = Field(default_factory=list)


# ── WiCER (wicer_*.md.j2) ────────────────────────────────────────────────


class WicerProbe(BaseModel):
    """A single Q&A probe generated from source text."""

    question: str
    answer: str = ""


class WicerProbeOutput(BaseModel):
    """LLM output schema for ``compilation/wicer_probe.md.j2``."""

    probes: list[WicerProbe] = Field(default_factory=list)


class WicerEvalResult(BaseModel):
    """Evaluation result for a single probe against the compiled wiki."""

    question: str = ""
    expected_answer: str = ""
    found_answer: str = ""
    correct: bool = False


class WicerEvaluateOutput(BaseModel):
    """LLM output schema for ``compilation/wicer_evaluate.md.j2``."""

    results: list[WicerEvalResult] = Field(default_factory=list)


class WicerDiagnoseOutput(BaseModel):
    """LLM output schema for ``compilation/wicer_diagnose.md.j2``."""

    critical_facts: list[str] = Field(default_factory=list)
