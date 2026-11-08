"""Agent-spec registry.

Each agent module owns one ``SPEC`` (:class:`AgentSpec`). The registry maps
the spec name to the module that declares it, so a caller that only knows the
name — a pipeline stage, a docs test, the schema validation test — can resolve
the definition without importing the whole agent surface.

The map is explicit rather than a scan of ``kompend.agents``: importing every
agent module eagerly would pull the compilation schema and the provider
stack into processes that only need one stage, and the bead's migration order
is per agent. Add an entry when an agent moves to a spec.
"""

from __future__ import annotations

from enum import StrEnum
from importlib import import_module

from kompend.agents.base.spec import AgentSpec


class AgentName(StrEnum):
    """Agent-spec vocabulary. The registry and ``build_agent`` share it."""

    EXTRACTION = "extraction"
    EXTRACTION_REDUCE = "extraction_reduce"
    CAPTURE = "capture"
    CATEGORIZE = "categorize"
    CONTEXT_CHUNK = "context_chunk"
    AGENTIC_CONTEXT_CHUNK = "agentic_context_chunk"
    DECISION = "decision"
    ADAPTATION = "adaptation"
    DISTILL = "distill"
    AGENTIC_DISTILL = "agentic_distill"
    DOCTOR = "doctor"
    AGENTIC_GENERATION = "agentic_generation"
    LINT_RULE = "lint_rule"
    MEMORY_CONSOLIDATION = "memory_consolidation"
    QUERY = "query"
    AGENTIC_QUERY = "agentic_query"
    REFERENCE_EXTRACTION = "reference_extraction"
    REFERENCE_RESOLVER = "reference_resolver"
    REMEDIATION = "remediation"
    AGENTIC_REMEDIATION = "agentic_remediation"
    QUERY_EXPANSION = "query_expansion"
    CRAG_GRADE = "crag_grade"
    HYDE = "hyde"
    SUMMARIZER = "summarizer"
    AGENTIC_SUMMARIZER = "agentic_summarizer"
    VERIFICATION = "verification"
    AGENTIC_VERIFICATION = "agentic_verification"
    VLM = "vlm"
    WICER_PROBE = "wicer_probe"
    WICER_EVALUATE = "wicer_evaluate"
    WICER_DIAGNOSE = "wicer_diagnose"
    WIKI_INDEX = "wiki_index"
    AGENTIC_OVERVIEW = "agentic_overview"
    WRITE = "write"
    AGENTIC_WRITE = "agentic_write"


#: spec name -> module that declares ``SPEC``.
_MODULES: dict[AgentName, str] = {
    AgentName.EXTRACTION: "kompend.agents.extraction",
    AgentName.EXTRACTION_REDUCE: "kompend.agents.extraction_reduce",
    AgentName.CAPTURE: "kompend.agents.capture",
    AgentName.CATEGORIZE: "kompend.agents.categorize",
    AgentName.CONTEXT_CHUNK: "kompend.agents.context_chunk",
    AgentName.AGENTIC_CONTEXT_CHUNK: "kompend.agents.context_chunk_agentic",
    AgentName.DECISION: "kompend.agents.decision",
    AgentName.ADAPTATION: "kompend.agents.decision",
    AgentName.DISTILL: "kompend.agents.distill",
    AgentName.AGENTIC_DISTILL: "kompend.agents.distill_agentic",
    AgentName.DOCTOR: "kompend.agents.doctor",
    AgentName.AGENTIC_GENERATION: "kompend.agents.generation",
    AgentName.LINT_RULE: "kompend.agents.lint",
    AgentName.MEMORY_CONSOLIDATION: "kompend.agents.memory_consolidation",
    AgentName.QUERY: "kompend.agents.query",
    AgentName.AGENTIC_QUERY: "kompend.agents.query_agentic",
    AgentName.REFERENCE_EXTRACTION: "kompend.agents.reference_extraction",
    AgentName.REFERENCE_RESOLVER: "kompend.agents.reference_resolver",
    AgentName.REMEDIATION: "kompend.agents.remediate",
    AgentName.AGENTIC_REMEDIATION: "kompend.agents.remediate",
    AgentName.QUERY_EXPANSION: "kompend.agents.search",
    AgentName.CRAG_GRADE: "kompend.agents.search",
    AgentName.HYDE: "kompend.agents.search",
    AgentName.SUMMARIZER: "kompend.agents.summarizer",
    AgentName.AGENTIC_SUMMARIZER: "kompend.agents.summarizer_agentic",
    AgentName.VERIFICATION: "kompend.agents.verification",
    AgentName.AGENTIC_VERIFICATION: "kompend.agents.verification",
    AgentName.VLM: "kompend.agents.vlm",
    AgentName.WICER_PROBE: "kompend.agents.wicer",
    AgentName.WICER_EVALUATE: "kompend.agents.wicer",
    AgentName.WICER_DIAGNOSE: "kompend.agents.wicer",
    AgentName.WIKI_INDEX: "kompend.agents.wiki_index",
    AgentName.AGENTIC_OVERVIEW: "kompend.agents.wiki_index",
    AgentName.WRITE: "kompend.agents.write",
    AgentName.AGENTIC_WRITE: "kompend.agents.write",
}

_CACHE: dict[str, AgentSpec] = {}


def get_spec(name: AgentName | str) -> AgentSpec:
    """Return the spec registered under *name*."""
    try:
        key = AgentName(name)
    except ValueError as exc:
        raise KeyError(f"no agent spec registered under {name!r}") from exc
    cached = _CACHE.get(str(key))
    if cached is not None:
        return cached
    module_path = _MODULES.get(key)
    if module_path is None:
        raise KeyError(f"no agent spec registered under {name!r}")
    module = import_module(module_path)
    spec = getattr(module, "SPEC", None)
    if not isinstance(spec, AgentSpec) or spec.name != str(key):
        spec = next(
            (value for value in vars(module).values() if isinstance(value, AgentSpec) and value.name == str(key)),
            None,
        )
    if spec is None:
        raise KeyError(f"{module_path} does not declare an AgentSpec named {str(key)!r}")
    _CACHE[str(key)] = spec
    return spec


def registered_names() -> tuple[str, ...]:
    """Names known to the registry, in declaration order."""
    return tuple(str(name) for name in _MODULES)


def all_specs() -> dict[str, AgentSpec]:
    """Resolve every registered spec (imports each declaring module once)."""
    return {str(name): get_spec(name) for name in _MODULES}
