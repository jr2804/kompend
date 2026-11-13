"""PydanticAI-based agents for the kompend compilation pipeline.

This package (note: plural ``agents/``) is distinct from ``kompend/acp/``, which
implements the ACP protocol adapter for editors. The two layers stack: an ACP
request from the editor dispatches to a TOML-defined conversational agent that
calls MCP tools; the MCP tools in turn drive the pipeline, which uses the
PydanticAI agents in this package.

Every stage that calls a model goes through an agent here: typed structured
output, one ``run_sync``/``run`` per unit of work, and capabilities (retry,
compaction, MCP tool injection) attached per agent instead of per call site.

The public surface is attached lazily: importing ``kompend.agents`` — directly
or as the package ``__init__`` of any submodule import — must not drag the
pydantic-ai stack into interpreter startup (measured: the vendor floor alone
costs ~4-5 s, which every CLI spawn paid through this facade). Set
``EAGER_IMPORT=1`` to materialise all names eagerly (lazy_loader's escape
hatch). The names are NOT imported at module level to satisfy a type checker:
an import here *is* the eager import this boundary exists to prevent.
"""

from __future__ import annotations

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "base": ["AgentSpec", "KompendAgentBase", "agent_class", "build_agent"],
        "decision_probe": ["DecisionProbeAgent", "DecisionProbeDeps"],
        "deps": [
            "MemoryConsolidationDeps",
            "WicerDiagnoseDeps",
            "WicerEvaluateDeps",
            "WicerProbeDeps",
        ],
        "doctor": ["DoctorAgent", "DoctorDeps"],
        "extraction": ["ExtractionAgent"],
        "extraction_reduce": ["ExtractionReduceAgent"],
        "failure": [
            "AgentFailureKind",
            "EmptyAnswerError",
            "UnitFailure",
            "UnitFailureLog",
            "classify_agent_failure",
            "is_retry_pointless",
            "record_unit_failure",
        ],
        "memory_consolidation": ["MemoryConsolidationAgent"],
        "query": ["QueryAgent"],
        "specs": ["AgentName", "get_spec"],
        "types": ["PipelineAgentDeps", "ToolGenerationDeps"],
        "wicer": [
            "WicerDiagnoseAgent",
            "WicerEvaluateAgent",
            "WicerProbeAgent",
        ],
    },
)
