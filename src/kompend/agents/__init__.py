"""PydanticAI-based agents for the kompend compilation pipeline.

This package (note: plural ``agents/``) is distinct from ``kompend/acp/``, which
implements the ACP protocol adapter for editors. The two layers stack: an ACP
request from the editor dispatches to a TOML-defined conversational agent that
calls MCP tools; the MCP tools in turn drive the pipeline, which uses the
PydanticAI agents in this package.

Every stage that calls a model goes through an agent here: typed structured
output, one ``run_sync``/``run`` per unit of work, and capabilities (retry,
compaction, MCP tool injection) attached per agent instead of per call site.
"""

from kompend.agents.base import KompendAgentBase
from kompend.agents.deps import (
    MemoryConsolidationDeps,
    QueryDeps,
    WicerDiagnoseDeps,
    WicerEvaluateDeps,
    WicerProbeDeps,
)
from kompend.agents.doctor import DoctorAgent, DoctorDeps
from kompend.agents.extraction import ExtractionAgent
from kompend.agents.extraction_reduce import ExtractionReduceAgent
from kompend.agents.failure import (
    AgentFailureKind,
    EmptyAnswerError,
    UnitFailure,
    UnitFailureLog,
    classify_agent_failure,
    is_retry_pointless,
    record_unit_failure,
)
from kompend.agents.memory_consolidation import MemoryConsolidationAgent
from kompend.agents.query import QueryAgent
from kompend.agents.types import (
    PipelineAgentDeps,
    ToolGenerationDeps,
)
from kompend.agents.wicer import WicerDiagnoseAgent, WicerEvaluateAgent, WicerProbeAgent

__all__ = [
    "AgentFailureKind",
    "DoctorAgent",
    "DoctorDeps",
    "EmptyAnswerError",
    "ExtractionAgent",
    "ExtractionReduceAgent",
    "KompendAgentBase",
    "MemoryConsolidationAgent",
    "MemoryConsolidationDeps",
    "PipelineAgentDeps",
    "QueryAgent",
    "QueryDeps",
    "ToolGenerationDeps",
    "UnitFailure",
    "UnitFailureLog",
    "WicerDiagnoseAgent",
    "WicerDiagnoseDeps",
    "WicerEvaluateAgent",
    "WicerEvaluateDeps",
    "WicerProbeAgent",
    "WicerProbeDeps",
    "classify_agent_failure",
    "is_retry_pointless",
    "record_unit_failure",
]
