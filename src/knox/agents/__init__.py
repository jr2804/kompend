"""PydanticAI-based agents for the KNOX compilation pipeline.

This package (note: plural ``agents/``) is distinct from ``knox/agent/``
(singular), which implements the ACP protocol adapter for editors. The
two layers stack: an ACP request from the editor can dispatch through
``knox/agent/`` to a TOML-defined conversational agent (knox-6xb1)
that calls MCP tools; the MCP tools in turn drive the pipeline, which
internally uses the PydanticAI agents in this package (knox-pyda).

Why PydanticAI here? Each compilation stage (extraction, generation,
wicer probe/eval, query synthesis) is a structured-output call to an
LLM. PydanticAI gives us:

- **Typed structured output** via Pydantic models — no more
  ``model_validate()`` boilerplate at every call site, auto-retry on
  mismatch, and per-model output_type negotiation.
- **Single-turn agents** — each stage becomes a one-shot
  ``agent.run_sync(...)`` rather than a stateless LLM wrapper.
- **Future MCP tool injection** — PydanticAI's capabilities system
  can attach MCP servers to a specific agent without touching the
  pipeline (Phase 2 of knox-pyda).

This package is **additive**. The existing pydantic-ai path in
``knox/llm/client.py`` continues to work in parallel; pipeline stages
opt into PydanticAI via a config flag.
"""

from knox.agents.base import KnoxAgentBase
from knox.agents.deps import (
    QueryDeps,
    WicerDiagnoseDeps,
    WicerEvaluateDeps,
    WicerProbeDeps,
)
from knox.agents.doctor import DoctorAgent, DoctorDeps
from knox.agents.extraction import ExtractionAgent
from knox.agents.extraction_reduce import ExtractionReduceAgent
from knox.agents.query import QueryAgent
from knox.agents.types import (
    PipelineAgentDeps,
    ToolGenerationDeps,
)
from knox.agents.wicer import WicerDiagnoseAgent, WicerEvaluateAgent, WicerProbeAgent

__all__ = [
    "DoctorAgent",
    "DoctorDeps",
    "ExtractionAgent",
    "ExtractionReduceAgent",
    "KnoxAgentBase",
    "PipelineAgentDeps",
    "QueryAgent",
    "QueryDeps",
    "ToolGenerationDeps",
    "WicerDiagnoseAgent",
    "WicerDiagnoseDeps",
    "WicerEvaluateAgent",
    "WicerEvaluateDeps",
    "WicerProbeAgent",
    "WicerProbeDeps",
]
