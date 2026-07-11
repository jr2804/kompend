"""Dependency types for KNOX PydanticAI agents.

PydanticAI passes a typed ``Deps`` object to every tool call and into the
system prompt template. The :class:`PipelineAgentDeps` carries the
runtime context an agent needs to do its job — the source text being
analysed, the source identifier for provenance, and per-call budget
hints from the surrounding pipeline.

Why a single ``Deps`` type instead of one per agent? It keeps the
agent-factory surface small. Agents that need additional context
(generation, wicer) extend the base class with their own Deps subclass
in a follow-up phase. Phase 1 (knox-pyda) only needs extraction.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PipelineAgentDeps:
    """Runtime context shared by all pipeline agents.

    Attributes:
        text: The source content the agent is operating on. For
            extraction this is the raw chunk text; for generation it
            would be the concept batch; etc. Each agent decides how
            to use it.
        source_id: Stable identifier for the source (file path stem,
            URL, etc.) used for provenance and logging. Does not need
            to be a Path — plain string is fine.
        timeout: Per-call timeout in seconds. Forwarded to
            PydanticAI's model settings so a single slow call doesn't
            stall the pipeline.
    """

    text: str
    source_id: str
    timeout: float = 120.0
