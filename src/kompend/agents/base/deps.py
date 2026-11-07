"""The deps contract of every kompend agent.

``AgentDeps`` is the structural protocol a deps dataclass must satisfy
(a ``source_id`` field); ``ResultT``/``DepsT`` are the agent's public
generic parameters. ``resolve_retry_budget`` turns the ``compile.llm``
retry keys into the tool/output retry split the agent base hands to
pydantic-ai, and the ``UNSET_*`` sentinels mark "the construction site
did not set this limit" so the config (or the default) can decide.
"""

from __future__ import annotations

import typing
from typing import Protocol, TypeVar, runtime_checkable

from pydantic_ai.agent.abstract import AgentRetries

from kompend.core.config._configs import LLMConfig

#: Retry fallback when neither the construction site nor the config set a
#: budget (``compile.llm.output_retries`` defaults to 2).
_DEFAULT_RETRIES: int = 2

#: Sentinels for "the construction site did not set this limit" — the
#: config (``compile.llm``) or the base default decides.
UNSET_TIMEOUT: float = -1.0
UNSET_RETRIES: int = -1


ResultT = TypeVar("ResultT")


@runtime_checkable
class AgentDeps(Protocol):
    """Structural type for deps objects passed to :class:`KompendAgentBase`.

    Any dataclass with a ``source_id: str`` field
    satisfies this protocol.
    """

    source_id: str


DepsT = TypeVar("DepsT", bound=AgentDeps)


def resolve_retry_budget(
    llm: LLMConfig | None,
    *,
    retries: int | AgentRetries | None,
    output_retries: int | None,
) -> int | AgentRetries:
    """Resolve the tool/output retry split from the config.

    ``retries`` bounds tool-call retries (``compile.llm.tool_retries``),
    ``output_retries`` bounds output validation
    (``compile.llm.agentic_output_retries`` for agentic agents). Both come
    from ``compile.llm`` when the caller left them unset;
    ``compile.llm.output_retries_enabled = false`` switches retries off
    entirely.
    """
    if llm is not None:
        if retries is None or retries == UNSET_RETRIES:
            retries = llm.tool_retries if llm.output_retries_enabled else 0
        if output_retries is None:
            output_retries = llm.agentic_output_retries if llm.output_retries_enabled else 0
    if retries is None or retries == UNSET_RETRIES:
        retries = _DEFAULT_RETRIES
    if output_retries is not None:
        tool_retries = retries if isinstance(retries, int) else retries.get("tools", _DEFAULT_RETRIES)
        return {"tools": tool_retries, "output": output_retries}
    return retries


def _coerce_type(arg: type, fallback: type) -> type:
    """Resolve a TypeVar arg to a usable runtime type.

    Handles plain types (``str``), generic aliases (``dict[str, Any]``),
    and TypeVars (falls back to *fallback*).
    """
    # Plain class — use directly
    if isinstance(arg, type):
        return arg
    # Generic alias like dict[str, Any] — get_origin returns dict
    origin = typing.get_origin(arg)
    if origin is not None and isinstance(origin, type):
        return origin
    return fallback
