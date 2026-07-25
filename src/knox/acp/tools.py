"""Native PydanticAI toolset for the conversational agent.

Wraps existing MCP server functions as PydanticAI ``FunctionToolset``
tools. Each tool is a proper ``@toolset.tool_plain`` function so
PydanticAI handles schema generation, validation, and retry natively.

The vault's ``base_dir`` is injected via closure so the LLM never needs
to specify it.

Tool list:
- ``wiki_status`` — check vault health and compiled wiki presence
- ``compile_wiki`` — compile sources into a wiki
- ``query_wiki`` — query the compiled wiki
- ``wiki_capture`` — capture a note
- ``config_get`` — read a config value
- ``config_set`` — write a config value
"""

from __future__ import annotations

import asyncio
import json
from enum import StrEnum
from pathlib import Path

from pydantic_ai.toolsets import FunctionToolset

from knox.core.config import DEFAULTS
from knox.core.pipeline.config import Compiler
from knox.logging import get_logger
from knox.mcp import server as mcp_server

logger = get_logger(__name__)


class ToolName(StrEnum):
    """All tool names exposed to the conversational agent."""

    WIKI_STATUS = "wiki_status"
    COMPILE_WIKI = "compile_wiki"
    QUERY_WIKI = "query_wiki"
    WIKI_CAPTURE = "wiki_capture"
    CONFIG_GET = "config_get"
    CONFIG_SET = "config_set"


def build_toolset(
    base_dir: str | Path,
    enabled: list[str] | None = None,
) -> FunctionToolset[None]:
    """Build a PydanticAI ``FunctionToolset`` with KNOX tools.

    Each tool calls the corresponding MCP server function with ``base_dir``
    injected via closure. PydanticAI generates JSON schemas from the type
    hints and docstrings automatically.

    Args:
        base_dir: Vault root directory (injected into every tool call).
        enabled: Tool names to register. *None* = all available tools.

    Returns:
        A ``FunctionToolset`` ready to pass to ``Agent(toolsets=[...])``.
    """
    base = Path(base_dir)
    sources_dir = DEFAULTS.get("paths", {}).get("sources_dir", "sources")
    output_dir = DEFAULTS.get("paths", {}).get("output_dir", "wiki")
    toolset: FunctionToolset[None] = FunctionToolset()

    allowed = set(enabled) if enabled is not None else None

    def _is_enabled(name: str) -> bool:
        return allowed is None or name in allowed

    # ── wiki_status ────────────────────────────────────────────────────

    if _is_enabled(ToolName.WIKI_STATUS):

        @toolset.tool_plain
        async def wiki_status() -> str:
            """Check vault health: whether a wiki is compiled, how many pages, source count."""
            result = await asyncio.to_thread(mcp_server.wiki_status, wiki_dir=str(base))
            return json.dumps(result, ensure_ascii=False, default=str)

    # ── compile_wiki ───────────────────────────────────────────────────

    if _is_enabled(ToolName.COMPILE_WIKI):

        @toolset.tool_plain
        async def compile_wiki(compiler: str = "graph") -> str:
            """Compile documents into a knowledge wiki. Sources must already be ingested.

            Args:
                compiler: Compiler backend: 'graph' (default) or 'wicer'.
            """
            comp = Compiler(compiler) if compiler else Compiler.GRAPH
            result = await asyncio.to_thread(
                mcp_server.compile_wiki,
                sources_dir=str(base / sources_dir),
                output=str(base / output_dir),
                compiler=comp.value,
            )
            return json.dumps(result, ensure_ascii=False, default=str)

    # ── query_wiki ─────────────────────────────────────────────────────

    if _is_enabled(ToolName.QUERY_WIKI):

        @toolset.tool_plain
        async def query_wiki(query: str, top_k: int = 5) -> str:
            """Query the compiled wiki with a natural language question.

            Args:
                query: Natural language question.
                top_k: Number of results to retrieve.
            """
            result = await asyncio.to_thread(
                mcp_server.query_wiki,
                query=query,
                wiki_dir=str(base),
                top_k=top_k,
            )
            return json.dumps(result, ensure_ascii=False, default=str)

    # ── wiki_capture ───────────────────────────────────────────────────

    if _is_enabled(ToolName.WIKI_CAPTURE):

        @toolset.tool_plain
        async def wiki_capture(text: str) -> str:
            """Capture a note or observation into the knowledge base.

            Args:
                text: The note text to capture.
            """
            result = await asyncio.to_thread(
                mcp_server.wiki_capture,
                text=text,
                sources_dir=str(base / sources_dir),
            )
            return json.dumps(result, ensure_ascii=False, default=str)

    # ── config_get ─────────────────────────────────────────────────────

    if _is_enabled(ToolName.CONFIG_GET):

        @toolset.tool_plain
        async def config_get(key: str) -> str:
            """Read a configuration value by dotted key (e.g. 'compile.llm.default_model').

            Args:
                key: Dotted config key.
            """
            result = await asyncio.to_thread(mcp_server.config_get, key=key)
            return json.dumps(result, ensure_ascii=False, default=str)

    # ── config_set ─────────────────────────────────────────────────────

    if _is_enabled(ToolName.CONFIG_SET):

        @toolset.tool_plain
        async def config_set(key: str, value: str) -> str:
            """Set a configuration value by dotted key.

            Args:
                key: Dotted config key.
                value: Value to set (as string).
            """
            result = await asyncio.to_thread(mcp_server.config_set, key=key, value=value)
            return json.dumps(result, ensure_ascii=False, default=str)

    return toolset
