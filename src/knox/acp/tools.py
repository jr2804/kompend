"""Tool dispatcher for the conversational agent loop.

Wraps existing MCP server functions as LLM-callable tools. Injects the
vault's ``base_dir`` automatically so the LLM doesn't need to specify it.

Each tool is defined with an explicit schema (name, description, parameters)
and an async executor that calls the underlying function via
``asyncio.to_thread`` (MCP functions are synchronous).

Tool list:
- ``wiki_status`` — check vault health and compiled wiki presence
- ``compile_wiki`` — compile sources into a wiki
- ``query_wiki`` — query the compiled wiki
- ``wiki_capture`` — capture a note
- ``wiki_learn`` — learn/distill from content
- ``config_get`` — read a config value
- ``config_set`` — write a config value
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from knox.core.config import DEFAULTS
from knox.core.pipeline.config import Compiler
from knox.logging import get_logger
from knox.mcp import server as _mcp

logger = get_logger(__name__)

_MAX_TOOL_RESULT_CHARS = 4000


class ToolName(StrEnum):
    """All tool names exposed to the conversational agent."""

    WIKI_STATUS = "wiki_status"
    COMPILE_WIKI = "compile_wiki"
    QUERY_WIKI = "query_wiki"
    WIKI_CAPTURE = "wiki_capture"
    CONFIG_GET = "config_get"
    CONFIG_SET = "config_set"


@dataclass(frozen=True)
class ToolSchema:
    """LLM-facing schema for a single tool."""

    name: str
    description: str
    parameters: dict[str, Any]


# ── Tool schemas (OpenAI/litellm function format) ──────────────────────


def _build_schemas() -> dict[str, ToolSchema]:
    """Return all available tool schemas keyed by name."""
    schemas: list[ToolSchema] = [
        ToolSchema(
            name=ToolName.WIKI_STATUS,
            description="Check vault health: whether a wiki is compiled, how many pages, source count.",
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
        ToolSchema(
            name=ToolName.COMPILE_WIKI,
            description="Compile documents into a knowledge wiki. Sources must already be ingested.",
            parameters={
                "type": "object",
                "properties": {
                    "compiler": {
                        "type": "string",
                        "description": "Compiler backend: 'graph' (default) or 'wicer'.",
                        "default": "graph",
                    },
                },
                "required": [],
            },
        ),
        ToolSchema(
            name=ToolName.QUERY_WIKI,
            description="Query the compiled wiki with a natural language question.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language question.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to retrieve.",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        ),
        ToolSchema(
            name=ToolName.WIKI_CAPTURE,
            description="Capture a note or observation into the knowledge base.",
            parameters={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The note text to capture.",
                    },
                },
                "required": ["text"],
            },
        ),
        ToolSchema(
            name=ToolName.CONFIG_GET,
            description="Read a configuration value by dotted key (e.g. 'compile.llm.default_model').",
            parameters={
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "Dotted config key.",
                    },
                },
                "required": ["key"],
            },
        ),
        ToolSchema(
            name=ToolName.CONFIG_SET,
            description="Set a configuration value by dotted key.",
            parameters={
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "Dotted config key.",
                    },
                    "value": {
                        "type": "string",
                        "description": "Value to set (as string).",
                    },
                },
                "required": ["key", "value"],
            },
        ),
    ]
    return {s.name: s for s in schemas}


_ALL_SCHEMAS = _build_schemas()


class ToolDispatcher:
    """Routes tool calls to existing MCP server functions.

    The vault's ``base_dir`` is injected automatically into tool calls that
    need it, so the LLM's tool schemas stay clean.
    """

    def __init__(self, base_dir: str | Path | None = None) -> None:
        self._base_dir = Path(base_dir) if base_dir else Path(".")
        self._sources_dir = DEFAULTS.get("paths", {}).get("sources_dir", "sources")
        self._output_dir = DEFAULTS.get("paths", {}).get("output_dir", "wiki")

    async def execute(self, name: str, arguments: dict[str, Any]) -> str:
        """Execute a tool by name and return the result as a JSON string.

        On error, returns ``{"status": "error", "message": "..."}`` as a string.
        Result is truncated to ``_MAX_TOOL_RESULT_CHARS`` to protect context.
        """
        try:
            result = await asyncio.to_thread(self._dispatch, name, arguments)
        except Exception as exc:
            logger.exception("Tool execution failed: %s", name)
            result = {"status": "error", "message": str(exc)}

        text = json.dumps(result, ensure_ascii=False, default=str)
        if len(text) > _MAX_TOOL_RESULT_CHARS:
            text = text[:_MAX_TOOL_RESULT_CHARS] + "\n…(truncated)"
        return text

    @staticmethod
    def schemas(enabled: list[str] | None = None) -> list[dict[str, Any]]:
        """Return tool schemas in litellm/OpenAI format for the LLM.

        Args:
            enabled: Tool names to include. *None* = all tools.
        """
        names = enabled if enabled is not None else list(_ALL_SCHEMAS)
        result: list[dict[str, Any]] = []
        for name in names:
            schema = _ALL_SCHEMAS.get(name)
            if schema is None:
                logger.warning("Unknown tool in enabled list: %s", name)
                continue
            result.append(
                {
                    "type": "function",
                    "function": {
                        "name": schema.name,
                        "description": schema.description,
                        "parameters": schema.parameters,
                    },
                }
            )
        return result

    def _dispatch(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Call the underlying MCP server function via dispatch table."""
        handler = _DISPATCH_TABLE.get(name)
        if handler is None:
            return {"status": "error", "message": f"Unknown tool: {name}"}
        return handler(self, arguments)


# ── Dispatch table ────────────────────────────────────────────────────
# Maps tool name → callable(dispatcher, arguments) → MCP result dict.
# Keeps _dispatch() clean and makes adding new tools a single entry.


def _cmd_wiki_status(self: ToolDispatcher, args: dict[str, Any]) -> dict[str, Any]:
    return _mcp.wiki_status(wiki_dir=str(self._base_dir))


def _cmd_compile_wiki(self: ToolDispatcher, args: dict[str, Any]) -> dict[str, Any]:
    compiler_raw = args.get("compiler", Compiler.GRAPH.value)
    compiler = Compiler(compiler_raw) if compiler_raw else Compiler.GRAPH
    return _mcp.compile_wiki(
        sources_dir=str(args.get("sources_dir", self._base_dir / self._sources_dir)),
        output=str(self._base_dir / self._output_dir),
        compiler=compiler.value,
    )


def _cmd_query_wiki(self: ToolDispatcher, args: dict[str, Any]) -> dict[str, Any]:
    return _mcp.query_wiki(
        query=args["query"],
        wiki_dir=str(self._base_dir),
        top_k=args.get("top_k", 5),
    )


def _cmd_wiki_capture(self: ToolDispatcher, args: dict[str, Any]) -> dict[str, Any]:
    return _mcp.wiki_capture(
        text=args["text"],
        sources_dir=str(self._base_dir / self._sources_dir),
    )


def _cmd_config_get(self: ToolDispatcher, args: dict[str, Any]) -> dict[str, Any]:
    return _mcp.config_get(key=args["key"])


def _cmd_config_set(self: ToolDispatcher, args: dict[str, Any]) -> dict[str, Any]:
    return _mcp.config_set(key=args["key"], value=args["value"])


_DISPATCH_TABLE: dict[str, Any] = {
    ToolName.WIKI_STATUS: _cmd_wiki_status,
    ToolName.COMPILE_WIKI: _cmd_compile_wiki,
    ToolName.QUERY_WIKI: _cmd_query_wiki,
    ToolName.WIKI_CAPTURE: _cmd_wiki_capture,
    ToolName.CONFIG_GET: _cmd_config_get,
    ToolName.CONFIG_SET: _cmd_config_set,
}
