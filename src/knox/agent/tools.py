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
from pathlib import Path
from typing import Any

from knox.logging import get_logger

logger = get_logger(__name__)

_MAX_TOOL_RESULT_CHARS = 4000


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
            name="wiki_status",
            description="Check vault health: whether a wiki is compiled, how many pages, source count.",
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
        ToolSchema(
            name="compile_wiki",
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
            name="query_wiki",
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
            name="wiki_capture",
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
            name="config_get",
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
            name="config_set",
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
        self._base_dir = str(base_dir) if base_dir else "."

    def schemas(self, enabled: list[str] | None = None) -> list[dict[str, Any]]:  # noqa: PLR6301
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

    def _dispatch(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:  # noqa: PLR0911
        """Call the underlying MCP server function."""
        # Lazy import — avoids loading litellm/sentence_transformers at module import.
        from knox.mcp import server as _mcp  # noqa: PLC0415

        if name == "wiki_status":
            return _mcp.wiki_status(wiki_dir=self._base_dir)
        if name == "compile_wiki":
            return _mcp.compile_wiki(
                sources_dir=arguments.get("sources_dir", self._base_dir + "/sources"),
                output=self._base_dir + "/wiki",
                compiler=arguments.get("compiler", "graph"),
            )
        if name == "query_wiki":
            return _mcp.query_wiki(
                query=arguments["query"],
                wiki_dir=self._base_dir,
                top_k=arguments.get("top_k", 5),
            )
        if name == "wiki_capture":
            return _mcp.wiki_capture(
                text=arguments["text"],
                sources_dir=self._base_dir + "/sources",
            )
        if name == "config_get":
            return _mcp.config_get(key=arguments["key"])
        if name == "config_set":
            return _mcp.config_set(
                key=arguments["key"],
                value=arguments["value"],
            )

        return {"status": "error", "message": f"Unknown tool: {name}"}
