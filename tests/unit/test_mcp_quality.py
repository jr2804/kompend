"""Contract pins for the MCP quality tools (names, input schemas, payload keys).

The MCP wire contract for the quality family is exactly three things: the 15
tool names registered on the server, the fastmcp-derived input schema per
name, and the payload key sets each tool may return. ``check_mcp_parity.py``
and the test_mcp parity tests compare names only, so without these pins a
renamed parameter, a changed default, or a renamed payload key would break
every client silently. The payload pin reads the defining source with ``ast``
and therefore keeps working wherever the functions live (module or package).
"""

from __future__ import annotations

import ast
import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

import kompend.mcp.server as server_module
from kompend.mcp.server import list_parsers
from kompend.mcp.server import server as mcp_server

QUALITY_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "lint_all",
        "lint_check",
        "lint_semantic",
        "lint_history",
        "lint_show",
        "lint_prune",
        "lint_gaps",
        "lint_remediate",
        "lint_format",
        "doctor",
        "explain",
        "review",
        "provenance",
        "log",
        "list_parsers",
    }
)

# Dumped from the live server's list_tools(); the parameters JSON Schema
# fastmcp derives from each tool signature. A parameter rename, a changed
# default, or an added/removed parameter changes this golden.
GOLDEN_SCHEMAS: dict[str, dict[str, Any]] = {
    "doctor": {
        "additionalProperties": False,
        "properties": {
            "config_file": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None},
            "json_output": {"default": False, "type": "boolean"},
            "verbose": {"default": False, "type": "boolean"},
            "wiki_dir": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None},
        },
        "type": "object",
    },
    "lint_all": {
        "additionalProperties": False,
        "properties": {
            "config_file": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None},
            "fix": {"default": True, "type": "boolean"},
            "wiki_dir": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None},
        },
        "type": "object",
    },
    "explain": {
        "additionalProperties": False,
        "properties": {
            "config_file": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None},
            "run_id": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None},
            "source": {"type": "string"},
            "wiki_dir": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None},
        },
        "required": ["source"],
        "type": "object",
    },
    "lint_check": {
        "additionalProperties": False,
        "properties": {
            "config_file": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None},
            "fix": {"default": True, "type": "boolean"},
            "wiki_dir": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None},
        },
        "type": "object",
    },
    "lint_format": {
        "additionalProperties": False,
        "properties": {
            "check": {"default": False, "type": "boolean"},
            "config_file": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None},
            "wiki_dir": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None},
        },
        "type": "object",
    },
    "lint_gaps": {
        "additionalProperties": False,
        "properties": {
            "config_file": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None},
            "json_output": {"default": False, "type": "boolean"},
            "wiki_dir": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None},
        },
        "type": "object",
    },
    "lint_history": {
        "additionalProperties": False,
        "properties": {
            "config_file": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None},
            "json_output": {"default": False, "type": "boolean"},
            "last": {"default": 10, "type": "integer"},
            "trend": {"default": False, "type": "boolean"},
            "wiki_dir": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None},
        },
        "type": "object",
    },
    "lint_prune": {
        "additionalProperties": False,
        "properties": {
            "config_file": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None},
            "keep_last": {"anyOf": [{"type": "integer"}, {"type": "null"}], "default": None},
            "max_age_days": {"anyOf": [{"type": "integer"}, {"type": "null"}], "default": None},
            "wiki_dir": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None},
        },
        "type": "object",
    },
    "lint_remediate": {
        "additionalProperties": False,
        "properties": {
            "config_file": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None},
            "dry_run": {"default": False, "type": "boolean"},
            "json_output": {"default": False, "type": "boolean"},
            "wiki_dir": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None},
        },
        "type": "object",
    },
    "lint_semantic": {
        "additionalProperties": False,
        "properties": {
            "config_file": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None},
            "wiki_dir": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None},
        },
        "type": "object",
    },
    "lint_show": {
        "additionalProperties": False,
        "properties": {
            "config_file": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None},
            "json_output": {"default": False, "type": "boolean"},
            "run": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None},
            "wiki_dir": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None},
        },
        "type": "object",
    },
    "list_parsers": {"additionalProperties": False, "properties": {"json_output": {"default": False, "type": "boolean"}}, "type": "object"},
    "log": {
        "additionalProperties": False,
        "properties": {
            "config_file": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None},
            "last": {"default": 25, "type": "integer"},
            "wiki_dir": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None},
            "write": {"default": False, "type": "boolean"},
        },
        "type": "object",
    },
    "provenance": {
        "additionalProperties": False,
        "properties": {
            "json_output": {"default": False, "type": "boolean"},
            "page_slugs": {"anyOf": [{"items": {"type": "string"}, "type": "array"}, {"type": "null"}], "default": None},
            "wiki_dir": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None},
            "with_content": {"default": False, "type": "boolean"},
        },
        "type": "object",
    },
    "review": {
        "additionalProperties": False,
        "properties": {
            "action": {"default": "list", "type": "string"},
            "approve_all": {"default": False, "type": "boolean"},
            "json_output": {"default": False, "type": "boolean"},
            "reason": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None},
            "slug": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None},
            "wiki_dir": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None},
        },
        "type": "object",
    },
}

# Sorted key sets of every ``return {...}`` dict literal per tool, AST-extracted
# from the defining source. A payload key rename or removal changes this golden.
GOLDEN_PAYLOAD_KEYS: dict[str, list[list[str]]] = {
    "doctor": [["last_run", "results", "status"], ["last_run", "status", "summary"], ["message", "status"]],
    "explain": [["candidates", "message", "status"], ["message", "status"], ["report", "report_text", "status"]],
    "lint_all": [["failures", "health_score", "passes", "report", "skips", "status"], ["message", "status"]],
    "lint_check": [["fixes_applied", "health_score", "passes", "report_text", "status", "total_checks"], ["message", "status"]],
    "lint_format": [["changed", "message", "status"], ["message", "status"]],
    "lint_gaps": [["based_on", "recommendations", "status"], ["based_on", "recommendations_text", "status"], ["message", "status"]],
    "lint_history": [["entries", "status"], ["entries_text", "status"]],
    "lint_prune": [["message", "removed", "status"], ["message", "status"]],
    "lint_remediate": [["actions", "applied", "dry_run", "status"], ["actions_text", "applied", "based_on", "dry_run", "status"], ["message", "status"]],
    "lint_semantic": [["health_score", "passes", "report_text", "status", "total_checks"], ["message", "status"]],
    "lint_show": [["entry", "status"], ["message", "status"], ["report", "status"]],
    "list_parsers": [["parsers", "status"], ["parsers_text", "status"]],
    "log": [["message", "status"], ["report", "status"]],
    "provenance": [["message", "status"], ["missing", "pages", "status"], ["missing", "pages_text", "status"], ["pages", "status"], ["pages_text", "status"]],
    "review": [["approved", "status"], ["entries", "status"], ["entries_text", "status"], ["message", "status"], ["rejected", "status"]],
}


class TestToolRegistration:
    def test_the_fifteen_quality_tools_are_registered(self) -> None:
        assert set(_registered_tools()) == QUALITY_TOOL_NAMES

    def test_the_payload_pin_sees_every_tool(self) -> None:
        found = {name for name, shape in _payload_key_sets().items() if shape}
        assert found == QUALITY_TOOL_NAMES


class TestInputSchemas:
    @pytest.mark.parametrize("name", sorted(QUALITY_TOOL_NAMES))
    def test_input_schema_is_pinned(self, name: str) -> None:
        tool = _registered_tools()[name]
        assert json.dumps(tool.parameters, sort_keys=True) == json.dumps(GOLDEN_SCHEMAS[name], sort_keys=True)


class TestPayloadShapes:
    @pytest.mark.parametrize("name", sorted(QUALITY_TOOL_NAMES))
    def test_payload_keys_are_pinned(self, name: str) -> None:
        shape = _payload_key_sets()[name]
        assert sorted(list(keys) for keys in shape) == GOLDEN_PAYLOAD_KEYS[name]


class TestLiveSurface:
    def test_list_parsers_reports_the_ingest_parser(self) -> None:
        result = list_parsers(json_output=True)
        assert result["status"] == "ok"
        (parser,) = result["parsers"]
        assert set(parser) == {"name", "type", "extensions", "figure_enrichment", "metadata"}


def _payload_key_sets() -> dict[str, set[tuple[str, ...]]]:
    """Collect the sorted key set of every ``return {...}`` literal per tool."""
    shapes: dict[str, set[tuple[str, ...]]] = {name: set() for name in QUALITY_TOOL_NAMES}
    for source in _quality_sources():
        for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name not in shapes:
                continue
            for stmt in ast.walk(node):
                if isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Dict):
                    keys = [k.value for k in stmt.value.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)]
                    shapes[node.name].add(tuple(sorted(keys)))
    return shapes


def _quality_sources() -> list[Path]:
    """The file(s) defining the quality tools: the module today, the package later."""
    tools_dir = Path(server_module.__file__).parent / "tools"
    module = tools_dir / "quality.py"
    if module.exists():
        return [module]
    return sorted((tools_dir / "quality").glob("*.py"))


def _registered_tools() -> dict[str, Any]:
    tools = asyncio.run(mcp_server.list_tools())
    return {tool.name: tool for tool in tools if tool.name in QUALITY_TOOL_NAMES}
