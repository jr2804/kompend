#!/usr/bin/env python
"""Check parity between kompend CLI commands and MCP server tools.

This script owns two contracts:

1. **Names** — every functional kompend CLI command has a matching MCP server
   tool, so the same functionality is reachable over the MCP surface.
2. **Shapes** — the quality tool family's input schema and output payload
   shape are pinned in ``scripts/checks/snapshots/mcp_tool_surface.json``. A
   parameter rename, a changed default, or a renamed payload key fails the
   gate until the snapshot is regenerated deliberately:

       uv run python scripts/checks/check_mcp_parity.py --update-snapshot

   Review the snapshot diff in the same commit as the tool change. The gate
   exits 0 (clean), 1 (a shape or name drifted), or 2 (it could not judge —
   no tools, no snapshot, no output shape).

Design notes
------------
* CLI sub-commands use spaces ("config get"); MCP tools use underscores
  ("config_get"). We normalize both by replacing spaces with underscores.
* The CLI groups the "knowledge loop" under top-level verbs (``compile``,
  ``capture``, ``query``, ``status``, ``init``, ...) while the MCP server
  prefixes those same verbs with ``wiki_`` (``wiki_compile``, ``wiki_query``,
  ...). We strip the ``wiki_`` prefix before comparing so the two naming
  conventions are treated as equivalent.
* A few CLI commands are *server bootstrap* or *interactive* commands that
  have no sensible MCP equivalent (you don't launch a server from inside a
  server, and interactive REPLs / lifecycle commands are out of scope). They
  are excluded via ``EXCLUDED_CLI_COMMANDS`` and listed at the end of the run.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from _common import repo_root, require_facts

from kompend.cli.app import app as cli_app
from kompend.cli.app import register_all_commands
from kompend.mcp.server import server as mcp_server

# CLI commands that intentionally have NO MCP tool equivalent.
# These are server-bootstrap (mcp/serve) or interactive/lifecycle (agent/agents)
# commands.
EXCLUDED_CLI_COMMANDS: set[str] = {
    "mcp",
    "serve",
    "agent",
    "agents",
}

# MCP tool name → the CLI command it covers, when the names cannot be derived
# by the normalization rules above (e.g. a prefix/suffix rename).
# Empty by default; maintained manually if a deliberate rename occurs.
MCP_TO_CLI_ALIASES: dict[str, str] = {
    # wiki_list_attachments covers `attachments list` but uses the
    # knowledge-loop `wiki_` prefix + noun-first ordering rather than the
    # `attachments list` group/verb ordering.
    "wiki_list_attachments": "attachments_list",
}


# CLI command groups that are aliases of another group and must not be
# reported as separate commands (Cyclopts lists each group under every name).
CLI_COMMAND_ALIASES: dict[str, str] = {"att": "attachments"}

#: The quality family whose wire shapes are pinned. The set is explicit rather
#: than "every tool on the server" so adding an unrelated tool does not churn
#: the snapshot; add a name here when its contract should be gated.
SNAPSHOT_TOOL_NAMES: frozenset[str] = frozenset(
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
        "review",
        "provenance",
        "log",
        "list_parsers",
    }
)

SNAPSHOT_SCHEMA_VERSION: int = 1
SNAPSHOT_PATH: Path = Path(__file__).resolve().parent / "snapshots" / "mcp_tool_surface.json"


def main(argv: list[str] | None = None) -> int:
    """Run the gate; ``argv=None`` means "no options" (see ``__main__``).

    Never falls back to ``sys.argv``: this function is imported and called
    from the dev-checks tests, where the process argv is pytest's own command
    line and argparse would reject it as an unknown option.
    """
    parser = argparse.ArgumentParser(description="Check CLI↔MCP name parity and the MCP shape snapshot.")
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=SNAPSHOT_PATH,
        help=f"snapshot file to compare against (default: {SNAPSHOT_PATH})",
    )
    parser.add_argument(
        "--update-snapshot",
        action="store_true",
        help="rewrite the snapshot from the live server, then exit 0",
    )
    args = parser.parse_args([] if argv is None else argv)

    if not args.update_snapshot and not args.snapshot.is_file():
        print(
            f"ERROR: no MCP shape snapshot at {args.snapshot}. Generate it with: uv run python {Path(__file__).name} --update-snapshot",
            file=sys.stderr,
        )
        return 2

    names_code = _check_names()

    print("\nChecking MCP tool shape snapshot...")
    try:
        live = build_snapshot()
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 2

    if args.update_snapshot:
        write_snapshot(live, args.snapshot)
        print(f"Wrote {args.snapshot} ({len(live['tools'])} tools).")
        return 0

    try:
        expected = load_snapshot(args.snapshot)
    except (OSError, ValueError) as exc:
        print(f"ERROR: cannot read the MCP shape snapshot at {args.snapshot}: {exc}", file=sys.stderr)
        return 2

    findings = compare_snapshots(expected, live)
    if findings:
        print(f"\nMCP SHAPE DRIFT ({len(findings)} finding(s)):")
        for finding in findings:
            print(f"  - {finding}")
        print(
            f"\nIf the change is intended, regenerate the snapshot and review its diff:\n  uv run python {Path(__file__).name} --update-snapshot",
            file=sys.stderr,
        )
        return 1

    print(f"OK: {len(live['tools'])} tool shape(s) match {args.snapshot.name}.")
    if names_code == 2:
        return 2
    return names_code


def _check_names() -> int:
    """The original CLI↔MCP name parity check."""
    print("Checking kompend CLI vs MCP Parity...")

    cli_commands = get_cli_commands()
    mcp_tools = get_mcp_tools()
    # An empty set on either side turns this into "compare nothing to nothing",
    # which still prints Full Parity achieved. It has happened before: fastmcp
    # 4.x dropped the private tool store and every CLI command looked missing.
    try:
        require_facts(len(cli_commands), "CLI commands", "Check that register_all_commands() ran and resolved sub-commands.")
        require_facts(len(mcp_tools), "MCP tools", "Check the FastMCP server: list_tools() returned nothing.")
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 2

    # Map normalized → original CLI path for reporting.
    cli_norm_to_full: dict[str, str] = {}
    for full, norm in cli_commands:
        cli_norm_to_full.setdefault(norm, full)

    cli_norm = set(cli_norm_to_full) - EXCLUDED_CLI_COMMANDS
    mcp_norm = {_normalize_mcp_name(t) for t in mcp_tools}

    # Apply explicit aliases (MCP tool → CLI normalized name).
    alias_hits: dict[str, str] = {}
    for mcp_tool, cli_cmd in MCP_TO_CLI_ALIASES.items():
        mcp_norm.discard(_normalize_mcp_name(mcp_tool))
        if cli_cmd in cli_norm:
            cli_norm.discard(cli_cmd)
            alias_hits[mcp_tool] = cli_cmd

    print(f"Found {len(cli_commands)} CLI commands ({len(EXCLUDED_CLI_COMMANDS)} excluded from parity)")
    print(f"Found {len(mcp_tools)} MCP tools")

    missing = sorted(cli_norm - mcp_norm)

    if not missing:
        print("\nOK: Full Parity achieved!")
        if alias_hits:
            print("\nAliased coverage:")
            for mcp_tool, cli_cmd in sorted(alias_hits.items()):
                print(f"  - {mcp_tool} -> {cli_cmd}")
        print("\nExcluded CLI commands (no MCP equivalent by design):")
        for c in sorted(EXCLUDED_CLI_COMMANDS):
            print(f"  - {c}")
        return 0

    print("\nMISSING MCP Tools (available in CLI but not MCP):")
    for m in missing:
        orig = cli_norm_to_full.get(m, m)
        print(f"  - {m}  (CLI: {orig})")

    # Report excluded for transparency
    print("\nExcluded CLI commands (no MCP equivalent by design):")
    for c in sorted(EXCLUDED_CLI_COMMANDS):
        print(f"  - {c}")

    return 1


def build_snapshot() -> dict[str, Any]:
    """Build the current MCP tool-shape snapshot from the live server and source."""
    tools = _tool_map()
    payload_shapes = _payload_shapes()

    require_facts(len(tools), "MCP tools", "Check the FastMCP server: list_tools() returned nothing.")

    missing = sorted(SNAPSHOT_TOOL_NAMES - set(tools))
    if missing:
        print(
            f"ERROR: the live MCP server is missing pinned tool(s): {', '.join(missing)}. "
            "Add them to the server or remove them from SNAPSHOT_TOOL_NAMES deliberately.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    absent_shapes = sorted(name for name in SNAPSHOT_TOOL_NAMES if not payload_shapes.get(name))
    if absent_shapes:
        print(
            f"ERROR: no return-dict shape found for {', '.join(absent_shapes)}. The output shape must be discoverable from the tool source.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    return {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "tools": {
            name: {
                "input_schema": _normalize_schema(tools[name].parameters),
                "output_schema": _normalize_schema(tools[name].output_schema),
                "payload_keys": payload_shapes[name],
            }
            for name in sorted(SNAPSHOT_TOOL_NAMES)
        },
    }


def load_snapshot(path: Path = SNAPSHOT_PATH) -> dict[str, Any]:
    """Load and validate a snapshot file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
        msg = f"unsupported snapshot schema_version in {path}"
        raise ValueError(msg)
    tools = data.get("tools")
    if not isinstance(tools, dict) or not tools:
        msg = f"snapshot {path} contains no tools"
        raise ValueError(msg)
    return data


def write_snapshot(snapshot: dict[str, Any], path: Path = SNAPSHOT_PATH) -> None:
    """Write *snapshot* in the canonical, diff-friendly form."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def compare_snapshots(expected: dict[str, Any], actual: dict[str, Any]) -> list[str]:
    """Return one finding per drifted tool shape; empty when the surfaces match."""
    expected_tools = expected.get("tools", {})
    actual_tools = actual.get("tools", {})
    findings: list[str] = []

    for name in sorted(SNAPSHOT_TOOL_NAMES - set(actual_tools)):
        findings.append(f"{name}: missing from the live server")
    for name in sorted(set(actual_tools) - set(expected_tools)):
        findings.append(f"{name}: live tool is not in the snapshot")

    for name in sorted(SNAPSHOT_TOOL_NAMES & set(expected_tools) & set(actual_tools)):
        before = expected_tools[name]
        after = actual_tools[name]
        for field in ("input_schema", "output_schema", "payload_keys"):
            if before.get(field) != after.get(field):
                findings.append(f"{name}: {field} changed")

    return findings


def _tool_map() -> dict[str, Any]:
    """The live server's tools, keyed by name."""
    return {tool.name: tool for tool in asyncio.run(mcp_server.list_tools())}


def _payload_shapes() -> dict[str, list[list[str]]]:
    """Sorted key sets of every ``return {...}`` literal per pinned tool.

    ``fastmcp`` reports ``output_schema`` as the generic
    ``{"type": "object"}`` for these ``dict[str, Any]`` tools, so the snapshot
    also pins the concrete payload keys, read from the defining source.
    """
    shapes: defaultdict[str, set[tuple[str, ...]]] = defaultdict(set)
    for source in _quality_sources():
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name not in SNAPSHOT_TOOL_NAMES:
                continue
            for stmt in ast.walk(node):
                if isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Dict):
                    keys = [k.value for k in stmt.value.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)]
                    shapes[node.name].add(tuple(sorted(keys)))
    return {name: sorted(list(keys) for keys in shapes[name]) for name in sorted(SNAPSHOT_TOOL_NAMES)}


def _quality_sources() -> list[Path]:
    """The file(s) defining the quality tools: the module today, the package later."""
    tools_dir = repo_root() / "src" / "kompend" / "mcp" / "tools"
    module = tools_dir / "quality.py"
    if module.exists():
        return [module]
    return sorted((tools_dir / "quality").glob("*.py"))


def _normalize_schema(schema: dict[str, Any] | None) -> dict[str, Any] | None:
    """Round-trip through JSON so the snapshot compares structure, not objects."""
    if schema is None:
        return None
    normalized: dict[str, Any] = json.loads(json.dumps(schema, sort_keys=True))
    return normalized


def _normalize_mcp_name(name: str) -> str:
    """Normalize an MCP tool name to its comparable form.

    Strips the ``wiki_`` prefix so CLI ``compile`` and MCP ``wiki_compile``
    match.
    """
    n = name
    if n.startswith("wiki_"):
        n = n[len("wiki_") :]
    return n


def get_cli_commands() -> list[tuple[str, str]]:
    """Extract all command names from the Cyclopts app with their full path.

    Returns a list of (cli_full_path, normalized_for_compare) tuples. Group
    commands are expanded one level into ``group verb``; the pseudo commands
    Cyclopts registers for ``--help``/``--version`` are skipped.
    """
    register_all_commands()
    commands: list[tuple[str, str]] = []

    for name, command in cli_app.resolved_commands().items():
        if name.startswith("-"):
            continue
        if name in CLI_COMMAND_ALIASES:
            continue
        resolved = getattr(command, "resolved_commands", None)
        children = [child for child in (resolved() if resolved else {}) if not child.startswith("-")]
        if children:
            for child in sorted(children):
                commands.append((f"{name} {child}", _normalize_cli_name(f"{name} {child}")))
        else:
            commands.append((name, _normalize_cli_name(name)))

    return commands


def _normalize_cli_name(name: str) -> str:
    """Normalize a CLI command path to its comparable form.

    * spaces → underscores
    * hyphens → underscores
    * strip leading/trailing underscores
    """
    return name.replace(" ", "_").replace("-", "_").strip("_")


def get_mcp_tools() -> set[str]:
    """Extract all tool names registered on the FastMCP server.

    ``fastmcp`` 4.x dropped the private ``_tool_manager._tools`` store, so the
    old accessor returned an empty set - every CLI command then looked missing
    and the gate failed for the wrong reason. Use the public async API.
    """
    tools = asyncio.run(mcp_server.list_tools())
    return {tool.name for tool in tools}


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
