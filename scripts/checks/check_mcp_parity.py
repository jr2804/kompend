#!/usr/bin/env python
"""Check parity between kompend CLI commands and MCP server tools.

This script verifies that every functional kompend CLI command has a matching
MCP server tool so the same functionality is reachable over the MCP surface.

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

import asyncio
import sys

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


def main() -> int:
    print("Checking kompend CLI vs MCP Parity...")

    cli_commands = get_cli_commands()
    mcp_tools = get_mcp_tools()

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
    sys.exit(main())
