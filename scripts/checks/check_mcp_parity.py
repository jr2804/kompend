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
    """Extract all command names from the Typer app with their full path.

    Returns a list of (cli_full_path, normalized_for_compare) tuples.
    """
    register_all_commands()
    commands: list[tuple[str, str]] = []

    # Top-level commands
    for cmd in cli_app.registered_commands:
        full = cmd.name
        commands.append((full, _normalize_cli_name(full)))

    # Sub-apps (groups)
    for group in cli_app.registered_groups:
        sub_app = group.typer_instance
        group_name = sub_app.info.name or str(group.name)
        group_norm = _normalize_cli_name(group_name)

        for cmd in sub_app.registered_commands:
            full = f"{group_name} {cmd.name}"
            norm = f"{group_norm}_{_normalize_cli_name(cmd.name)}"
            commands.append((full, norm))

        # Sub-sub-groups (nested)
        for sub_group in sub_app.registered_groups:
            sub_app2 = sub_group.typer_instance
            sub_name = sub_app2.info.name or str(sub_group.name)
            sub_norm = _normalize_cli_name(sub_name)
            for cmd in sub_app2.registered_commands:
                full = f"{group_name} {sub_name} {cmd.name}"
                norm = f"{group_norm}_{sub_norm}_{_normalize_cli_name(cmd.name)}"
                commands.append((full, norm))

    return commands


def _normalize_cli_name(name: str) -> str:
    """Normalize a CLI command path to its comparable form.

    * spaces → underscores
    * hyphens → underscores
    * strip leading/trailing underscores
    """
    return name.replace(" ", "_").replace("-", "_").strip("_")


def get_mcp_tools() -> set[str]:
    """Extract all tool names registered on the FastMCP server."""
    if hasattr(mcp_server, "_tool_manager") and hasattr(mcp_server._tool_manager, "_tools"):
        return set(mcp_server._tool_manager._tools.keys())
    return set()


if __name__ == "__main__":
    sys.exit(main())
