# DOX — knox.mcp

<!-- on entry: loaded when agent enters src/knox/mcp -->
<!-- parent: ../../../AGENTS.md | .agents/POLICIES.md, .agents/FILES.md, .agents/CHECKS.md -->

## Purpose

MCP (Model Context Protocol) server exposing KNOX compilation, query,
memory, and configuration operations as MCP tools. Serves over stdio transport.

## Ownership

- ``src/knox/mcp/`` — FastMCP server, tool implementations

## Architecture

| Module | Contents |
|--------|----------|
| ``server.py`` | ``FastMCP`` instance, tool registration, ``main()`` entry point |
| ``tools/`` | Tool implementation modules |
| ``tools/config.py`` | ``config_get``, ``config_set``, ``config_check`` — wiki config inspection/mutation/diff |
| ``tools/knowledge.py`` | ``wiki_capture``, ``wiki_distill``, ``wiki_learn``, ``query_wiki`` |
| ``tools/memory.py`` | ``memory_add``, ``memory_search``, ``memory_synthesize`` |
| ``tools/query.py`` | ``query_wiki`` — markdown-first search (zvec vectors optional, keyword fallback) |
| ``tools/_helpers.py`` | Shared tool utilities (wiki dir resolution, error formatting) |

## Rules

- Tool implementations go in ``tools/``, not in ``server.py``.
- ``server.py`` registers tools via function reference, not strings.
- Import: ``from knox.mcp.server import server`` for the FastMCP instance.
- The ``main()`` entry point is invoked by the ``knox serve`` CLI command.
