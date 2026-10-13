"""Serve and MCP server commands.

Heavy imports (the MCP server stack with its LLM SDKs, the web server)
live inside the command bodies: this module is imported eagerly by the
CLI so Typer can list every command in help — it must stay cheap to
import.
"""

from __future__ import annotations

from kompend.cli.app import app
from kompend.cli.args import ServeHost, ServeOpen, ServePort, WikiDir
from kompend.logging import get_console, get_err_console
from kompend.mcp.server import main as mcp_main
from kompend.web.server import serve


@app.command(name="mcp")
def mcp_cmd(
    wiki_dir: WikiDir = None,
) -> None:
    """Start the kompend MCP server over stdio transport.

    Connects to Claude Code, Cursor, and other MCP-compatible clients.
    Uses the Prefect fastmcp framework (built on official mcp SDK 2.x)
    with JSON-RPC 2.0 protocol.

    Examples:
        kompend mcp                        # start stdio server (wiki = cwd)
        kompend mcp --wiki-dir ./my-wiki   # point at a specific wiki
        kompend mcp -w /path/to/wiki       # short form
    """
    mcp_main(wiki_dir=wiki_dir)


@app.command(name="serve")
def serve_cmd(
    host: ServeHost = "127.0.0.1",
    port: ServePort = 8765,
    open_browser: ServeOpen = False,
) -> None:
    """Start the kompend web UI server.

    Opens a browser-based dashboard for browsing, searching, and managing
    the compiled wiki. Binds to localhost by default.

    Examples:
        kompend serve
        kompend serve --port 8080 --open
    """
    get_console().print(f"Starting kompend web UI at http://{host}:{port}")
    if host != "127.0.0.1":
        get_err_console().print("Warning: Binding to non-localhost exposes the server to the network.")

    serve(host=host, port=port, open_browser=open_browser)
