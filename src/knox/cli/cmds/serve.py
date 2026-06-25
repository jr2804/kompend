"""Serve and MCP server commands."""

from __future__ import annotations

from knox.cli.app import app
from knox.cli.args import ServeHost, ServeOpen, ServePort
from knox.logging import get_console, get_err_console
from knox.mcp.server import main as mcp_main
from knox.web.server import serve


@app.command(name="mcp")
def mcp_cmd() -> None:
    """Start the KNOX MCP server over stdio transport.

    Connects to Claude Code, Cursor, and other MCP-compatible clients.
    Uses the official FastMCP SDK with JSON-RPC 2.0 protocol.

    Examples:
        knox mcp                        # start stdio server
        knox mcp --help                 # this help
    """
    mcp_main()


@app.command(name="serve")
def serve_cmd(
    host: ServeHost = "127.0.0.1",
    port: ServePort = 8765,
    open_browser: ServeOpen = False,
) -> None:
    """Start the KNOX web UI server.

    Opens a browser-based dashboard for browsing, searching, and managing
    the compiled wiki. Binds to localhost by default.

    Examples:
        knox serve
        knox serve --port 8080 --open
    """
    get_console().print(f"Starting KNOX web UI at http://{host}:{port}")
    if host != "127.0.0.1":
        get_err_console().print("Warning: Binding to non-localhost exposes the server to the network.")
    serve(host=host, port=port, open_browser=open_browser)
