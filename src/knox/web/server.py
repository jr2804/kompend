"""KNOX FastAPI web server — app setup and serve().

HTML page routes live in :mod:`knox.web.routes_pages`.
API routes live in :mod:`knox.web.routes_api`.
View helpers live in :mod:`knox.web._view_helpers`.
"""

from __future__ import annotations

import webbrowser

import uvicorn
from fastapi import FastAPI

from knox.web.routes_api import router as api_router
from knox.web.routes_pages import router as pages_router

app = FastAPI(title="KNOX Web UI")
app.include_router(pages_router)
app.include_router(api_router)


def serve(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = False) -> None:
    """Start the web server."""
    if open_browser:
        webbrowser.open(f"http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")
