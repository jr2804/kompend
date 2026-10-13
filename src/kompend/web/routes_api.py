"""API routes for the KOMPEND web server."""

from __future__ import annotations

import subprocess
import sys

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from kompend.capture.ingest import ingest
from kompend.capture.learn import learn
from kompend.core.paths import VaultPaths
from kompend.web._view_helpers import project_dir

router = APIRouter(prefix="/api")


@router.post("/ingest-url")
async def api_ingest_url(url: str = Query(...)) -> JSONResponse:
    result = ingest(url, sources_dir=VaultPaths(project_dir()).sources_dir)
    if result:
        return JSONResponse({"status": "ok", "file": str(result)})
    return JSONResponse({"status": "error", "message": f"Failed to ingest {url}"}, status_code=400)


@router.post("/learn")
async def api_learn(text: str = Query(...)) -> JSONResponse:
    path = learn(text, sources_dir=VaultPaths(project_dir()).sources_dir)
    return JSONResponse({"status": "ok", "file": str(path)})


@router.post("/compile")
async def api_compile() -> JSONResponse:
    result = subprocess.run([sys.executable, "-m", "kompend.cli.app", "compile"], capture_output=True, text=True, cwd=str(project_dir()), check=False)  # noqa: S603
    if result.returncode == 0:
        return JSONResponse({"status": "ok", "output": result.stdout})
    return JSONResponse({"status": "error", "message": result.stderr or result.stdout}, status_code=400)
