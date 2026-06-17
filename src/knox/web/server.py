import json
import logging
import subprocess
import sys
import webbrowser
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse

from knox.capture.ingest import ingest
from knox.capture.learn import learn
from knox.core.config._core import DEFAULTS
from knox.core.models import CompiledWiki
from knox.core.paths import KNOX_DIR, WIKI_CACHE_FILE
from knox.search.classifier import classify
from knox.search.retriever import TokenOverlapRetriever

logger = logging.getLogger(__name__)

app = FastAPI(title="KNOX Web UI")

_PROJECT_DIR: Path = Path.cwd()


def _source_of(node: dict[str, Any]) -> str:
    """Extract source file from a wiki node's metadata, or '' if absent."""
    meta = node.get("metadata") or {}
    return meta.get("source", "")


def _load_wiki() -> dict[str, Any]:
    cache = _PROJECT_DIR / DEFAULTS["paths"]["output_dir"] / KNOX_DIR / WIKI_CACHE_FILE
    if not cache.exists():
        return {"nodes": [], "edges": [], "toc": [], "quality_score": 0.0, "node_count": 0}
    try:
        data = json.loads(cache.read_text(encoding="utf-8"))
        data["node_count"] = len(data.get("nodes", []))
        return data
    except Exception as exc:
        logger.warning("Failed to load wiki cache: %s", exc, exc_info=True)
        return {"nodes": [], "edges": [], "toc": [], "quality_score": 0.0, "node_count": 0}


@app.get("/", response_class=HTMLResponse)
async def dashboard() -> str:
    wiki = _load_wiki()
    nodes = wiki.get("nodes", [])
    edges = wiki.get("edges", [])
    toc = wiki.get("toc", [])
    source_count = len({_source_of(n) for n in nodes if _source_of(n)})
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>KNOX Dashboard</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f5f5; color: #333; }}
  nav {{ background: #1a1a2e; color: #fff; padding: 1rem 2rem; display: flex; gap: 2rem; align-items: center; }}
  nav a {{ color: #a0a0cc; text-decoration: none; }}
  nav a:hover {{ color: #fff; }}
  nav .brand {{ font-weight: 700; font-size: 1.2rem; color: #fff; }}
  .container {{ max-width: 1200px; margin: 2rem auto; padding: 0 1rem; }}
  .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 2rem; }}
  .card {{ background: #fff; border-radius: 8px; padding: 1.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  .card h3 {{ font-size: 0.85rem; text-transform: uppercase; color: #666; margin-bottom: 0.5rem; }}
  .card .value {{ font-size: 2rem; font-weight: 700; color: #1a1a2e; }}
  .section {{ background: #fff; border-radius: 8px; padding: 1.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 1rem; }}
  .section h2 {{ margin-bottom: 1rem; font-size: 1.1rem; color: #1a1a2e; }}
  table {{ width: 100%; border-collapse: collapse; }}
  th, td {{ text-align: left; padding: 0.5rem; border-bottom: 1px solid #eee; }}
  th {{ font-size: 0.8rem; text-transform: uppercase; color: #666; }}
  tr:hover {{ background: #f9f9f9; }}
</style>
</head>
<body>
<nav>
  <span class="brand">KNOX</span>
  <a href="/">Dashboard</a>
  <a href="/wiki">Wiki</a>
  <a href="/query">Query</a>
  <a href="/graph">Graph</a>
  <a href="/ingest">Ingest</a>
  <a href="/lint">Lint</a>
  <a href="/settings">Settings</a>
</nav>
<div class="container">
  <div class="cards">
    <div class="card"><h3>Wiki Pages</h3><div class="value">{wiki["node_count"]}</div></div>
    <div class="card"><h3>Sources</h3><div class="value">{source_count}</div></div>
    <div class="card"><h3>Cross-References</h3><div class="value">{len(edges)}</div></div>
    <div class="card"><h3>Quality Score</h3><div class="value">{wiki.get("quality_score", 0):.0%}</div></div>
    <div class="card"><h3>TOC Entries</h3><div class="value">{len(toc)}</div></div>
  </div>
  <div class="section">
    <h2>Recent Wiki Pages</h2>
    <table>
      <tr><th>Title</th><th>Type</th><th>Source</th></tr>
      {
        "".join(
            f'<tr><td><a href="/wiki/{n.get("id", "")}">{n.get("title", "")}</a></td>'
            f"<td>{n.get('node_type', '')}</td>"
            f"<td>{_source_of(n)}</td></tr>"
            for n in nodes[:10]
        )
    }
    </table>
  </div>
</div>
</body>
</html>"""


@app.get("/wiki", response_class=HTMLResponse)
async def wiki_list() -> str:
    wiki = _load_wiki()
    nodes = wiki.get("nodes", [])
    rows = "".join(
        f'<tr><td><a href="/wiki/{n.get("id", "")}">{n.get("title", "")}</a></td>'
        f"<td>{n.get('node_type', '')}</td>"
        f"<td>{_source_of(n)}</td></tr>"
        for n in nodes
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>KNOX Wiki</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, sans-serif; background: #f5f5f5; color: #333; }}
  nav {{ background: #1a1a2e; color: #fff; padding: 1rem 2rem; display: flex; gap: 2rem; }}
  nav a {{ color: #a0a0cc; text-decoration: none; }}
  nav a:hover {{ color: #fff; }}
  nav .brand {{ font-weight: 700; color: #fff; }}
  .container {{ max-width: 1200px; margin: 2rem auto; padding: 0 1rem; }}
  table {{ width: 100%; background: #fff; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  th, td {{ padding: 0.75rem; text-align: left; border-bottom: 1px solid #eee; }}
  th {{ background: #f9f9f9; font-size: 0.8rem; text-transform: uppercase; color: #666; }}
  tr:hover {{ background: #f0f0ff; }}
  a {{ color: #1a1a2e; text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
</style>
</head>
<body>
<nav>
  <span class="brand">KNOX</span>
  <a href="/">Dashboard</a>
  <a href="/wiki">Wiki</a>
  <a href="/query">Query</a>
  <a href="/graph">Graph</a>
  <a href="/ingest">Ingest</a>
  <a href="/lint">Lint</a>
  <a href="/settings">Settings</a>
</nav>
<div class="container">
  <h2 style="margin-bottom:1rem">Wiki Pages ({len(nodes)})</h2>
  <table><tr><th>Title</th><th>Type</th><th>Source</th></tr>{rows}</table>
</div>
</body>
</html>"""


@app.get("/wiki/{node_id}", response_class=HTMLResponse)
async def wiki_page(node_id: str) -> str:
    wiki = _load_wiki()
    node = next((n for n in wiki.get("nodes", []) if n.get("id") == node_id), None)
    if not node:
        return HTMLResponse("<h1>Not Found</h1>", status_code=404)
    content = node.get("content", "")
    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>{node.get("title", "")} - KNOX</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, sans-serif; background: #f5f5f5; color: #333; }}
  nav {{ background: #1a1a2e; color: #fff; padding: 1rem 2rem; display: flex; gap: 2rem; }}
  nav a {{ color: #a0a0cc; text-decoration: none; }}
  nav a:hover {{ color: #fff; }}
  .container {{ max-width: 800px; margin: 2rem auto; padding: 0 1rem; }}
  .content {{ background: #fff; border-radius: 8px; padding: 2rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1); line-height: 1.6; }}
  .meta {{ font-size: 0.85rem; color: #666; margin-bottom: 1rem; }}
</style>
</head>
<body>
<nav><a href="/">KNOX</a><a href="/wiki">Wiki</a></nav>
<div class="container">
  <div class="content">
    <h1>{node.get("title", "")}</h1>
    <div class="meta">Type: {node.get("node_type", "")} | Source: {_source_of(node)}</div>
    <pre style="white-space:pre-wrap;font-family:inherit">{content}</pre>
  </div>
</div>
</body>
</html>"""


@app.get("/query", response_class=HTMLResponse)
async def query_page(q: str = Query(default="")) -> str:
    wiki = _load_wiki()
    answer_html = ""
    if q:
        w = CompiledWiki(
            nodes=wiki.get("nodes", []),
            edges=wiki.get("edges", []),
            toc=wiki.get("toc", []),
            quality_score=wiki.get("quality_score", 0.0),
            iteration=wiki.get("iteration", 0),
            metadata=wiki.get("metadata", {}),
        )
        retriever = TokenOverlapRetriever(w)
        results = retriever.search(q, top_k=5)

        intent = classify(q)
        if intent != "knowledge":
            answer_html = f"<p><em>Intent: {intent}</em></p><p>Hello! I can answer questions about your compiled wiki. Try asking something specific.</p>"
        elif results:
            answer_html = "<h3>Results</h3>"
            for r in results:
                answer_html += '<div style="margin-bottom:1rem;padding:1rem;background:#f9f9f9;border-radius:4px">'
                answer_html += f'<strong>{r.title}</strong> <span style="color:#666;font-size:0.85rem">(score: {r.score:.3f})</span>'
                if r.source_file:
                    answer_html += f'<br><span style="font-size:0.85rem;color:#666">Source: {r.source_file}</span>'
                answer_html += f"<p>{r.snippet}</p></div>"
        else:
            answer_html = "<p>No relevant content found.</p>"

    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>KNOX Query</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, sans-serif; background: #f5f5f5; color: #333; }}
  nav {{ background: #1a1a2e; color: #fff; padding: 1rem 2rem; display: flex; gap: 2rem; }}
  nav a {{ color: #a0a0cc; text-decoration: none; }}
  nav a:hover {{ color: #fff; }}
  .container {{ max-width: 800px; margin: 2rem auto; padding: 0 1rem; }}
  form {{ display: flex; gap: 0.5rem; margin-bottom: 2rem; }}
  input[type=text] {{ flex: 1; padding: 0.75rem; border: 1px solid #ddd; border-radius: 4px; font-size: 1rem; }}
  button {{ padding: 0.75rem 1.5rem; background: #1a1a2e; color: #fff; border: none; border-radius: 4px; cursor: pointer; }}
  .results {{ background: #fff; border-radius: 8px; padding: 1.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
</style>
</head>
<body>
<nav>
  <span class="brand">KNOX</span>
  <a href="/">Dashboard</a>
  <a href="/wiki">Wiki</a>
  <a href="/query">Query</a>
  <a href="/graph">Graph</a>
  <a href="/ingest">Ingest</a>
  <a href="/lint">Lint</a>
  <a href="/settings">Settings</a>
</nav>
<div class="container">
  <h2 style="margin-bottom:1rem">Query the Wiki</h2>
  <form method="get" action="/query">
    <input type="text" name="q" placeholder="Ask a question..." value="{q}">
    <button type="submit">Search</button>
  </form>
  <div class="results">{answer_html}</div>
</div>
</body>
</html>"""


@app.get("/graph", response_class=HTMLResponse)
async def force_graph() -> str:
    wiki = _load_wiki()
    nodes_json = json.dumps([{"id": n.get("id"), "title": n.get("title"), "type": n.get("node_type")} for n in wiki.get("nodes", [])])
    edges_json = json.dumps(list(wiki.get("edges", [])))
    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>KNOX Graph</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, sans-serif; background: #f5f5f5; color: #333; }}
  nav {{ background: #1a1a2e; color: #fff; padding: 1rem 2rem; display: flex; gap: 2rem; }}
  nav a {{ color: #a0a0cc; text-decoration: none; }}
  nav a:hover {{ color: #fff; }}
  .container {{ max-width: 1200px; margin: 2rem auto; padding: 0 1rem; }}
  svg {{ background: #fff; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); width: 100%; height: 600px; }}
</style>
<script src="https://d3js.org/d3.v7.min.js"></script>
</head>
<body>
<nav>
  <span class="brand">KNOX</span>
  <a href="/">Dashboard</a>
  <a href="/wiki">Wiki</a>
  <a href="/query">Query</a>
  <a href="/graph">Graph</a>
  <a href="/ingest">Ingest</a>
  <a href="/lint">Lint</a>
  <a href="/settings">Settings</a>
</nav>
<div class="container">
  <h2 style="margin-bottom:1rem">Wiki Graph</h2>
  <svg id="graph"></svg>
</div>
<script>
const nodes = {nodes_json};
const edges = {edges_json};
const svg = d3.select("#graph"),
    width = svg.node().getBoundingClientRect().width,
    height = 600;
svg.attr("viewBox", [0, 0, width, height]);
const link = svg.append("g").selectAll("line").data(edges).join("line")
    .attr("stroke", "#ccc").attr("stroke-width", 1.5);
const node = svg.append("g").selectAll("g").data(nodes).join("g")
    .call(d3.drag().on("start", (e,d) => {{ if(!e.active) sim.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; }})
        .on("drag", (e,d) => {{ d.fx = e.x; d.fy = e.y; }})
        .on("end", (e,d) => {{ if(!e.active) sim.alphaTarget(0); d.fx = null; d.fy = null; }}));
node.append("circle").attr("r", 8).attr("fill", "#1a1a2e");
node.append("text").text(d => d.title).attr("x", 12).attr("y", 4).attr("font-size", "12px");
const sim = d3.forceSimulation(nodes).force("link", d3.forceLink(edges).id(d => d.id).distance(100))
    .force("charge", d3.forceManyBody().strength(-200))
    .force("center", d3.forceCenter(width/2, height/2))
    .on("tick", () => {{ link.attr("x1", d => d.source.x).attr("y1", d => d.source.y).attr("x2", d => d.target.x).attr("y2", d => d.target.y);
        node.attr("transform", d => `translate(${{d.x}},${{d.y}})`); }});
</script>
</body>
</html>"""


@app.get("/ingest", response_class=HTMLResponse)
async def ingest_page() -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>KNOX Ingest</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, sans-serif; background: #f5f5f5; color: #333; }
  nav { background: #1a1a2e; color: #fff; padding: 1rem 2rem; display: flex; gap: 2rem; }
  nav a { color: #a0a0cc; text-decoration: none; }
  nav a:hover { color: #fff; }
  .container { max-width: 800px; margin: 2rem auto; padding: 0 1rem; }
  .card { background: #fff; border-radius: 8px; padding: 1.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 1rem; }
  form { display: flex; gap: 0.5rem; }
  input[type=text] { flex: 1; padding: 0.75rem; border: 1px solid #ddd; border-radius: 4px; }
  button { padding: 0.75rem 1.5rem; background: #1a1a2e; color: #fff; border: none; border-radius: 4px; cursor: pointer; }
</style>
</head>
<body>
<nav>
  <span class="brand">KNOX</span>
  <a href="/">Dashboard</a>
  <a href="/wiki">Wiki</a>
  <a href="/query">Query</a>
  <a href="/graph">Graph</a>
  <a href="/ingest">Ingest</a>
  <a href="/lint">Lint</a>
  <a href="/settings">Settings</a>
</nav>
<div class="container">
  <h2 style="margin-bottom:1rem">Ingest Sources</h2>
  <div class="card">
    <h3 style="margin-bottom:0.5rem">Ingest from URL</h3>
    <form action="/api/ingest-url" method="post">
      <input type="text" name="url" placeholder="https://arxiv.org/pdf/..." required>
      <button type="submit">Download & Ingest</button>
    </form>
  </div>
  <div class="card">
    <h3 style="margin-bottom:0.5rem">Quick Learn</h3>
    <form action="/api/learn" method="post">
      <input type="text" name="text" placeholder="The transformer uses d_model=512 and h=8" required>
      <button type="submit">Learn Fact</button>
    </form>
  </div>
  <div class="card">
    <h3 style="margin-bottom:0.5rem">Drag & Drop</h3>
    <p style="color:#666;margin-bottom:0.5rem">Place files in the <code>sources/</code> directory, then run <code>knox compile</code> from the terminal.</p>
  </div>
</div>
</body>
</html>"""


@app.get("/lint", response_class=HTMLResponse)
async def lint_page() -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>KNOX Lint</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, sans-serif; background: #f5f5f5; color: #333; }
  nav { background: #1a1a2e; color: #fff; padding: 1rem 2rem; display: flex; gap: 2rem; }
  nav a { color: #a0a0cc; text-decoration: none; }
  nav a:hover { color: #fff; }
  .container { max-width: 800px; margin: 2rem auto; padding: 0 1rem; }
  .card { background: #fff; border-radius: 8px; padding: 1.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
</style>
</head>
<body>
<nav>
  <span class="brand">KNOX</span>
  <a href="/">Dashboard</a>
  <a href="/wiki">Wiki</a>
  <a href="/query">Query</a>
  <a href="/graph">Graph</a>
  <a href="/ingest">Ingest</a>
  <a href="/lint">Lint</a>
  <a href="/settings">Settings</a>
</nav>
<div class="container">
  <div class="card">
    <h2 style="margin-bottom:0.5rem">Lint the Wiki</h2>
    <p style="color:#666;margin-bottom:1rem">Run <code>knox lint</code> from the terminal to check for issues.</p>
  </div>
</div>
</body>
</html>"""


@app.get("/settings", response_class=HTMLResponse)
async def settings_page() -> str:
    config_path = _PROJECT_DIR / "knox.toml"
    config_content = ""
    if config_path.exists():
        config_content = config_path.read_text(encoding="utf-8")
    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>KNOX Settings</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, sans-serif; background: #f5f5f5; color: #333; }}
  nav {{ background: #1a1a2e; color: #fff; padding: 1rem 2rem; display: flex; gap: 2rem; }}
  nav a {{ color: #a0a0cc; text-decoration: none; }}
  nav a:hover {{ color: #fff; }}
  .container {{ max-width: 800px; margin: 2rem auto; padding: 0 1rem; }}
  .card {{ background: #fff; border-radius: 8px; padding: 1.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  pre {{ background: #f5f5f5; padding: 1rem; border-radius: 4px; overflow-x: auto; font-size: 0.85rem; }}
</style>
</head>
<body>
<nav>
  <span class="brand">KNOX</span>
  <a href="/">Dashboard</a>
  <a href="/wiki">Wiki</a>
  <a href="/query">Query</a>
  <a href="/graph">Graph</a>
  <a href="/ingest">Ingest</a>
  <a href="/lint">Lint</a>
  <a href="/settings">Settings</a>
</nav>
<div class="container">
  <div class="card">
    <h2 style="margin-bottom:0.5rem">Configuration</h2>
    <p style="color:#666;margin-bottom:1rem">{config_path if config_path.exists() else "No knox.toml found"}</p>
    <pre>{config_content or "No configuration file found."}</pre>
  </div>
</div>
</body>
</html>"""


@app.post("/api/ingest-url")
async def api_ingest_url(url: str = Query(...)) -> JSONResponse:
    result = ingest(url, sources_dir=_PROJECT_DIR / DEFAULTS["paths"]["sources_dir"])
    if result:
        return JSONResponse({"status": "ok", "file": str(result)})
    return JSONResponse({"status": "error", "message": f"Failed to ingest {url}"}, status_code=400)


@app.post("/api/learn")
async def api_learn(text: str = Query(...)) -> JSONResponse:
    path = learn(text, sources_dir=_PROJECT_DIR / DEFAULTS["paths"]["sources_dir"])
    return JSONResponse({"status": "ok", "file": str(path)})


@app.post("/api/compile")
async def api_compile() -> JSONResponse:
    result = subprocess.run([sys.executable, "-m", "knox.cli.app", "compile"], capture_output=True, text=True, cwd=str(_PROJECT_DIR), check=False)  # noqa: S603
    if result.returncode == 0:
        return JSONResponse({"status": "ok", "output": result.stdout})
    return JSONResponse({"status": "error", "message": result.stderr or result.stdout}, status_code=400)


def serve(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = False) -> None:
    if open_browser:
        webbrowser.open(f"http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")
