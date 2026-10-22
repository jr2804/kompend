"""HTML page routes for the kompend web server."""

from __future__ import annotations

import json

from fastapi import APIRouter, Query
from fastapi.responses import HTMLResponse

from kompend.core.models import CompiledWiki, MetadataField, WikiField
from kompend.core.paths import PROJECT_CONFIG_FILE
from kompend.search.classifier import classify
from kompend.search.retriever import HybridRetriever
from kompend.web._view_helpers import display_config, load_wiki, project_dir, render_result, source_of

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def dashboard() -> str:
    wiki = load_wiki()
    nodes = wiki.get(WikiField.NODES, [])
    edges = wiki.get(WikiField.EDGES, [])
    toc = wiki.get(WikiField.TOC, [])
    source_count = len({source_of(n) for n in nodes if source_of(n)})
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>kompend Dashboard</title>
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
  <span class="brand">kompend</span>
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
    <div class="card"><h3>Wiki Pages</h3><div class="value">{wiki[MetadataField.NODE_COUNT]}</div></div>
    <div class="card"><h3>Sources</h3><div class="value">{source_count}</div></div>
    <div class="card"><h3>Cross-References</h3><div class="value">{len(edges)}</div></div>
    <div class="card"><h3>Quality Score</h3><div class="value">{wiki.get(WikiField.QUALITY_SCORE, 0):.0%}</div></div>
    <div class="card"><h3>TOC Entries</h3><div class="value">{len(toc)}</div></div>
  </div>
  <div class="section">
    <h2>Recent Wiki Pages</h2>
    <table>
      <tr><th>Title</th><th>Type</th><th>Source</th></tr>
      {
        "".join(
            f'<tr><td><a href="/wiki/{n.get("id", "")}">{n.get("title", "")}</a></td><td>{n.get("node_type", "")}</td><td>{source_of(n)}</td></tr>'
            for n in nodes[: display_config().web_nodes]
        )
    }
    </table>
  </div>
</div>
</body>
</html>"""


@router.get("/wiki", response_class=HTMLResponse)
async def wiki_list() -> str:
    wiki = load_wiki()
    nodes = wiki.get(WikiField.NODES, [])
    rows = "".join(
        f'<tr><td><a href="/wiki/{n.get("id", "")}">{n.get("title", "")}</a></td><td>{n.get("node_type", "")}</td><td>{source_of(n)}</td></tr>' for n in nodes
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>kompend Wiki</title>
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
  <span class="brand">kompend</span>
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


@router.get("/wiki/{node_id}", response_class=HTMLResponse)
async def wiki_page(node_id: str) -> str:
    wiki = load_wiki()
    node = next((n for n in wiki.get(WikiField.NODES, []) if n.get("id") == node_id), None)
    if not node:
        return "<h1>Not Found</h1>"
    content = node.get("content", "")
    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>{node.get("title", "")} - kompend</title>
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
<nav><a href="/">kompend</a><a href="/wiki">Wiki</a></nav>
<div class="container">
  <div class="content">
    <h1>{node.get("title", "")}</h1>
    <div class="meta">Type: {node.get("node_type", "")} | Source: {source_of(node)}</div>
    <pre style="white-space:pre-wrap;font-family:inherit">{content}</pre>
  </div>
</div>
</body>
</html>"""


@router.get("/query", response_class=HTMLResponse)
async def query_page(q: str = Query(default="")) -> str:
    wiki = load_wiki()
    answer_html = ""
    if q:
        w = CompiledWiki(
            nodes=wiki.get(WikiField.NODES, []),
            edges=wiki.get(WikiField.EDGES, []),
            toc=wiki.get(WikiField.TOC, []),
            quality_score=wiki.get(WikiField.QUALITY_SCORE, 0.0),
            iteration=wiki.get(WikiField.ITERATION, 0),
            metadata=wiki.get(WikiField.METADATA, {}),
        )
        retriever = HybridRetriever(w, display=display_config())
        results = retriever.search(q, top_k=5)

        intent = classify(q)
        if intent != "knowledge":
            answer_html = f"<p><em>Intent: {intent}</em></p><p>Hello! I can answer questions about your compiled wiki. Try asking something specific.</p>"
        elif results:
            answer_html = "<h3>Results</h3>" + "".join(render_result(r) for r in results)
        else:
            answer_html = "<p>No relevant content found.</p>"

    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>kompend Query</title>
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
  <span class="brand">kompend</span>
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


@router.get("/graph", response_class=HTMLResponse)
async def force_graph() -> str:
    wiki = load_wiki()
    nodes_json = json.dumps([{"id": n.get("id"), "title": n.get("title"), "type": n.get("node_type")} for n in wiki.get(WikiField.NODES, [])])
    edges_json = json.dumps(list(wiki.get(WikiField.EDGES, [])))
    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>kompend Graph</title>
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
  <span class="brand">kompend</span>
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


@router.get("/ingest", response_class=HTMLResponse)
async def ingest_page() -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>kompend Ingest</title>
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
  <span class="brand">kompend</span>
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
    <p style="color:#666;margin-bottom:0.5rem">Place files in the <code>sources/</code> directory, then run <code>kompend compile</code> from the terminal.</p>
  </div>
</div>
</body>
</html>"""


@router.get("/lint", response_class=HTMLResponse)
async def lint_page() -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>kompend Lint</title>
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
  <span class="brand">kompend</span>
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
    <p style="color:#666;margin-bottom:1rem">Run <code>kompend lint</code> from the terminal to check for issues.</p>
  </div>
</div>
</body>
</html>"""


@router.get("/settings", response_class=HTMLResponse)
async def settings_page() -> str:
    config_file = project_dir() / PROJECT_CONFIG_FILE
    config_content = ""
    if config_file.exists():
        config_content = config_file.read_text(encoding="utf-8")
    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>kompend Settings</title>
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
  <span class="brand">kompend</span>
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
    <p style="color:#666;margin-bottom:1rem">{config_file if config_file.exists() else "No kompend.toml found"}</p>
    <pre>{config_content or "No configuration file found."}</pre>
  </div>
</div>
</body>
</html>"""
