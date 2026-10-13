"""Memory MCP tools."""

from pathlib import Path
from typing import Any

from kompend.core.config import DEFAULTS
from kompend.core.config.keys import DISPLAY_MEMORY_ENTRIES
from kompend.core.templates import get_template_loader
from kompend.logging import get_logger
from kompend.mcp.tools._helpers import make_memory_store, resolve_wiki_dir

logger = get_logger(__name__)


def memory_add(
    query: str,
    answer: str,
    feedback: str | None = None,
    wiki_dir: str | None = None,
) -> dict[str, Any]:
    """Add a query/answer pair to the memory store for long-term context."""
    try:
        resolved_dir = resolve_wiki_dir(wiki_dir)
        store, actor = make_memory_store(resolved_dir)
        store.add_entry(query=query, answer=answer, feedback=feedback, memory_by=actor)
        store.close()
        return {"status": "ok", "message": "Memory entry added successfully."}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


def memory_search(
    query: str,
    top_k: int = 5,
    wiki_dir: str | None = None,
) -> dict[str, Any]:
    """Search past memory entries semantically by query."""
    try:
        resolved_dir = resolve_wiki_dir(wiki_dir)
        store, _actor = make_memory_store(resolved_dir)
        results = store.search(query=query, top_k=top_k)
        store.close()
        return {"status": "ok", "results": results}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


def memory_synthesize(
    limit: int = 20,
    wiki_dir: str | None = None,
) -> dict[str, Any]:
    """Synthesize recent memory entries into a system context file."""
    try:
        resolved_dir = resolve_wiki_dir(wiki_dir)
        store, _actor = make_memory_store(resolved_dir)
        entries = store.get_recent_entries(limit=limit)
        if not entries:
            store.close()
            return {"status": "ok", "message": "No memory entries found to synthesize."}

        base_dir = Path(resolved_dir)
        context_file = base_dir / "sources" / "_memory" / "system_context.md"
        context_file.parent.mkdir(parents=True, exist_ok=True)

        loader = get_template_loader()
        content = loader.render(
            "memory/system_context.md.j2",
            entries=entries[: DEFAULTS.get(DISPLAY_MEMORY_ENTRIES, 10)],
        )
        context_file.write_text(content, encoding="utf-8")

        store.close()
        return {"status": "ok", "message": f"Synthesized context written to {context_file}"}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}
