"""Memory MCP tools."""

from pathlib import Path
from typing import Any

from knox.core.config import DEFAULTS
from knox.core.config.keys import DISPLAY_MEMORY_ENTRIES
from knox.core.templates import get_template_loader
from knox.logging import get_logger
from knox.mcp.tools._helpers import _make_memory_store

logger = get_logger(__name__)


def memory_add(
    query: str,
    answer: str,
    feedback: str | None = None,
    wiki_dir: str = ".",
) -> dict[str, Any]:
    """Add a query/answer pair to the memory store for long-term context."""
    try:
        store, actor = _make_memory_store(wiki_dir)
        store.add_entry(query=query, answer=answer, feedback=feedback, memory_by=actor)
        store.close()
        return {"status": "ok", "message": "Memory entry added successfully."}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


def memory_search(
    query: str,
    top_k: int = 5,
    wiki_dir: str = ".",
) -> dict[str, Any]:
    """Search past memory entries semantically by query."""
    try:
        store, _actor = _make_memory_store(wiki_dir)
        results = store.search(query=query, top_k=top_k)
        store.close()
        return {"status": "ok", "results": results}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


def memory_synthesize(
    limit: int = 20,
    wiki_dir: str = ".",
) -> dict[str, Any]:
    """Synthesize recent memory entries into a system context file."""
    try:
        store, _actor = _make_memory_store(wiki_dir)
        entries = store.get_recent_entries(limit=limit)
        if not entries:
            store.close()
            return {"status": "ok", "message": "No memory entries found to synthesize."}

        base_dir = Path(wiki_dir)
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
