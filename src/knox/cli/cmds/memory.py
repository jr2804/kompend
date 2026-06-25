"""``knox memory`` — manage query memory and context synthesis."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer

from knox.cli.app import app
from knox.cli.args.memory import MemoryAnswer, MemoryFeedback, MemoryLimit, MemoryQuery, MemoryTopK
from knox.core.config import resolve_config
from knox.core.config.keys import COMPILE_EMBEDDING_MODEL, PATHS_WIKI_DIR, TRUST_USER_TRACKING
from knox.core.template_loader import get_template_loader
from knox.core.user import resolve_actor
from knox.logging import get_console, get_err_console
from knox.memory.store import MemoryStore

memory_app = typer.Typer(
    name="memory",
    help="Manage query memory and context synthesis.",
    no_args_is_help=True,
)
app.add_typer(memory_app)


def _get_memory_store(ctx: typer.Context | None) -> tuple[MemoryStore, Path, str]:
    """Resolve the memory store, project base directory, and acting user."""
    overrides: dict[str, Any] = {}
    if ctx and ctx.obj:
        if ctx.obj.get("wiki_dir"):
            overrides[PATHS_WIKI_DIR] = ctx.obj["wiki_dir"]
        if ctx.obj.get("no_user_tracking"):
            overrides[TRUST_USER_TRACKING] = False
    cfg = resolve_config(cli_overrides=overrides)
    paths = cfg.get("paths")
    output_dir = Path(paths["output_dir"])
    base_dir = output_dir.parent
    embed_model = cfg.get(COMPILE_EMBEDDING_MODEL)
    return MemoryStore(base_dir=base_dir, embedding_model=embed_model), base_dir, resolve_actor(cfg)


@memory_app.command("add", help="Add a query/answer pair to the memory log.")
def memory_add(
    ctx: typer.Context,
    query: MemoryQuery,
    answer: MemoryAnswer,
    feedback: MemoryFeedback = None,
) -> None:
    """Add a query, answer, and optional feedback to the memory store."""
    store, _base_dir, actor = _get_memory_store(ctx)
    try:
        store.add_entry(query=query, answer=answer, feedback=feedback, memory_by=actor)
        get_console().print("✓ Memory entry added successfully.")
    except Exception as exc:
        get_err_console().print(f"Error adding memory entry: {exc}")
        raise typer.Exit(1)
    finally:
        store.close()


@memory_app.command("search", help="Search past memory entries semantically.")
def memory_search(
    ctx: typer.Context,
    query: MemoryQuery,
    top_k: MemoryTopK = 5,
) -> None:
    """Search the memory store for past queries and answers."""
    store, _base_dir, _actor = _get_memory_store(ctx)
    try:
        results = store.search(query=query, top_k=top_k)
        if not results:
            get_console().print("No relevant memory entries found.")
            return

        get_console().print(f"Found {len(results)} relevant memory entries:\n")
        for i, res in enumerate(results, 1):
            get_console().print(f"[{i}] Query: {res['query']}", markup=False)
            get_console().print(f"    Answer: {res['answer']}")
            if res.get("feedback"):
                get_console().print(f"    Feedback: {res['feedback']}")
            get_console().print(f"    Score: {res['score']:.4f}\n")
    except Exception as exc:
        get_err_console().print(f"Error searching memory: {exc}")
        raise typer.Exit(1)
    finally:
        store.close()


@memory_app.command("synthesize", help="Synthesize recent memory into a system context file.")
def memory_synthesize(
    ctx: typer.Context,
    limit: MemoryLimit = 20,
) -> None:
    """Review recent memory entries and generate a system context file."""
    store, base_dir, _actor = _get_memory_store(ctx)

    try:
        entries = store.get_recent_entries(limit=limit)
        if not entries:
            get_console().print("No memory entries found to synthesize.")
            return

        context_file = base_dir / "sources" / "_memory" / "system_context.md"
        context_file.parent.mkdir(parents=True, exist_ok=True)

        loader = get_template_loader()
        content = loader.render(
            "memory/system_context.md.j2",
            entries=entries[:10],
        )
        context_file.write_text(content, encoding="utf-8")

        get_console().print(f"✓ Synthesized context written to {context_file}")

    except Exception as exc:
        get_err_console().print(f"Error synthesizing memory: {exc}")
        raise typer.Exit(1)
    finally:
        store.close()
