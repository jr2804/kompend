"""Variant B — navigating summarizer session (knox-u84v.6).

One PydanticAI agent session walks the synthetic chunk tree with four
tools (``list_children`` / ``browse`` / ``summarize_node`` / ``skip``)
and decides node by node. The shipped "planner + worker" traversal is
the comparison arm; this module owns only the navigating session.

Budgets (from ``common``) are enforced in the tools, not suggested in
the prompt: ``summarize_node`` refuses past ``BUDGET_NODES`` (the last
slot is reserved for the root) and ``browse`` refuses past
``BUDGET_WINDOWS``. A refused call returns a message, never content, so
an over-budget model cannot spend what it does not have.

Prompt text lives in ``prompts/``; this module only fills in run data
(ids, counts, budgets). Content is windowed, never cropped: ``browse``
serves fixed-size windows with named next offsets, mirroring
``kompend.compilation.write_tools.read_source``.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from string import Template

from common import (
    BUDGET_NODES,
    BUDGET_WINDOWS,
    Counter,
    RequestKind,
    SyntheticDoc,
    VariantResult,
    build_counted_model,
    configure_provider,
    task_statement,
)
from pydantic_ai import Agent, ModelRetry, RunContext, UsageLimits
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.settings import ModelSettings

from kompend.chunking.outline import OutlineNode, build_outline, outline_to_prompt
from kompend.core.models.structured import structured_to_text
from kompend.core.models.trees import ChunkTree

#: Characters of node content served per ``browse`` call.
WINDOW_CHARS = 4000

#: Output ceiling per reply. Variant A inherits kompend's window-derived
#: ceiling; a raw Agent would otherwise sit on pydantic-ai's own default.
#: 8,192 tokens is protocol-safe for the registered 200k window and cannot
#: bind a 2-5 sentence node summary in practice.
OUTPUT_MAX_TOKENS = 8192

#: Model requests before the run aborts no matter what. Anti-loop backstop
#: only — the node and window budgets are enforced in the tools.
REQUEST_BACKSTOP = 500

#: Tool retries for correctable mistakes (e.g. a mangled node id).
TOOL_RETRIES = 3

_PromptsDir = Path(__file__).with_name("prompts")


def _load_prompt(name: str, **data: object) -> str:
    """Render a prompt file, filling in run data only."""
    return Template((_PromptsDir / name).read_text(encoding="utf-8")).substitute(**data)


@dataclass
class NavDeps:
    """Runtime context for the navigating summarizer session."""

    tree: ChunkTree
    outline: dict[str, OutlineNode]
    summaries: dict[str, str] = field(default_factory=dict)
    skips: dict[str, str] = field(default_factory=dict)
    counter: Counter | None = None
    windows_used: int = 0


def _node_id(value: str) -> str:
    """A node id as the tree spells it."""
    return value.strip()


def _known_ids(deps: NavDeps) -> set[str]:
    """Every id the session may name."""
    return set(deps.outline)


def _own_content(deps: NavDeps, node_id: str) -> str:
    """The node's own payload, mirroring the summarizer's projection.

    Summary, then chunk text, then structured text — the same priority as
    ``kompend.chunking.outline._own_content``. During the session internal
    nodes have no summary yet, so they read empty until summarized.
    """
    node = deps.tree.nodes[node_id]
    if node.summary:
        return node.summary
    if node.chunk is not None:
        return node.chunk.text
    if node.structured is not None:
        return structured_to_text(node.structured)
    return ""


async def list_children(ctx: RunContext[NavDeps], node_id: str) -> str:
    """Outline rows of *node_id*'s children: id, heading, depth, size, preview."""
    deps = ctx.deps
    node_id = _node_id(node_id)
    if node_id not in deps.outline:
        raise ModelRetry(
            f"Unknown node id {node_id!r}. Use an id from a list_children reply, verbatim. "
            f"Children of the root start the walk: list_children(node_id={deps.tree.root_id!r})."
        )
    node = deps.tree.nodes[node_id]
    if not node.children_ids:
        return outline_to_prompt([deps.outline[node_id]]) + ("\nThis node is a leaf: it has no children. Read it with browse, then summarize or skip it.")
    rows = [deps.outline[child_id] for child_id in node.children_ids if child_id in deps.outline]
    return outline_to_prompt(rows)


async def browse(ctx: RunContext[NavDeps], node_id: str, offset: int = 0) -> str:
    """One window of *node_id*'s own text, naming the next offset while more remains."""
    deps = ctx.deps
    node_id = _node_id(node_id)
    if node_id not in deps.outline:
        raise ModelRetry(f"Unknown node id {node_id!r}. Use an id from a list_children reply, verbatim.")
    offset = max(offset, 0)
    text = _own_content(deps, node_id)
    if not text:
        return (
            f"Node {node_id} carries no prose of its own (an internal node with no summary yet). "
            "Summarize it from the summaries you recorded for its children, or skip it."
        )
    if offset >= len(text):
        return ""
    if deps.windows_used >= BUDGET_WINDOWS and deps.counter is not None:
        return (
            f"Window budget spent ({BUDGET_WINDOWS} windows). No more content will be served: "
            "summarize or skip from what you have read, then summarize the root."
        )
    window = text[offset : offset + WINDOW_CHARS]
    next_offset = offset + len(window)
    if deps.counter is not None:
        deps.counter.record_window()
    deps.windows_used += 1
    if next_offset >= len(text):
        return window
    return f"{window}\n\n[end of window — {next_offset} of {len(text)} characters read; call browse(node_id={node_id!r}, offset={next_offset}) for the rest]"


async def summarize_node(ctx: RunContext[NavDeps], node_id: str, summary: str) -> str:
    """Record the summary for a node."""
    deps = ctx.deps
    node_id = _node_id(node_id)
    if node_id not in deps.outline:
        raise ModelRetry(f"Unknown node id {node_id!r}. Use an id from a list_children reply, verbatim.")
    summary = summary.strip()
    if not summary:
        raise ModelRetry(f"Empty summary for node {node_id!r}. Write 2-5 sentences from text you actually read, then call again.")
    if node_id in deps.summaries:
        deps.summaries[node_id] = summary
        return f"Summary updated for node {node_id}."
    if len(deps.summaries) >= BUDGET_NODES:
        return f"Summary budget spent ({BUDGET_NODES} nodes). No more summaries will be recorded: finish with the root if it is still missing, otherwise stop."
    if len(deps.summaries) >= BUDGET_NODES - 1 and node_id != deps.tree.root_id:
        return f"One summary slot remains and it is reserved for the root ({deps.tree.root_id}). Summarize the root now, or skip more branches and stop."
    deps.summaries[node_id] = summary
    return f"Summary recorded for node {node_id}."


async def skip(ctx: RunContext[NavDeps], node_id: str, reason: str) -> str:
    """Record a skip for a node with a reason.

    Keep *reason* to the planner's vocabulary so both variants stay
    comparable: "boilerplate", "appendix", "repeated material", "other".
    """
    deps = ctx.deps
    node_id = _node_id(node_id)
    if node_id not in deps.outline:
        raise ModelRetry(f"Unknown node id {node_id!r}. Use an id from a list_children reply, verbatim.")
    deps.skips[node_id] = reason.strip() or "other"
    return f"Skip recorded for node {node_id}."


def classify(response: ModelResponse) -> RequestKind:
    """A response that called ``summarize_node`` is a summarize request, else decision."""
    for part in response.parts:
        if isinstance(part, ToolCallPart) and part.tool_name == "summarize_node":
            return "summarize"
    return "decision"


def _check_task_prompt() -> str:
    """The shipped task statement, loaded from the prompt file, drift-checked."""
    loaded = _load_prompt("navigation_task.md")
    shipped = task_statement()
    if loaded != shipped:
        msg = "prompts/navigation_task.md drifted from common.task_statement() — refresh the copy so both variants judge the same task"
        raise ValueError(msg)
    return loaded


def run(doc: SyntheticDoc, counter: Counter) -> VariantResult:
    """Run a navigating summarizer session over doc.tree and return what it produced."""
    return asyncio.run(_run(doc, counter))


async def _run(doc: SyntheticDoc, counter: Counter) -> VariantResult:
    """The session: one agent run over the tree, then a root-summary follow-up if needed."""
    configure_provider()
    statement = _check_task_prompt()
    entries = build_outline(doc.tree)
    deps = NavDeps(tree=doc.tree, outline={entry.node_id: entry for entry in entries}, counter=counter)
    system_prompt = _load_prompt(
        "navigation_system.md",
        budget_nodes=BUDGET_NODES,
        budget_windows=BUDGET_WINDOWS,
        window_chars=WINDOW_CHARS,
        root_id=doc.tree.root_id,
    )
    start_prompt = _load_prompt(
        "navigation_start.md",
        task_statement=statement,
        node_count=len(entries),
        root_id=doc.tree.root_id,
        chapters=", ".join(doc.chapter_headings),
        budget_nodes=BUDGET_NODES,
        budget_windows=BUDGET_WINDOWS,
    )
    model = build_counted_model(counter, classify=classify)
    agent: Agent[NavDeps, str] = Agent(
        model,
        deps_type=NavDeps,
        system_prompt=system_prompt,
        tools=[list_children, browse, summarize_node, skip],
        retries=TOOL_RETRIES,
        model_settings=ModelSettings(max_tokens=OUTPUT_MAX_TOKENS),
    )
    limits = UsageLimits(request_limit=REQUEST_BACKSTOP)
    notes = [
        f"window_chars={WINDOW_CHARS}",
        f"request_backstop={REQUEST_BACKSTOP} output_max_tokens={OUTPUT_MAX_TOKENS}",
        "last summary slot reserved for the root",
        f"tree nodes={len(entries)} chapters={len(doc.chapter_headings)}",
    ]
    history = None
    # The runner owns the wall clock (one Stopwatch around the whole variant),
    # so this module does not start a second one — two clocks would report the
    # elapsed time twice.
    try:
        result = await agent.run(start_prompt, deps=deps, usage_limits=limits)
        history = result.all_messages()
    except Exception as exc:  # a dead session still reports what it produced
        notes.append(f"main session ended with {type(exc).__name__}: {exc}")
    if doc.tree.root_id not in deps.summaries and len(deps.summaries) < BUDGET_NODES and history is not None:
        root_prompt = _load_prompt(
            "navigation_root.md",
            summaries_recorded=len(deps.summaries),
            skips_recorded=len(deps.skips),
            nodes_left=BUDGET_NODES - len(deps.summaries),
            root_id=doc.tree.root_id,
        )
        try:
            await agent.run(root_prompt, message_history=history, deps=deps, usage_limits=limits)
        except Exception as exc:  # same deal: report, don't crash the runner
            notes.append(f"root follow-up ended with {type(exc).__name__}: {exc}")
    if doc.tree.root_id not in deps.summaries:
        notes.append("no root summary recorded")
    return VariantResult(
        variant="b",
        summaries=dict(deps.summaries),
        skips=dict(deps.skips),
        counter=counter,
        notes=notes,
    )
