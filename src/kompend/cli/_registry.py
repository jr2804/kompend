"""Command registry: every CLI command, its module, and its help panel.

One entry per command, so a command's name, its registering module, and its help
panel cannot drift apart. ``app.py`` and the lazy group both read this module;
no other file declares a command name.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final


class HelpPanel(StrEnum):
    """Workflow panel a command is grouped under in ``kompend --help``."""

    BUILD_PIPELINE = "Build pipeline"
    CAPTURE_SYNTHESIS = "Capture & synthesis"
    KNOWLEDGE_BASE = "Knowledge base"
    QUALITY_INSPECTION = "Quality & inspection"
    SETUP = "Setup"
    SERVING = "Serving"


#: Panel display order for help output — workflow order, not alphabetical.
PANEL_ORDER: Final[tuple[HelpPanel, ...]] = (
    HelpPanel.BUILD_PIPELINE,
    HelpPanel.CAPTURE_SYNTHESIS,
    HelpPanel.KNOWLEDGE_BASE,
    HelpPanel.QUALITY_INSPECTION,
    HelpPanel.SETUP,
    HelpPanel.SERVING,
)


class CommandModule(StrEnum):
    """Import path of the module whose import registers a command."""

    AGENT = "kompend.cli.cmds.agent"
    AGENTS = "kompend.cli.cmds.agents"
    ATTACHMENTS = "kompend.cli.cmds.knowledge.attachments"
    CAPTURE = "kompend.cli.cmds.knowledge.capture"
    CONFIG = "kompend.cli.cmds.pipeline.config"
    DOCTOR = "kompend.cli.cmds.quality.doctor_cmd"
    INIT = "kompend.cli.cmds.init"
    LINT = "kompend.cli.cmds.quality.lint"
    LIST_PARSERS = "kompend.cli.cmds.quality.list_parsers_cmd"
    LOG = "kompend.cli.cmds.quality.log"
    MEMORY = "kompend.cli.cmds.knowledge.memory"
    ORGANIZE = "kompend.cli.cmds.knowledge.organize"
    PACK = "kompend.cli.cmds.knowledge.pack"
    PIPELINE = "kompend.cli.cmds.pipeline.pipeline"
    PROVENANCE = "kompend.cli.cmds.quality.provenance_cmd"
    QUERY = "kompend.cli.cmds.knowledge.query"
    REFERENCES = "kompend.cli.cmds.knowledge.references"
    REVIEW = "kompend.cli.cmds.quality.review"
    SERVE = "kompend.cli.cmds.serve"
    SOURCES = "kompend.cli.cmds.pipeline.sources"
    STATUS = "kompend.cli.cmds.quality.status_cmd"


class CommandName(StrEnum):
    """CLI-visible command name."""

    AGENT = "agent"
    AGENTS = "agents"
    ATTACHMENTS = "attachments"
    ATT = "att"
    CAPTURE = "capture"
    COMPILE = "compile"
    CONFIG = "config"
    CONSOLIDATE = "consolidate"
    DISTILL = "distill"
    DOCTOR = "doctor"
    INDEX = "index"
    INGEST = "ingest"
    INIT = "init"
    LEARN = "learn"
    LINT = "lint"
    LIST_PARSERS = "list-parsers"
    LOG = "log"
    MCP = "mcp"
    MEMORY = "memory"
    ORGANIZE = "organize"
    PACK_INSTALL = "pack-install"
    PACK_LIST = "pack-list"
    PARSE = "parse"
    PROVENANCE = "provenance"
    QUERY = "query"
    REFERENCES = "references"
    REVIEW = "review"
    SERVE = "serve"
    SOURCES = "sources"
    STATUS = "status"
    WRITE = "write"


class OnDemandModule(StrEnum):
    """Import path of a module the CLI loads only when it is needed.

    Unlike :class:`CommandModule`, importing one of these registers no command —
    they exist because their cost is real and their work is occasional. The
    telemetry module is the case that motivated this: importing ``logfire``
    costs roughly 700 ms and pulls in ``requests``, so ``kompend --version``,
    ``--help`` and every in-process import of the CLI used to pay for telemetry
    none of them emit. The run log is the second: it imports the config engine
    and the vault store, and neither ``--version`` nor ``--help`` records a run.
    The app module is the third: the console launcher answers a bare
    ``--version`` without building the Typer tree.
    """

    TELEMETRY = "kompend.telemetry"
    RUNLOG = "kompend.core.reporting.runlog"
    APP = "kompend.cli.app"


@dataclass(frozen=True, slots=True)
class CommandSpec:
    """Where a command lives, where help lists it, and its one-line summary.

    ``short_help`` is the summary ``kompend --help`` shows. It lives here so
    help renders from this table instead of importing the command's module.
    """

    module: CommandModule
    panel: HelpPanel
    short_help: str


#: Every command kompend exposes. A name absent here is not part of the CLI.
COMMANDS: Final[dict[CommandName, CommandSpec]] = {
    # Build pipeline
    CommandName.COMPILE: CommandSpec(
        CommandModule.PIPELINE,
        HelpPanel.BUILD_PIPELINE,
        "Compile documents into a knowledge wiki.",
    ),
    CommandName.INGEST: CommandSpec(
        CommandModule.PIPELINE,
        HelpPanel.BUILD_PIPELINE,
        "Ingest documents into the knowledge graph without compiling a wiki.",
    ),
    CommandName.PARSE: CommandSpec(
        CommandModule.PIPELINE,
        HelpPanel.BUILD_PIPELINE,
        "Parse documents and show their structure.",
    ),
    CommandName.INDEX: CommandSpec(
        CommandModule.PIPELINE,
        HelpPanel.BUILD_PIPELINE,
        "Rebuild the zvec vector index for semantic search.",
    ),
    # Capture & synthesis
    CommandName.CAPTURE: CommandSpec(
        CommandModule.CAPTURE,
        HelpPanel.CAPTURE_SYNTHESIS,
        "Extract knowledge items from text via LLM; save each as a source file.",
    ),
    CommandName.CONSOLIDATE: CommandSpec(
        CommandModule.CAPTURE,
        HelpPanel.CAPTURE_SYNTHESIS,
        "Find and archive redundant captured/learned/distilled notes.",
    ),
    CommandName.LEARN: CommandSpec(
        CommandModule.CAPTURE,
        HelpPanel.CAPTURE_SYNTHESIS,
        "Store a single fact verbatim as a source file — no LLM involved.",
    ),
    CommandName.DISTILL: CommandSpec(
        CommandModule.CAPTURE,
        HelpPanel.CAPTURE_SYNTHESIS,
        "Merge captured notes into coherent wiki pages via LLM.",
    ),
    CommandName.WRITE: CommandSpec(
        CommandModule.CAPTURE,
        HelpPanel.CAPTURE_SYNTHESIS,
        "Create a wiki page synthesising content from one or more sources.",
    ),
    # Knowledge base
    CommandName.QUERY: CommandSpec(
        CommandModule.QUERY,
        HelpPanel.KNOWLEDGE_BASE,
        "Ask a natural language question against the compiled wiki.",
    ),
    CommandName.MEMORY: CommandSpec(
        CommandModule.MEMORY,
        HelpPanel.KNOWLEDGE_BASE,
        "Manage query memory and context synthesis.",
    ),
    CommandName.SOURCES: CommandSpec(
        CommandModule.SOURCES,
        HelpPanel.KNOWLEDGE_BASE,
        "Manage ingested source files and their tags.",
    ),
    CommandName.REFERENCES: CommandSpec(
        CommandModule.REFERENCES,
        HelpPanel.KNOWLEDGE_BASE,
        "Manage extracted bibliographic references.",
    ),
    CommandName.ATTACHMENTS: CommandSpec(
        CommandModule.ATTACHMENTS,
        HelpPanel.KNOWLEDGE_BASE,
        "Manage source attachments (supplementary files).",
    ),
    CommandName.ATT: CommandSpec(
        CommandModule.ATTACHMENTS,
        HelpPanel.KNOWLEDGE_BASE,
        "Manage source attachments (supplementary files).",
    ),
    CommandName.ORGANIZE: CommandSpec(
        CommandModule.ORGANIZE,
        HelpPanel.KNOWLEDGE_BASE,
        "Categorise files in ``sources/`` into a hierarchical directory tree.",
    ),
    CommandName.PACK_LIST: CommandSpec(
        CommandModule.PACK,
        HelpPanel.KNOWLEDGE_BASE,
        "List installed contribution packs.",
    ),
    CommandName.PACK_INSTALL: CommandSpec(
        CommandModule.PACK,
        HelpPanel.KNOWLEDGE_BASE,
        "Install a built-in contribution pack.",
    ),
    CommandName.AGENTS: CommandSpec(
        CommandModule.AGENTS,
        HelpPanel.KNOWLEDGE_BASE,
        "Manage TOML-defined conversational agents.",
    ),
    # Quality & inspection
    CommandName.LINT: CommandSpec(
        CommandModule.LINT,
        HelpPanel.QUALITY_INSPECTION,
        "Lint a compiled wiki for quality issues and query lint history.",
    ),
    CommandName.DOCTOR: CommandSpec(
        CommandModule.DOCTOR,
        HelpPanel.QUALITY_INSPECTION,
        "Validate configuration and connectivity for a wiki.",
    ),
    CommandName.STATUS: CommandSpec(
        CommandModule.STATUS,
        HelpPanel.QUALITY_INSPECTION,
        "Show per-wiki statistics.",
    ),
    CommandName.LOG: CommandSpec(
        CommandModule.LOG,
        HelpPanel.QUALITY_INSPECTION,
        "Regenerate the pipeline log from structured stores (no LLM calls).",
    ),
    CommandName.REVIEW: CommandSpec(
        CommandModule.REVIEW,
        HelpPanel.QUALITY_INSPECTION,
        "Manage the review queue for compiled pages.",
    ),
    CommandName.PROVENANCE: CommandSpec(
        CommandModule.PROVENANCE,
        HelpPanel.QUALITY_INSPECTION,
        "Trace compiled wiki page(s) back to their source chunks.",
    ),
    CommandName.LIST_PARSERS: CommandSpec(
        CommandModule.LIST_PARSERS,
        HelpPanel.QUALITY_INSPECTION,
        "List all registered parsers (built-in + external + HTTP).",
    ),
    # Setup
    CommandName.INIT: CommandSpec(
        CommandModule.INIT,
        HelpPanel.SETUP,
        "Scaffold a new kompend wiki project.",
    ),
    CommandName.CONFIG: CommandSpec(
        CommandModule.CONFIG,
        HelpPanel.SETUP,
        "Inspect and edit kompend configuration.",
    ),
    # Serving
    CommandName.MCP: CommandSpec(
        CommandModule.SERVE,
        HelpPanel.SERVING,
        "Start the kompend MCP server over stdio transport.",
    ),
    CommandName.SERVE: CommandSpec(
        CommandModule.SERVE,
        HelpPanel.SERVING,
        "Start the kompend web UI server.",
    ),
    CommandName.AGENT: CommandSpec(
        CommandModule.AGENT,
        HelpPanel.SERVING,
        "Run kompend as an ACP agent on stdio.",
    ),
}
