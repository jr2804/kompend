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


@dataclass(frozen=True, slots=True)
class CommandSpec:
    """Where a command lives and where help lists it."""

    module: CommandModule
    panel: HelpPanel


#: Every command kompend exposes. A name absent here is not part of the CLI.
COMMANDS: Final[dict[CommandName, CommandSpec]] = {
    # Build pipeline
    CommandName.COMPILE: CommandSpec(CommandModule.PIPELINE, HelpPanel.BUILD_PIPELINE),
    CommandName.INGEST: CommandSpec(CommandModule.PIPELINE, HelpPanel.BUILD_PIPELINE),
    CommandName.PARSE: CommandSpec(CommandModule.PIPELINE, HelpPanel.BUILD_PIPELINE),
    CommandName.INDEX: CommandSpec(CommandModule.PIPELINE, HelpPanel.BUILD_PIPELINE),
    # Capture & synthesis
    CommandName.CAPTURE: CommandSpec(CommandModule.CAPTURE, HelpPanel.CAPTURE_SYNTHESIS),
    CommandName.CONSOLIDATE: CommandSpec(CommandModule.CAPTURE, HelpPanel.CAPTURE_SYNTHESIS),
    CommandName.LEARN: CommandSpec(CommandModule.CAPTURE, HelpPanel.CAPTURE_SYNTHESIS),
    CommandName.DISTILL: CommandSpec(CommandModule.CAPTURE, HelpPanel.CAPTURE_SYNTHESIS),
    CommandName.WRITE: CommandSpec(CommandModule.CAPTURE, HelpPanel.CAPTURE_SYNTHESIS),
    # Knowledge base
    CommandName.QUERY: CommandSpec(CommandModule.QUERY, HelpPanel.KNOWLEDGE_BASE),
    CommandName.MEMORY: CommandSpec(CommandModule.MEMORY, HelpPanel.KNOWLEDGE_BASE),
    CommandName.SOURCES: CommandSpec(CommandModule.SOURCES, HelpPanel.KNOWLEDGE_BASE),
    CommandName.REFERENCES: CommandSpec(CommandModule.REFERENCES, HelpPanel.KNOWLEDGE_BASE),
    CommandName.ATTACHMENTS: CommandSpec(CommandModule.ATTACHMENTS, HelpPanel.KNOWLEDGE_BASE),
    CommandName.ATT: CommandSpec(CommandModule.ATTACHMENTS, HelpPanel.KNOWLEDGE_BASE),
    CommandName.ORGANIZE: CommandSpec(CommandModule.ORGANIZE, HelpPanel.KNOWLEDGE_BASE),
    CommandName.PACK_LIST: CommandSpec(CommandModule.PACK, HelpPanel.KNOWLEDGE_BASE),
    CommandName.PACK_INSTALL: CommandSpec(CommandModule.PACK, HelpPanel.KNOWLEDGE_BASE),
    CommandName.AGENTS: CommandSpec(CommandModule.AGENTS, HelpPanel.KNOWLEDGE_BASE),
    # Quality & inspection
    CommandName.LINT: CommandSpec(CommandModule.LINT, HelpPanel.QUALITY_INSPECTION),
    CommandName.DOCTOR: CommandSpec(CommandModule.DOCTOR, HelpPanel.QUALITY_INSPECTION),
    CommandName.STATUS: CommandSpec(CommandModule.STATUS, HelpPanel.QUALITY_INSPECTION),
    CommandName.LOG: CommandSpec(CommandModule.LOG, HelpPanel.QUALITY_INSPECTION),
    CommandName.REVIEW: CommandSpec(CommandModule.REVIEW, HelpPanel.QUALITY_INSPECTION),
    CommandName.PROVENANCE: CommandSpec(CommandModule.PROVENANCE, HelpPanel.QUALITY_INSPECTION),
    CommandName.LIST_PARSERS: CommandSpec(CommandModule.LIST_PARSERS, HelpPanel.QUALITY_INSPECTION),
    # Setup
    CommandName.INIT: CommandSpec(CommandModule.INIT, HelpPanel.SETUP),
    CommandName.CONFIG: CommandSpec(CommandModule.CONFIG, HelpPanel.SETUP),
    # Serving
    CommandName.MCP: CommandSpec(CommandModule.SERVE, HelpPanel.SERVING),
    CommandName.SERVE: CommandSpec(CommandModule.SERVE, HelpPanel.SERVING),
    CommandName.AGENT: CommandSpec(CommandModule.AGENT, HelpPanel.SERVING),
}
