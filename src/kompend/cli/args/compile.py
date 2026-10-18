"""CLI arguments for the compile command."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from kompend.core.env import EnvVar

# ── `compile` command surface ───────────────────────────────────────────

OnlineUrls = Annotated[
    list[str] | None,
    typer.Option(
        "--url",
        help="URLs to download before processing (repeatable).",
        rich_help_panel="Sources",
    ),
]

UrlFile = Annotated[
    Path | None,
    typer.Option(
        "--url-file",
        help="File with URLs to download (one per line, # comments ignored).",
        rich_help_panel="Sources",
    ),
]

Ignore = Annotated[
    list[str] | None,
    typer.Option(
        "--ignore",
        help="Ignore pattern (glob-style, repeatable).",
        rich_help_panel="Sources",
    ),
]

# ── compile --distill pre-step ──────────────────────────────────────────

CompileDistill = Annotated[
    bool,
    typer.Option(
        "--distill",
        help=("Distill captured/learned notes into sources/_distilled/ pages before compiling (archives the notes it merged). Default: off."),
        rich_help_panel="Sources",
    ),
]

IndexStore = Annotated[
    bool | None,
    typer.Option(
        "--index/--no-index",
        help="Index compiled wiki in zvec for semantic search. Default: on.",
        rich_help_panel="Output",
    ),
]

EmbedModel = Annotated[
    str | None,
    typer.Option(
        "--embed-model",
        help="Embedding model for zvec indexing. Default: config compile.embedding.model (empty = no vector index).",
        rich_help_panel="Output",
    ),
]

# ── Q18: custom frontmatter fields ──────────────────────────────────────

Frontmatter = Annotated[
    list[str] | None,
    typer.Option(
        "--frontmatter",
        help=("Frontmatter field in key=value format (repeatable). E.g. --frontmatter language=en --frontmatter domain=ml"),
    ),
]

SynthesizeFlag = Annotated[
    bool | None,
    typer.Option(
        "--synthesize/--no-synthesize",
        help="Also generate an LLM-synthesized overview.md page. Default: on.",
        rich_help_panel="Output",
    ),
]

# ── N1 compile --lint integration ───────────────────────────────────────

CompileLint = Annotated[
    bool,
    typer.Option(
        "--lint/--no-lint",
        help="Run deterministic structural lint (no LLM) after compile. Default: on (override with KOMP_COMPILE_LINT).",
        rich_help_panel="Output",
        envvar=EnvVar.COMPILE_LINT,
    ),
]

CompileLintAll = Annotated[
    bool,
    typer.Option(
        "--lint-all/--no-lint-all",
        help="Run the full lint pipeline (structural + LLM rules + remediate) after compile. Implies --lint. Override with KOMP_COMPILE_LINT_ALL.",
        rich_help_panel="Output",
        envvar=EnvVar.COMPILE_LINT_ALL,
    ),
]


LlmDefaultModel = Annotated[
    str | None,
    typer.Option(
        "--llm-default-model",
        help=(
            "Primary model in provider/model format ('provider/model'), "
            "e.g. 'openai/gpt-4o' or 'ollama/llama3.2'. "
            "Overrides the config-file model/provider when set."
        ),
        envvar=EnvVar.LLM_DEFAULT_MODEL,
        rich_help_panel="LLM",
    ),
]

VlmModel = Annotated[
    str | None,
    typer.Option(
        "--vlm-model",
        help=(
            "Vision model in provider/model format ('provider/model') used to caption "
            "IMAGE chunks during resolution. When set, VLMEnricher is added to "
            "the resolver chain. Example: 'openai/gpt-4o'. Default: off."
        ),
        envvar=EnvVar.VLM_MODEL,
        rich_help_panel="LLM",
    ),
]

LlmFallbackChain = Annotated[
    list[str] | None,
    typer.Option(
        "--llm-fallback-chain",
        help=(
            "Ordered fallback model(s) tried when the primary LLM fails "
            "(repeatable). Each value is a message-dict 'provider/model' "
            "string, e.g. 'openai/gpt-4o'. Empty means single-model mode. "
            "Env var value: comma-separated list."
        ),
        envvar=EnvVar.LLM_FALLBACK_CHAIN,
        rich_help_panel="LLM",
    ),
]

LlmFallbackMaxRetries = Annotated[
    int,
    typer.Option(
        "--llm-fallback-max-retries",
        help=("Transient-error retries before advancing to the next model in the failover chain. 2 means up to three attempts per model."),
        min=0,
        envvar=EnvVar.LLM_FALLBACK_MAX_RETRIES,
        rich_help_panel="LLM",
    ),
]
