"""CLI arguments for the compile command."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from cyclopts import Parameter

from kompend.cli.args.common import LLM_GROUP, OUTPUT_GROUP, SOURCES_GROUP
from kompend.core.env import EnvVar

# ── `compile` command surface ───────────────────────────────────────────

OnlineUrls = Annotated[
    list[str] | None,
    Parameter(
        negative="",
        name="--url",
        help="URLs to download before processing (repeatable).",
        group=SOURCES_GROUP,
    ),
]

UrlFile = Annotated[
    Path | None,
    Parameter(
        name="--url-file",
        help="File with URLs to download (one per line, # comments ignored).",
        group=SOURCES_GROUP,
    ),
]

Ignore = Annotated[
    list[str] | None,
    Parameter(
        negative="",
        name="--ignore",
        help="Ignore pattern (glob-style, repeatable).",
        group=SOURCES_GROUP,
    ),
]

# ── compile --distill pre-step ──────────────────────────────────────────

CompileDistill = Annotated[
    bool,
    Parameter(
        name="--distill",
        help="Distill captured/learned notes into sources/_distilled/ pages before compiling (archives the notes it merged). Default: off.",
        group=SOURCES_GROUP,
        negative="",
    ),
]

IndexStore = Annotated[
    bool | None,
    Parameter(
        name="--index",
        help="Index compiled wiki in zvec for semantic search. Default: on.",
        group=OUTPUT_GROUP,
    ),
]

EmbedModel = Annotated[
    str | None,
    Parameter(
        name="--embed-model",
        help="Embedding model for zvec indexing. Default: config compile.embedding.model (empty = no vector index).",
        group=OUTPUT_GROUP,
    ),
]

# ── Q18: custom frontmatter fields ──────────────────────────────────────

Frontmatter = Annotated[
    list[str] | None,
    Parameter(
        negative="",
        name="--frontmatter",
        help="Frontmatter field in key=value format (repeatable). E.g. --frontmatter language=en --frontmatter domain=ml",
    ),
]

SynthesizeFlag = Annotated[
    bool | None,
    Parameter(
        name="--synthesize",
        help="Also generate an LLM-synthesized overview.md page. Default: on.",
        group=OUTPUT_GROUP,
    ),
]

# ── N1 compile --lint integration ───────────────────────────────────────

CompileLint = Annotated[
    bool,
    Parameter(
        name="--lint",
        help="Run deterministic structural lint (no LLM) after compile. Default: on (override with KOMP_COMPILE_LINT).",
        group=OUTPUT_GROUP,
        env_var=EnvVar.COMPILE_LINT,
    ),
]

CompileLintAll = Annotated[
    bool,
    Parameter(
        name="--lint-all",
        help="Run the full lint pipeline (structural + LLM rules + remediate) after compile. Implies --lint. Override with KOMP_COMPILE_LINT_ALL.",
        group=OUTPUT_GROUP,
        env_var=EnvVar.COMPILE_LINT_ALL,
    ),
]


LlmDefaultModel = Annotated[
    str | None,
    Parameter(
        name="--llm-default-model",
        help=(
            "Primary model in provider/model format ('provider/model'), "
            "e.g. 'openai/gpt-4o' or 'ollama/llama3.2'. "
            "Overrides the config-file model/provider when set."
        ),
        env_var=EnvVar.LLM_DEFAULT_MODEL,
        group=LLM_GROUP,
    ),
]

VlmModel = Annotated[
    str | None,
    Parameter(
        name="--vlm-model",
        help=(
            "Vision model in provider/model format ('provider/model') used to caption "
            "IMAGE chunks during resolution. When set, VLMEnricher is added to "
            "the resolver chain. Example: 'openai/gpt-4o'. Default: off."
        ),
        env_var=EnvVar.VLM_MODEL,
        group=LLM_GROUP,
    ),
]

LlmFallbackChain = Annotated[
    list[str] | None,
    Parameter(
        negative="",
        name="--llm-fallback-chain",
        help=(
            "Ordered fallback model(s) tried when the primary LLM fails "
            "(repeatable). Each value is a message-dict 'provider/model' "
            "string, e.g. 'openai/gpt-4o'. Empty means single-model mode. "
            "Env var value: comma-separated list."
        ),
        env_var=EnvVar.LLM_FALLBACK_CHAIN,
        group=LLM_GROUP,
    ),
]
