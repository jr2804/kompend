"""Unified configuration system for KNOX.

5-layer precedence (highest wins):

1. CLI flags           — explicit per-invocation overrides
2. Shell env vars      — ``KNOX_*`` per-session overrides
3. Project config      — ``./knox.toml`` and ``./.knox/config.toml``,
                         discovered hierarchically by walking up from cwd
4. Global config       — ``~/.config/knox/config.toml`` (per-user)
5. Built-in defaults   — hard-coded in :data:`DEFAULTS`

Config files use TOML. Example ``knox.toml``::

    [compile]
    sources_dir = "sources"
    output_dir = "wiki"
    format = "markdown"
    chunk_size = 1024
    image_mode = "file"
    mode = "full"
    workers = 1
    two_phase = true

    [compile.llm]
    default_model = "ollama/llama3.2"
    fallback_chain = ["openai/gpt-4o"]
    fallback_max_retries = 2

    [compile.output]
    flavor = "obsidian"

    [compile.output.frontmatter]
    language = "en"
    domain = "ml"

    [lint]
    enabled = ["broken-wikilinks", "orphan-page"]

    [eval]
    threshold = 80
    history_size = 20

    [ignore]
    patterns = ["*.tmp", ".drafts/"]
"""

from __future__ import annotations

import contextlib
import copy
import os
import re
import tomllib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tomli_w
from pydantic import Field, create_model
from pydantic_settings import BaseSettings, SettingsConfigDict

from knox.core.config.keys import (
    COMPILE_BUDGET_CHUNK_SIZE_MAX,
    COMPILE_BUDGET_CHUNK_SIZE_MIN,
    COMPILE_BUDGET_CONCEPTS_PER_BATCH,
    COMPILE_BUDGET_EXTRACT_OUTPUT_MAX,
    COMPILE_BUDGET_GEN_OUTPUT_MAX,
    COMPILE_BUDGET_MAX_CONCEPT_PAGES,
    COMPILE_BUDGET_MIN_CONCEPT_SOURCES,
    COMPILE_BUDGET_MIN_CONTENT_LENGTH,
    COMPILE_BUDGET_OVERVIEW_OUTPUT_MAX,
    COMPILE_BUDGET_WICER_MAX_ITERATIONS,
    COMPILE_CHUNK_SIZE,
    COMPILE_EMBEDDING_BATCH_SIZE,
    COMPILE_EMBEDDING_DIM,
    COMPILE_EMBEDDING_MODEL,
    COMPILE_FORCE,
    COMPILE_FORMAT,
    COMPILE_IMAGE_MODE,
    COMPILE_IN_PLACE_SOURCES,
    COMPILE_LINT,
    COMPILE_LLM_CONTEXT_WINDOW_CAP,
    COMPILE_LLM_DATA_FORMAT,
    COMPILE_LLM_DEFAULT_MODEL,
    COMPILE_LLM_FALLBACK_CHAIN,
    COMPILE_LLM_FALLBACK_MAX_INPUT,
    COMPILE_LLM_FALLBACK_MAX_OUTPUT,
    COMPILE_LLM_FALLBACK_MAX_RETRIES,
    COMPILE_LLM_NUM_CTX_MAX,
    COMPILE_LLM_NUM_CTX_MIN,
    COMPILE_LLM_TIMEOUT,
    COMPILE_LLM_VLM_MODEL,
    COMPILE_MEMORY_DECAY_THRESHOLD,
    COMPILE_MEMORY_ENABLED,
    COMPILE_MEMORY_HALF_LIFE_DAYS,
    COMPILE_NO_CACHE,
    COMPILE_OUTPUT_DIR,
    COMPILE_OUTPUT_FLAVOR,
    COMPILE_OUTPUT_MAX_RELATED_LINKS,
    COMPILE_OUTPUT_RUMDL_CONFIG,
    COMPILE_PAGE_KIND,
    COMPILE_PDF_MAX_PAGES_BEFORE_CHUNKING,
    COMPILE_PDF_PAGE_CHUNK_OVERLAP,
    COMPILE_PDF_PAGE_CHUNK_SIZE,
    COMPILE_SOURCES_DIR,
    COMPILE_START_AT_TIER,
    COMPILE_WORKERS,
    DEDUP_ENABLED,
    DEDUP_MATCHERS,
    DEDUP_MINHASH_LSH_THRESHOLD,
    DEDUP_MODEL,
    DEDUP_THRESHOLD_IDENTICAL,
    DEDUP_THRESHOLD_SIMILAR,
    ORGANIZE_COLLAPSE_SINGLE_FILE,
    ORGANIZE_MAX_DEPTH,
    ORGANIZE_MIN_SUBCATEGORIES,
    PATHS_CACHE_DIR,
    PATHS_OUTPUT_DIR,
    PATHS_SOURCES_DIR,
    PATHS_STATE_DIR,
    PATHS_WIKI_DIR,
    SEARCH_CONTEXTUAL_ENABLED,
    SEARCH_CONTEXTUAL_MODEL,
    SEARCH_CRAG_ENABLED,
    SEARCH_CRAG_THRESHOLD,
    SEARCH_EXPANSION_ENABLED,
    SEARCH_EXPANSION_VARIANTS,
    SEARCH_HYDE_ENABLED,
    SEARCH_RERANKER_ENABLED,
    SEARCH_RERANKER_MODEL,
    SEARCH_RERANKER_TOP_K,
    TRUST_PRIVACY_FILTER,
    TRUST_USER_TRACKING,
)
from knox.core.paths import CACHE_DIR, CONFIG_FILE, KNOX_CONFIG_FILE, KNOX_DIR, PROJECT_CONFIG_FILE, VaultPaths
from knox.logging import get_logger

logger = get_logger(__name__)

# ── File paths ───────────────────────────────────────────────────────────

_PROJECT_CONFIG_NAMES: tuple[str, ...] = (KNOX_CONFIG_FILE, PROJECT_CONFIG_FILE)
_GLOBAL_CONFIG_DIR: str = "knox"
_GLOBAL_CONFIG_NAME: str = CONFIG_FILE

# ── Built-in defaults ───────────────────────────────────────────────────

DEFAULTS: dict[str, Any] = {
    "compile": {
        "sources_dir": "sources",
        "output_dir": "wiki",
        "format": "markdown",
        "chunk_size": 1024,  # legacy fallback only — chunk target derived from context window
        "image_mode": "file",
        "mode": "full",
        "workers": 1,
        "no_cache": False,
        "force": False,
        "in_place_sources": False,
        "page_kind": None,
        "memory_enabled": True,
        "memory_half_life_days": 90.0,
        "memory_decay_threshold": 0.3,
        "review": False,
        "bypass_review": False,
        "provenance": False,
        "private": False,
        "verify": False,
        "synthesize": False,
        "index": False,
        "lint": False,
        "budget": {
            "concepts_per_batch": 8,
            "verify_min_ngram": 3,
            "wicer_max_iterations": 3,
            "min_content_length": 50,
            "min_concept_sources": 2,
            "max_concept_pages": 50,
            "chunk_size_min": 2048,
            "chunk_size_max": 131072,
            "extract_output_max": 8192,
            "gen_output_max": 16384,
            "overview_output_max": 4096,
            "verify_output_max": 4096,
            "lint_output_max": 8192,
            "wicer_probe_output_max": 4096,
            "wicer_eval_output_max": 4096,
            "summary_max_content_chars": 8000,
            "summary_target_chars": 600,
            "summary_skip_single_child": True,
            "summary_chunk_threshold": 10,
        },
        "llm": {
            "default_model": None,
            "vlm_model": None,
            "fallback_chain": [],
            "fallback_max_retries": 2,
            "timeout": 120.0,
            "fallback_max_input": 8192,
            "fallback_max_output": 4096,
            "num_ctx_min": 2048,
            "num_ctx_max": 131072,
            "context_window_cap": None,
            "api_base": None,
            "api_key": None,
            "data_format": "json",
        },
        "output": {
            "flavor": None,
            "frontmatter": {},
            "rumdl_config": None,
            "max_related_links": 15,
        },
        "pdf": {
            "max_pages_before_chunking": 50,
            "page_chunk_size": 30,
            "page_chunk_overlap": 5,
        },
        "embedding": {
            "model": None,
            "dim": 384,  # deprecated — auto-detected from model at runtime
            "batch_size": 10,
        },
    },
    "lint": {
        "enabled": None,  # None = all structural rules
    },
    "eval": {
        "threshold": None,
        "history_size": 20,
    },
    "dedup": {
        "enabled": False,  # opt-in by default
        "matchers": "embedding,minhash",
        "threshold_identical": 0.95,
        "threshold_similar": 0.80,
        "model": "",  # empty = compile.embedding.model → all-MiniLM-L6-v2
        "minhash_lsh_threshold": 0.8,
    },
    "trust": {
        "privacy_filter": True,
        "user_tracking": True,
    },
    "search": {
        "expansion": {
            "enabled": False,
            "variants": 3,
        },
        "crag": {
            "enabled": False,
            "threshold": 0.3,
        },
        "reranker": {
            "enabled": False,
            "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "top_k": 20,
        },
        "hyde": {
            "enabled": False,
        },
        "contextual": {
            "enabled": False,
            "model": "",
        },
    },
    "organize": {
        "max_depth": 3,
        "min_subcategories": 2,
        "collapse_single_file": True,
    },
    "ignore": {
        "patterns": [],
    },
    "hooks": {},
    "paths": {
        "wiki_dir": None,
        "sources_dir": "sources",
        "output_dir": "wiki",
        "cache_dir": CACHE_DIR,
        "state_dir": KNOX_DIR,
    },
    "notes": {
        "file": "NOTES.md",
    },
}

# ── Env var mapping ──────────────────────────────────────────────────────
# Maps dotted config key → env var name.
# Only keys that make sense as env overrides are listed.

_ENV_MAP: dict[str, str] = {
    PATHS_WIKI_DIR: "KNOX_WIKI_DIR",
    PATHS_SOURCES_DIR: "KNOX_SOURCES_DIR",
    PATHS_OUTPUT_DIR: "KNOX_OUTPUT_DIR",
    PATHS_CACHE_DIR: "KNOX_CACHE_DIR",
    PATHS_STATE_DIR: "KNOX_STATE_DIR",
    COMPILE_FORMAT: "KNOX_FORMAT",
    COMPILE_CHUNK_SIZE: "KNOX_CHUNK_SIZE",  # deprecated — derived from context window
    COMPILE_IMAGE_MODE: "KNOX_IMAGE_MODE",
    COMPILE_START_AT_TIER: "KNOX_START_AT_TIER",
    COMPILE_WORKERS: "KNOX_WORKERS",
    COMPILE_NO_CACHE: "KNOX_NO_CACHE",
    COMPILE_FORCE: "KNOX_FORCE",
    COMPILE_IN_PLACE_SOURCES: "KNOX_IN_PLACE_SOURCES",
    COMPILE_LINT: "KNOX_LINT",
    COMPILE_PAGE_KIND: "KNOX_PAGE_KIND",
    COMPILE_BUDGET_CONCEPTS_PER_BATCH: "KNOX_BUDGET_CONCEPTS_PER_BATCH",
    COMPILE_BUDGET_WICER_MAX_ITERATIONS: "KNOX_BUDGET_WICER_MAX_ITERATIONS",
    COMPILE_BUDGET_MIN_CONTENT_LENGTH: "KNOX_BUDGET_MIN_CONTENT_LENGTH",
    COMPILE_BUDGET_MIN_CONCEPT_SOURCES: "KNOX_BUDGET_MIN_CONCEPT_SOURCES",
    COMPILE_BUDGET_MAX_CONCEPT_PAGES: "KNOX_BUDGET_MAX_CONCEPT_PAGES",
    COMPILE_BUDGET_CHUNK_SIZE_MIN: "KNOX_BUDGET_CHUNK_SIZE_MIN",
    COMPILE_BUDGET_CHUNK_SIZE_MAX: "KNOX_BUDGET_CHUNK_SIZE_MAX",
    COMPILE_BUDGET_EXTRACT_OUTPUT_MAX: "KNOX_BUDGET_EXTRACT_OUTPUT_MAX",
    COMPILE_BUDGET_GEN_OUTPUT_MAX: "KNOX_BUDGET_GEN_OUTPUT_MAX",
    COMPILE_BUDGET_OVERVIEW_OUTPUT_MAX: "KNOX_BUDGET_OVERVIEW_OUTPUT_MAX",
    COMPILE_LLM_DEFAULT_MODEL: "KNOX_LLM_DEFAULT_MODEL",
    COMPILE_LLM_VLM_MODEL: "KNOX_VLM_MODEL",
    COMPILE_LLM_DATA_FORMAT: "KNOX_LLM_DATA_FORMAT",
    COMPILE_MEMORY_ENABLED: "KNOX_MEMORY_ENABLED",
    COMPILE_MEMORY_HALF_LIFE_DAYS: "KNOX_MEMORY_HALF_LIFE_DAYS",
    COMPILE_MEMORY_DECAY_THRESHOLD: "KNOX_MEMORY_DECAY_THRESHOLD",
    COMPILE_LLM_FALLBACK_CHAIN: "KNOX_LLM_FALLBACK_CHAIN",
    COMPILE_LLM_FALLBACK_MAX_RETRIES: "KNOX_LLM_FALLBACK_MAX_RETRIES",
    COMPILE_LLM_TIMEOUT: "KNOX_LLM_TIMEOUT",
    COMPILE_LLM_FALLBACK_MAX_INPUT: "KNOX_LLM_FALLBACK_MAX_INPUT",
    COMPILE_LLM_FALLBACK_MAX_OUTPUT: "KNOX_LLM_FALLBACK_MAX_OUTPUT",
    COMPILE_LLM_NUM_CTX_MIN: "KNOX_LLM_NUM_CTX_MIN",
    COMPILE_LLM_NUM_CTX_MAX: "KNOX_LLM_NUM_CTX_MAX",
    COMPILE_LLM_CONTEXT_WINDOW_CAP: "KNOX_LLM_CONTEXT_WINDOW_CAP",
    COMPILE_OUTPUT_FLAVOR: "KNOX_OUTPUT_FLAVOR",
    COMPILE_OUTPUT_RUMDL_CONFIG: "KNOX_RUMDL_CONFIG",
    COMPILE_OUTPUT_MAX_RELATED_LINKS: "KNOX_OUTPUT_MAX_RELATED_LINKS",
    COMPILE_PDF_MAX_PAGES_BEFORE_CHUNKING: "KNOX_PDF_MAX_PAGES_BEFORE_CHUNKING",
    COMPILE_PDF_PAGE_CHUNK_SIZE: "KNOX_PDF_PAGE_CHUNK_SIZE",
    COMPILE_PDF_PAGE_CHUNK_OVERLAP: "KNOX_PDF_PAGE_CHUNK_OVERLAP",
    COMPILE_EMBEDDING_MODEL: "KNOX_EMBEDDING_MODEL",
    COMPILE_EMBEDDING_DIM: "KNOX_EMBEDDING_DIM",  # deprecated — auto-detected from model
    COMPILE_EMBEDDING_BATCH_SIZE: "KNOX_EMBEDDING_BATCH_SIZE",
    TRUST_PRIVACY_FILTER: "KNOX_PRIVACY_FILTER",
    TRUST_USER_TRACKING: "KNOX_USER_TRACKING",
    SEARCH_EXPANSION_ENABLED: "KNOX_SEARCH_EXPANSION_ENABLED",
    SEARCH_EXPANSION_VARIANTS: "KNOX_SEARCH_EXPANSION_VARIANTS",
    SEARCH_CRAG_ENABLED: "KNOX_SEARCH_CRAG_ENABLED",
    SEARCH_CRAG_THRESHOLD: "KNOX_SEARCH_CRAG_THRESHOLD",
    SEARCH_RERANKER_ENABLED: "KNOX_SEARCH_RERANKER_ENABLED",
    SEARCH_RERANKER_MODEL: "KNOX_SEARCH_RERANKER_MODEL",
    SEARCH_RERANKER_TOP_K: "KNOX_SEARCH_RERANKER_TOP_K",
    SEARCH_HYDE_ENABLED: "KNOX_SEARCH_HYDE_ENABLED",
    SEARCH_CONTEXTUAL_ENABLED: "KNOX_SEARCH_CONTEXTUAL_ENABLED",
    SEARCH_CONTEXTUAL_MODEL: "KNOX_SEARCH_CONTEXTUAL_MODEL",
    "eval.threshold": "KNOX_EVAL_THRESHOLD",
    "notes.file": "KNOX_NOTES_FILE",
    ORGANIZE_MAX_DEPTH: "KNOX_ORGANIZE_MAX_DEPTH",
    ORGANIZE_MIN_SUBCATEGORIES: "KNOX_ORGANIZE_MIN_SUBCATEGORIES",
    ORGANIZE_COLLAPSE_SINGLE_FILE: "KNOX_ORGANIZE_COLLAPSE_SINGLE_FILE",
    DEDUP_ENABLED: "KNOX_DEDUP_ENABLED",
    DEDUP_THRESHOLD_IDENTICAL: "KNOX_DEDUP_THRESHOLD_IDENTICAL",
    DEDUP_THRESHOLD_SIMILAR: "KNOX_DEDUP_THRESHOLD_SIMILAR",
    DEDUP_MODEL: "KNOX_DEDUP_MODEL",
    DEDUP_MATCHERS: "KNOX_DEDUP_MATCHERS",
    DEDUP_MINHASH_LSH_THRESHOLD: "KNOX_DEDUP_MINHASH_LSH_THRESHOLD",
}


# ── Deep merge ──────────────────────────────────────────────────────────


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *override* into *base*, returning a new dict.

    Dict values are merged recursively. Non-dict values from *override*
    replace those in *base*. All values are deep-copied so the returned
    dict never shares references with the inputs.
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value) if isinstance(value, dict) else value
    return result


def _nest_flat(flat: dict[str, Any]) -> dict[str, Any]:
    """Convert a flat dotted-key dict to a nested dict."""
    result: dict[str, Any] = {}
    for key, value in flat.items():
        _set_dot(result, key, value)
    return result


# ── Dot-path access ─────────────────────────────────────────────────────


def _get_dot(data: dict[str, Any], key: str, default: Any = None) -> Any:
    """Retrieve a dotted key from a nested dict."""
    parts = key.split(".")
    current: Any = data
    for part in parts:
        if isinstance(current, dict):
            current = current.get(part)
        else:
            return default
    return current if current is not None else default


def _set_dot(data: dict[str, Any], key: str, value: Any) -> None:
    """Set a dotted key in a nested dict, creating intermediate dicts."""
    parts = key.split(".")
    current = data
    for part in parts[:-1]:
        if part not in current or not isinstance(current[part], dict):
            current[part] = {}
        current = current[part]
    current[parts[-1]] = value


# ── Config file discovery ───────────────────────────────────────────────


def find_project_config(start_dir: Path | None = None) -> Path | None:
    """Walk up from *start_dir* looking for project config files.

    Prefers ``.knox/config.toml`` over ``knox.toml`` in the same directory.
    Returns the first found path, or ``None``.
    """
    current = (start_dir or Path.cwd()).resolve()
    for parent in [current] + list(current.parents):
        for name in _PROJECT_CONFIG_NAMES:
            candidate = parent / name
            if candidate.is_file():
                return candidate
    return None


def find_all_project_configs(start_dir: Path | None = None) -> list[Path]:
    """Walk up from *start_dir* collecting all project config files.

    Returns a list ordered from closest to farthest from *start_dir*.
    Within each directory, ``.knox/config.toml`` is preferred over
    ``knox.toml``; only one file per directory is returned.
    """
    current = (start_dir or Path.cwd()).resolve()
    found: list[Path] = []
    seen_dirs: set[Path] = set()
    for parent in [current] + list(current.parents):
        if parent in seen_dirs:
            continue
        seen_dirs.add(parent)
        for name in _PROJECT_CONFIG_NAMES:
            candidate = parent / name
            if candidate.is_file():
                found.append(candidate)
                break
    return found


def find_global_config() -> Path | None:
    """Return the global config path if it exists.

    Location follows the XDG convention:
    ``$XDG_CONFIG_HOME/knox/config.toml`` (defaults to
    ``~/.config/knox/config.toml``).
    """
    xdg = os.environ.get("XDG_CONFIG_HOME")
    config_dir = Path(xdg) if xdg else Path.home() / ".config"
    candidate = config_dir / _GLOBAL_CONFIG_DIR / _GLOBAL_CONFIG_NAME
    return candidate if candidate.is_file() else None


# ── TOML loading ────────────────────────────────────────────────────────


def _expand_env_vars(value: str) -> str:
    """Expand ``$VAR`` and ``${VAR}`` patterns in a string.

    Unknown variables are left as-is. This is applied to string values
    when loading TOML config files so users can reference env vars like
    ``$HOME`` in path settings.
    """

    def _replace(match: re.Match) -> str:
        var = match.group(1) or match.group(2) or ""
        return os.environ.get(var, match.group(0))

    return re.sub(r"\$([A-Z_][A-Z0-9_]*|\{[A-Z_][A-Z0-9_]*\})|\$\{(\w+)", _replace, value)


def _expand_env_vars_recursive(data: Any) -> Any:
    """Recursively expand env vars in all string values."""
    if isinstance(data, str):
        return _expand_env_vars(data)
    if isinstance(data, dict):
        return {k: _expand_env_vars_recursive(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_expand_env_vars_recursive(v) for v in data]
    return data


def _strip_strings_recursive(data: Any) -> Any:
    """Recursively strip leading/trailing whitespace from all string values."""
    if isinstance(data, str):
        return data.strip()
    if isinstance(data, dict):
        return {k: _strip_strings_recursive(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_strip_strings_recursive(v) for v in data]
    return data


def _load_toml(toml_file: Path) -> dict[str, Any]:
    """Load a TOML file, returning an empty dict on failure.

    String values are processed through ``_expand_env_vars`` to support
    ``$VAR`` and ``${VAR}`` patterns.
    """
    try:
        raw = toml_file.read_bytes()
        data: dict[str, Any] = tomllib.loads(raw.decode("utf-8"))
        data = _expand_env_vars_recursive(data)
        data = _strip_strings_recursive(data)
        logger.debug("Loaded config from %s", toml_file)
        return data
    except Exception as exc:
        logger.warning("Failed to load config %s: %s", toml_file, exc, exc_info=True)
        return {}


# ── Env var layer ───────────────────────────────────────────────────────


# Build pydantic-settings model dynamically from _ENV_MAP.
# Field names are KNOX_* suffixes lowercased; types inferred from patterns.
_KnoxEnvFields: dict[str, tuple[Any, Any]] = {}
for _env_name in _ENV_MAP.values():
    _fn = _env_name.removeprefix("KNOX_").lower()
    # Comma-separated lists (stored as str, split in _read_env_vars)
    if _env_name in ("KNOX_LLM_FALLBACK_CHAIN", "KNOX_LINT", "KNOX_DEDUP_MATCHERS", "KNOX_SEARCH_EXPANSION_VARIANTS"):
        _KnoxEnvFields[_fn] = (str | None, Field(None))
    # Booleans
    elif _env_name.endswith("_ENABLED") or _env_name in (
        "KNOX_NO_CACHE",
        "KNOX_FORCE",
        "KNOX_IN_PLACE_SOURCES",
        "KNOX_MEMORY_ENABLED",
        "KNOX_PRIVACY_FILTER",
        "KNOX_USER_TRACKING",
        "KNOX_ORGANIZE_COLLAPSE_SINGLE_FILE",
        "KNOX_DEDUP_ENABLED",
    ):
        _KnoxEnvFields[_fn] = (bool | None, Field(None))
    # Numeric: check float patterns first, then int
    elif "_THRESHOLD" in _env_name or "_HALF_LIFE_" in _env_name or "_DECAY_" in _env_name:
        _KnoxEnvFields[_fn] = (float | None, Field(None))
    elif _env_name in (
        "KNOX_CHUNK_SIZE",
        "KNOX_WORKERS",
        "KNOX_LINT_THRESHOLD",
        "KNOX_SEARCH_RERANKER_TOP_K",
        "KNOX_SEARCH_CRAG_THRESHOLD",
        "KNOX_EVAL_THRESHOLD",
        "KNOX_ORGANIZE_MAX_DEPTH",
        "KNOX_ORGANIZE_MIN_SUBCATEGORIES",
        "KNOX_EMBEDDING_DIM",
        "KNOX_EMBEDDING_BATCH_SIZE",
        "KNOX_LLM_NUM_CTX_MIN",
        "KNOX_LLM_NUM_CTX_MAX",
        "KNOX_LLM_FALLBACK_MAX_RETRIES",
        "KNOX_BUDGET_CONCEPTS_PER_BATCH",
        "KNOX_BUDGET_WICER_MAX_ITERATIONS",
        "KNOX_BUDGET_MIN_CONTENT_LENGTH",
        "KNOX_BUDGET_MIN_CONCEPT_SOURCES",
        "KNOX_BUDGET_MAX_CONCEPT_PAGES",
        "KNOX_BUDGET_CHUNK_SIZE_MIN",
        "KNOX_BUDGET_CHUNK_SIZE_MAX",
        "KNOX_BUDGET_EXTRACT_OUTPUT_MAX",
        "KNOX_BUDGET_GEN_OUTPUT_MAX",
        "KNOX_BUDGET_OVERVIEW_OUTPUT_MAX",
        "KNOX_LLM_CONTEXT_WINDOW_CAP",
        "KNOX_PDF_MAX_PAGES_BEFORE_CHUNKING",
        "KNOX_PDF_PAGE_CHUNK_SIZE",
        "KNOX_PDF_PAGE_CHUNK_OVERLAP",
        "KNOX_DEDUP_THRESHOLD_IDENTICAL",
        "KNOX_DEDUP_THRESHOLD_SIMILAR",
        "KNOX_LLM_TIMEOUT",
        "KNOX_LLM_FALLBACK_MAX_INPUT",
        "KNOX_LLM_FALLBACK_MAX_OUTPUT",
        "KNOX_MEMORY_HALF_LIFE_DAYS",
        "KNOX_MEMORY_DECAY_THRESHOLD",
    ):
        _KnoxEnvFields[_fn] = (int | None, Field(None))
    else:
        _KnoxEnvFields[_fn] = (str | None, Field(None))

_KnoxEnvSettingsModel = create_model(
    "_KnoxEnvSettings",
    __base__=BaseSettings,
    __config__=SettingsConfigDict(env_prefix="KNOX_", extra="ignore"),
    **_KnoxEnvFields,
)


def _read_env_vars() -> dict[str, Any]:
    """Read KNOX_* env vars via pydantic-settings and return a nested config dict."""
    settings = _KnoxEnvSettingsModel()
    result: dict[str, Any] = {}
    for dotted_key, env_name in _ENV_MAP.items():
        field_name = env_name.removeprefix("KNOX_").lower()
        value = getattr(settings, field_name, None)
        if value is not None:
            # Strip whitespace for string values
            if isinstance(value, str):
                value = value.strip()
            # Comma-separated list fields
            if env_name in ("KNOX_LLM_FALLBACK_CHAIN", "KNOX_LINT", "KNOX_DEDUP_MATCHERS", "KNOX_SEARCH_EXPANSION_VARIANTS"):
                if isinstance(value, str):
                    value = [v.strip() for v in value.split(",") if v.strip()]
                elif isinstance(value, list):
                    pass  # already a list
            _set_dot(result, dotted_key, value)
    return result


# ── ResolvedConfig ──────────────────────────────────────────────────────


@dataclass
class ConfigSource:
    """Tracks where each config value came from."""

    path: str  # "default", "global", "project", "env", "cli"
    location: str = ""  # file path or env var name

    def __str__(self) -> str:
        if self.location:
            return f"{self.path} ({self.location})"
        return self.path


@dataclass
class ResolvedConfig:
    """Fully merged configuration with provenance tracking.

    Layers are applied in order: defaults → global → project → env → cli.
    Each value carries a :class:`ConfigSource` so ``knox config show``
    can report where it came from.
    """

    raw: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, ConfigSource] = field(default_factory=dict)
    project_config_file: Path | None = None
    project_config_paths: list[Path] = field(default_factory=list)
    global_config_file: Path | None = None

    def get(self, key: str, default: Any = None) -> Any:
        """Get a dotted key from the resolved config."""
        return _get_dot(self.raw, key, default)

    def source_of(self, key: str) -> ConfigSource:
        """Return where a specific key's value came from."""
        return self.provenance.get(key, ConfigSource("default"))

    def flat(self) -> dict[str, Any]:
        """Return a flat dotted-key dict of all values."""
        result: dict[str, Any] = {}
        self._flatten(self.raw, "", result)
        return result

    def _flatten(self, data: dict[str, Any], prefix: str, out: dict[str, Any]) -> None:
        for key, value in data.items():
            full = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                self._flatten(value, full, out)
            else:
                out[full] = value


def _merge_layer(
    base: dict[str, Any],
    layer: dict[str, Any],
    provenance: dict[str, ConfigSource],
    source: ConfigSource,
) -> dict[str, Any]:
    """Merge a layer into base, recording provenance for new/changed keys."""
    result = _deep_merge(base, layer)
    # Record provenance for keys this layer contributed
    for key in _flat_keys(layer):
        if _get_dot(layer, key) is not None:
            provenance[key] = source
    return result


def _merge_project_configs(
    project_configs: list[Path],
) -> tuple[dict[str, Any], dict[str, ConfigSource]]:
    """Merge a project config hierarchy (farthest-to-closest) into one layer.

    Returns the merged data and per-key provenance pointing at the winning file.
    Closest file wins, matching ``find_all_project_configs`` ordering.
    """
    merged: dict[str, Any] = {}
    prov: dict[str, ConfigSource] = {}
    for config_file in reversed(project_configs):  # closest wins
        file_data = _load_toml(config_file)
        for key in _flat_keys(file_data):
            prov[key] = ConfigSource("project", str(config_file))
        merged = _deep_merge(merged, file_data)
    return merged, prov


def _flat_keys(data: dict[str, Any], prefix: str = "") -> list[str]:
    """Return all dotted keys in a nested dict."""
    keys: list[str] = []
    for key, value in data.items():
        full = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            keys.extend(_flat_keys(value, full))
        else:
            keys.append(full)
    return keys


def _resolve_paths(
    result: dict[str, Any],
    provenance: dict[str, ConfigSource],
) -> None:
    """Normalize path values under ``paths``.

    - Migrates deprecated ``compile.sources_dir`` / ``compile.output_dir``
      values to ``paths.sources_dir`` / ``paths.output_dir`` when the new
      keys were not explicitly set.
    - Resolves relative paths against ``paths.wiki_dir`` (or cwd).
    """
    paths = result.setdefault("paths", {})

    # Backwards compatibility: old [compile] path keys migrate to [paths]
    for old_key, new_key in (
        (COMPILE_SOURCES_DIR, PATHS_SOURCES_DIR),
        (COMPILE_OUTPUT_DIR, PATHS_OUTPUT_DIR),
    ):
        old_val = _get_dot(result, old_key)
        new_val = _get_dot(result, new_key)
        old_default = _get_dot(DEFAULTS, old_key)
        new_default = _get_dot(DEFAULTS, new_key)
        if old_val is not None and old_val != old_default and new_val == new_default:
            _set_dot(result, new_key, old_val)
            provenance[new_key] = ConfigSource(
                provenance.get(old_key, ConfigSource("project")).path,
                f"deprecated {old_key}",
            )

    wiki_dir_raw = paths.get("wiki_dir")
    base_dir = Path(wiki_dir_raw).expanduser().resolve() if wiki_dir_raw else Path.cwd()
    paths["wiki_dir"] = base_dir

    for key in ("sources_dir", "output_dir", "cache_dir", "state_dir"):
        dotted = f"paths.{key}"
        value = paths.get(key)
        if value is None:
            value = _get_dot(DEFAULTS, dotted)
        resolved_dir = Path(value).expanduser()
        if not resolved_dir.is_absolute():
            resolved_dir = base_dir / resolved_dir
        paths[key] = resolved_dir.resolve()


def resolve_config(
    *,
    cli_overrides: dict[str, Any] | None = None,
    config_path: str | Path | None = None,
) -> ResolvedConfig:
    """Build a fully resolved config by merging all 5 layers.

    Parameters
    ----------
    cli_overrides:
        Flat dotted-key dict of CLI-provided values. Only non-``None``
        values should be included — the caller filters out unspecified
        flags.
    config_path:
        Explicit config file path (``--config`` flag). When set, this
        is treated as a project-level override — it replaces the
        auto-discovered project hierarchy but is still merged below CLI.

    Returns:
    -------
    ResolvedConfig
        The merged config with provenance.
    """
    provenance: dict[str, ConfigSource] = {}

    # Layer 5: defaults
    result: dict[str, Any] = _deep_merge({}, DEFAULTS)
    for key in _flat_keys(DEFAULTS):
        provenance[key] = ConfigSource("default")

    # Layer 4: global config
    global_path = find_global_config()
    if global_path is not None:
        global_data = _load_toml(global_path)
        if global_data:
            result = _merge_layer(result, global_data, provenance, ConfigSource("global", str(global_path)))

    # Layer 3: project config (explicit --config overrides auto-discovered)
    project_dir: Path | None = None
    project_configs: list[Path] = []
    if config_path is not None:
        project_dir = Path(config_path)
        if project_dir.is_file():
            project_data = _load_toml(project_dir)
            if project_data:
                result = _merge_layer(result, project_data, provenance, ConfigSource("project", str(project_dir)))
        else:
            logger.warning("Config file not found: %s", config_path)
    else:
        project_configs = find_all_project_configs()
        if project_configs:
            # Merge from farthest to closest so closest wins.
            merged_project, project_provenance = _merge_project_configs(project_configs)
            if merged_project:
                result = _deep_merge(result, merged_project)
                provenance.update(project_provenance)

    # Layer 2: env vars
    env_data = _read_env_vars()
    if env_data:
        # Record per-key provenance with specific env var names
        env_result = _deep_merge(result, env_data)
        for key in _flat_keys(env_data):
            env_name = _ENV_MAP.get(key, "KNOX_*")
            provenance[key] = ConfigSource("env", env_name)
        result = env_result

    # Layer 1: CLI overrides
    cli = cli_overrides or {}
    if cli:
        cli_nested = _nest_flat(cli)
        cli_result = _deep_merge(result, cli_nested)
        for key in cli:
            if cli[key] is not None:
                provenance[key] = ConfigSource("cli")
        result = cli_result

    _resolve_paths(result, provenance)

    return ResolvedConfig(
        raw=result,
        provenance=provenance,
        project_config_file=project_configs[0] if project_configs else project_dir,
        project_config_paths=project_configs,
        global_config_file=global_path,
    )


# ── Config mutation helpers ───────────────────────────────────────────────


def _flat_schema() -> set[str]:
    """Return all leaf dotted keys present in :data:`DEFAULTS`."""
    return set(_flat_keys(DEFAULTS))


def validate_config_key(key: str) -> bool:
    """Return ``True`` if *key* is a known leaf config key."""
    return key in _flat_schema()


_TRUE_VALUES = {"true", "1", "yes", "on"}
_FALSE_VALUES = {"false", "0", "no", "off"}


def _coerce_boolean(value: str, key: str) -> bool:
    """Coerce *value* to a boolean, raising ValueError on ambiguity."""
    lowered = value.lower()
    if lowered in _TRUE_VALUES:
        return True
    if lowered in _FALSE_VALUES:
        return False
    raise ValueError(f"expected boolean for {key}")


def _coerce_list(value: str) -> list[str]:
    """Coerce *value* to a comma-separated list."""
    return [] if value == "" else [v.strip() for v in value.split(",") if v.strip()]


def _coerce_generic(value: str) -> int | float | str:
    """Coerce *value* to int or float if possible, otherwise return as-is."""
    with contextlib.suppress(ValueError):
        return int(value)
    with contextlib.suppress(ValueError):
        return float(value)
    return value


_COERCE_HANDLERS: dict[type, Callable[[str, str], Any]] = {
    bool: _coerce_boolean,
    int: lambda v, _k: int(v),
    float: lambda v, _k: float(v),
    list: lambda v, _k: _coerce_list(v),
}


def coerce_config_value(value: str, key: str) -> Any:
    """Coerce a CLI string to the type expected by ``DEFAULTS`` for *key*.

    Raises:
        ValueError: when *value* cannot be coerced to the expected type.
    """
    value = value.strip()
    current = _get_dot(DEFAULTS, key)
    handler = _COERCE_HANDLERS.get(type(current))
    if handler is not None:
        return handler(value, key)

    lowered = value.lower()
    if lowered in _TRUE_VALUES:
        return True
    if lowered in _FALSE_VALUES:
        return False
    return _coerce_generic(value)


def get_project_config_path(start_dir: Path | None = None, *, create: bool = False) -> Path:
    """Return the project config file path used by ``knox config set``.

    Prefers ``.knox/config.toml`` over ``knox.toml``. If no project config
    exists and *create* is ``True``, returns ``<start_dir>/.knox/config.toml``
    and creates the parent directory.
    """
    current = (start_dir or Path.cwd()).resolve()
    found = find_project_config(current)
    if found is not None:
        return found
    target = VaultPaths(current).config_path
    if create:
        target.parent.mkdir(parents=True, exist_ok=True)
    return target


def get_global_config_path(*, create: bool = False) -> Path:
    """Return the global config file path.

    Location follows the XDG convention. When *create* is ``True``, the
    parent directory is created if it does not exist.
    """
    xdg = os.environ.get("XDG_CONFIG_HOME")
    config_dir = Path(xdg) if xdg else Path.home() / ".config"
    target = config_dir / _GLOBAL_CONFIG_DIR / _GLOBAL_CONFIG_NAME
    if create:
        target.parent.mkdir(parents=True, exist_ok=True)
    return target


def _atomic_write_toml(toml_file: Path, data: dict[str, Any]) -> None:
    """Write *data* to *toml_file* atomically, keeping a ``.bak`` copy."""
    if toml_file.is_file():
        backup = toml_file.with_suffix(toml_file.suffix + ".bak")
        backup.write_bytes(toml_file.read_bytes())
    tmp = toml_file.with_suffix(toml_file.suffix + ".tmp")
    tmp.write_text(tomli_w.dumps(data), encoding="utf-8")
    tmp.replace(toml_file)


def set_config_file_value(config_file: Path, key: str, value: Any) -> None:
    """Set *key* to *value* in *config_file*, creating the file if needed.

    Raises:
        KeyError: if *key* is not a known config key.
        ValueError: if *value* cannot be serialised as TOML.
    """
    if not validate_config_key(key):
        raise KeyError(f"unknown config key: {key}")
    data = _load_toml(config_file) if config_file.is_file() else {}
    _set_dot(data, key, value)
    _atomic_write_toml(config_file, data)
