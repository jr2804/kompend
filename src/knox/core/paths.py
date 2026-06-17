"""Single source of truth for all KNOX path component names.

These constants define the fixed directory and file names used throughout
the KNOX project structure. They must NEVER be hardcoded as string literals
elsewhere in the codebase.

Configurable directory names (sources_dir, output_dir) live in DEFAULTS
(``knox.core.config._core``) and are resolved through PipelineConfig, NOT here.

Usage::

    from knox.core.paths import KNOX_DIR, CACHE_SUBDIR, WIKI_CACHE_FILE

    cache_path = base_dir / KNOX_DIR / CACHE_SUBDIR
    wiki_cache = base_dir / KNOX_DIR / WIKI_CACHE_FILE

Naming convention for the project root directory: **always** use ``base_dir``.
Never use ``target_dir``, ``resolved_base``, ``output_dir``, or
``sources_dir.parent`` to refer to the project root — these are derived
quantities or refer to different concepts.
"""

from __future__ import annotations

from typing import Final

# ── Hidden KNOX artifacts folder ────────────────────────────────────────
KNOX_DIR: Final[str] = ".knox"

# ── Config ──────────────────────────────────────────────────────────────
CONFIG_FILE: Final[str] = "config.toml"
KNOX_CONFIG_FILE: Final[str] = f"{KNOX_DIR}/{CONFIG_FILE}"

# ── Cache ───────────────────────────────────────────────────────────────
CACHE_SUBDIR: Final[str] = "cache"
CACHE_DIR: Final[str] = f"{KNOX_DIR}/{CACHE_SUBDIR}"

# ── Wiki cache (serialised CompiledWiki) ────────────────────────────────
WIKI_CACHE_FILE: Final[str] = "last_wiki.json"

# ── Review queue ────────────────────────────────────────────────────────
REVIEW_DIR: Final[str] = "review"

# ── Vector store ────────────────────────────────────────────────────────
VECTORS_DIR: Final[str] = "vectors"

# ── Search database ─────────────────────────────────────────────────────
SEARCH_DB_FILE: Final[str] = "search.db"

# ── Download tracking ───────────────────────────────────────────────────
DOWNLOADED_FILE: Final[str] = "downloaded.json"

# ── Memory ──────────────────────────────────────────────────────────────
MEMORY_DB_FILE: Final[str] = "memory.db"
MEMORY_LOG_FILE: Final[str] = "memory.jsonl"
MEMORY_STORE_FILE: Final[str] = "memory.json"

# ── Provenance ──────────────────────────────────────────────────────────
PROVENANCE_DIR: Final[str] = "provenance"

# ── Verification ────────────────────────────────────────────────────────
VERIFICATION_FILE: Final[str] = "verification.json"

# ── Trust / Privacy ─────────────────────────────────────────────────────
PRIVATE_FILE: Final[str] = "private.json"

# ── Pipeline state ──────────────────────────────────────────────────────
COMPILE_STATE_FILE: Final[str] = "compile_state.json"
STATE_FILE: Final[str] = "state.json"
