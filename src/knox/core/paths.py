"""Single source of truth for all KNOX path component names and vault paths.

Path *names* are ``Final[str]`` constants (``KNOX_DIR``, ``CACHE_SUBDIR``, …).
Resolved paths for a specific vault live in the :class:`VaultPaths` dataclass,
which takes a ``base_dir`` and exposes every artifact path as a property.

Usage — names (for scaffolding / string-level operations)::

    from knox.core.paths import KNOX_DIR, CACHE_SUBDIR

    knox_path = target / KNOX_DIR

Usage — resolved paths (preferred for all runtime code)::

    from knox.core.paths import VaultPaths

    paths = VaultPaths(base_dir)
    cache_file = paths.cache_dir / "my_data.json"

Naming convention for the project root directory: **always** use ``base_dir``.
Never use ``target_dir``, ``resolved_base``, ``output_dir``, or
``sources_dir.parent`` to refer to the project root — these are derived
quantities or refer to different concepts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

# ── Hidden KNOX artifacts folder ────────────────────────────────────────
KNOX_DIR: Final[str] = ".knox"

# ── Config ──────────────────────────────────────────────────────────────
CONFIG_FILE: Final[str] = "config.toml"
KNOX_CONFIG_FILE: Final[str] = f"{KNOX_DIR}/{CONFIG_FILE}"
PROJECT_CONFIG_FILE: Final[str] = "knox.toml"

# ── Cache ───────────────────────────────────────────────────────────────
CACHE_SUBDIR: Final[str] = "cache"
CACHE_DIR: Final[str] = f"{KNOX_DIR}/{CACHE_SUBDIR}"
IMAGES_SUBDIR: Final[str] = "images"
EXTRACT_SUBDIR: Final[str] = "extract"

# ── Wiki cache (serialised CompiledWiki) ────────────────────────────────
WIKI_CACHE_FILE: Final[str] = "last_wiki.json"
COMPILED_WIKI_FILE: Final[str] = "compiled_wiki.json"
INDEX_FILE: Final[str] = "index.md"

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

# ── Logging ─────────────────────────────────────────────────────────────
LOG_FILE: Final[str] = "log.md"

# ── Trust / Privacy ─────────────────────────────────────────────────────
PRIVATE_FILE: Final[str] = "private.json"

# ── Pipeline state ──────────────────────────────────────────────────────
COMPILE_STATE_FILE: Final[str] = "compile_state.json"
STATE_FILE: Final[str] = "state.json"

# ── Lint history (append-only JSONL) ─────────────────────────────────────
LINT_HISTORY_FILE: Final[str] = "lint_history.jsonl"

# ── Organize (LLM-based file categorisation) ─────────────────────────────
ORGANIZE_REPORT_FILE: Final[str] = "last_organize.json"

# ── Dedup (content-level duplicate detection) ────────────────────────────
DEDUP_DIR: Final[str] = "dedup"

# ── Default configurable directory names ────────────────────────────────
DEFAULT_SOURCES_DIR_NAME: Final[str] = "sources"
DEFAULT_OUTPUT_DIR_NAME: Final[str] = "wiki"


@dataclass(frozen=True)
class VaultPaths:
    """All resolved paths for a KNOX vault.

    Given a ``base_dir`` (the project root), every artifact path is available
    as a read-only property.  This eliminates the repeated
    ``base_dir / KNOX_DIR / SOMETHING`` pattern scattered across modules.

    Configurable directory names (``sources``, ``wiki``) come from
    :data:`DEFAULTS` and can be overridden via the constructor.

    Directory layout::

        base_dir/
        ├── .knox/                  ← knox_dir
        │   ├── cache/              ← cache_dir
        │   │   ├── images/         ← images_dir
        │   │   └── extract/        ← extract_dir
        │   ├── dedup/              ← dedup_dir
        │   ├── provenance/         ← provenance_dir
        │   ├── review/             ← review_dir
        │   ├── vectors/            ← vectors_dir
        │   ├── config.toml         ← config_path
        │   ├── compile_state.json  ← compile_state_path
        │   ├── last_wiki.json      ← wiki_cache_path
        │   ├── compiled_wiki.json  ← compiled_wiki_path
        │   ├── search.db           ← search_db_path
        │   ├── memory.db           ← memory_db_path
        │   ├── memory.jsonl        ┆ memory_log_path
        │   ├── memory.json         ← memory_store_path
        │   ├── verification.json   ← verification_path
        │   ├── private.json        ← private_path
        │   ├── downloaded.json     ← downloaded_path
    │   ├── last_organize.json  ← organize_report_path
    │   ├── lint_history.jsonl  ← lint_history_path
    │   └── state.json          ← state_file
        ├── sources/                ← sources_dir
        ├── wiki/                   ← output_dir
        │   └── index.md            ← index_file
    """

    base_dir: Path
    sources_dir_name: str = DEFAULT_SOURCES_DIR_NAME
    output_dir_name: str = DEFAULT_OUTPUT_DIR_NAME

    # ── Top-level directories ───────────────────────────────────────────

    @property
    def sources_dir(self) -> Path:
        """User-owned source files directory."""
        return self.base_dir / self.sources_dir_name

    @property
    def output_dir(self) -> Path:
        """Generated wiki output directory."""
        return self.base_dir / self.output_dir_name

    @property
    def knox_dir(self) -> Path:
        """Internal ``.knox`` state directory."""
        return self.base_dir / KNOX_DIR

    # ── Sub-directories of .knox/ ───────────────────────────────────────

    @property
    def cache_dir(self) -> Path:
        """Document parse / compilation cache directory."""
        return self.knox_dir / CACHE_SUBDIR

    @property
    def images_dir(self) -> Path:
        """Extracted images directory (inside cache)."""
        return self.cache_dir / IMAGES_SUBDIR

    @property
    def extract_dir(self) -> Path:
        """LLM extraction cache directory (inside cache)."""
        return self.cache_dir / EXTRACT_SUBDIR

    @property
    def dedup_dir(self) -> Path:
        """Content-level duplicate detection fingerprints directory."""
        return self.knox_dir / DEDUP_DIR

    @property
    def provenance_dir(self) -> Path:
        """Per-node provenance JSON directory."""
        return self.knox_dir / PROVENANCE_DIR

    @property
    def review_dir(self) -> Path:
        """Review queue directory."""
        return self.knox_dir / REVIEW_DIR

    @property
    def vectors_dir(self) -> Path:
        """Embedding vector store directory."""
        return self.knox_dir / VECTORS_DIR

    # ── Files inside .knox/ ─────────────────────────────────────────────

    @property
    def config_path(self) -> Path:
        """Project-level config file (``.knox/config.toml``)."""
        return self.knox_dir / CONFIG_FILE

    @property
    def compile_state_path(self) -> Path:
        """Persistent tier-state checkpoint file."""
        return self.knox_dir / COMPILE_STATE_FILE

    @property
    def wiki_cache_path(self) -> Path:
        """Serialised ``CompiledWiki`` snapshot."""
        return self.knox_dir / WIKI_CACHE_FILE

    @property
    def compiled_wiki_path(self) -> Path:
        """Full compiled wiki JSON (used by pipeline internally)."""
        return self.knox_dir / COMPILED_WIKI_FILE

    @property
    def search_db_path(self) -> Path:
        """SQLite search index database."""
        return self.knox_dir / SEARCH_DB_FILE

    @property
    def memory_db_path(self) -> Path:
        """SQLite memory store database."""
        return self.knox_dir / MEMORY_DB_FILE

    @property
    def memory_log_path(self) -> Path:
        """Append-only memory log."""
        return self.knox_dir / MEMORY_LOG_FILE

    @property
    def memory_store_path(self) -> Path:
        """Memory JSON export."""
        return self.knox_dir / MEMORY_STORE_FILE

    @property
    def verification_path(self) -> Path:
        """Verification results JSON."""
        return self.knox_dir / VERIFICATION_FILE

    @property
    def private_path(self) -> Path:
        """Private page manifest."""
        return self.knox_dir / PRIVATE_FILE

    @property
    def downloaded_path(self) -> Path:
        """URL download manifest (inside cache)."""
        return self.cache_dir / DOWNLOADED_FILE

    @property
    def organize_report_path(self) -> Path:
        """Last ``knox organize`` report."""
        return self.knox_dir / ORGANIZE_REPORT_FILE

    @property
    def state_file(self) -> Path:
        """General state file."""
        return self.knox_dir / STATE_FILE

    @property
    def log_file(self) -> Path:
        """Pipeline log file (prepended after each stage)."""
        return self.knox_dir / LOG_FILE

    @property
    def lint_history_path(self) -> Path:
        """Append-only JSONL log of lint runs."""
        return self.knox_dir / LINT_HISTORY_FILE

    # ── Files inside output_dir/ ────────────────────────────────────────

    @property
    def index_file(self) -> Path:
        """Wiki entry point (table of contents)."""
        return self.output_dir / INDEX_FILE

    # ── Files at project root ───────────────────────────────────────────

    @property
    def project_config_file(self) -> Path:
        """User project config file (``knox.toml`` at root)."""
        return self.base_dir / PROJECT_CONFIG_FILE

    # ── Directory lifecycle ─────────────────────────────────────────────

    def ensure_dirs(self) -> None:
        """Create all standard directories if they do not exist.

        Creates: ``sources_dir``, ``output_dir``, ``knox_dir``, ``cache_dir``,
        ``images_dir``, ``extract_dir``, ``provenance_dir``, ``review_dir``,
        ``vectors_dir``.
        """
        for directory in (
            self.sources_dir,
            self.output_dir,
            self.knox_dir,
            self.cache_dir,
            self.dedup_dir,
            self.images_dir,
            self.extract_dir,
            self.provenance_dir,
            self.review_dir,
            self.vectors_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
