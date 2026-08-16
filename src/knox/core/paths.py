"""Single source of truth for all KNOX path component names and vault paths.

Path *names* are ``Final[str]`` constants (``KNOX_DIR``, ``CACHE_SUBDIR``, …).
Resolved paths for a specific vault live in the :class:`VaultPaths` dataclass,
which takes a ``base_dir`` and exposes every artifact path as a property.

Usage — names (for scaffolding / string-level operations)::

    from knox.core.paths import KNOX_DIR, CACHE_SUBDIR

    knox_dir = target / KNOX_DIR

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

import os
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Final

from knox.core.env import EnvVar

# ── Hidden KNOX artifacts folder ────────────────────────────────────────
KNOX_DIR: Final[str] = ".knox"

# ── Capture (learn/capture/distill inputs live under sources/<subdir>) ──
CAPTURED_SUBDIR: Final[str] = "_captured"
DISTILLED_SUBDIR: Final[str] = "_distilled"
# Subdirectories under sources/ that `knox organize` must NEVER flatten or
# relocate. These are tool-managed input zones, not topically-categorized
# material. Compile still processes them (they are legitimate sources);
# organize leaves them in place.
PROTECTED_SOURCE_SUBDIRS: Final[frozenset[str]] = frozenset({CAPTURED_SUBDIR, DISTILLED_SUBDIR})

# ── Config ──────────────────────────────────────────────────────────────
CONFIG_FILE: Final[str] = "config.toml"
KNOX_CONFIG_FILE: Final[str] = f"{KNOX_DIR}/{CONFIG_FILE}"
PROJECT_CONFIG_FILE: Final[str] = "knox.toml"
RUMDL_CONFIG_FILE: Final[str] = f"{KNOX_DIR}/rumdl.toml"

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
RUNS_FILE: Final[str] = "runs.jsonl"

# ── Organize (LLM-based file categorisation) ─────────────────────────────
ORGANIZE_REPORT_FILE: Final[str] = "last_organize.json"

# ── References (bibliographic citation extraction) ───────────────────────
REFERENCES_FILE: Final[str] = "references.json"

# ── Assets (separated tables/figures) ────────────────────────────────────
ASSETS_FILE: Final[str] = "assets.json"

# ── Attachments (user-provided supplementary files per source) ───────────
ATTACHMENTS_DIR_NAME: Final[str] = "attachments"
ATTACHMENTS_MANIFEST: Final[str] = "attachments.jsonl"

# ── Dedup (content-level duplicate detection) ────────────────────────────
DEDUP_DIR: Final[str] = "dedup"

# ── Default configurable directory names ────────────────────────────────
DEFAULT_SOURCES_DIR_NAME: Final[str] = "sources"
DEFAULT_OUTPUT_DIR_NAME: Final[str] = "wiki"

# ── Attachment slug generation ───────────────────────────────────────────
_SLUG_INVALID_RE: Final = __import__("re").compile(r"[^a-z0-9_-]+")
_SLUG_UNDERSCORE_RE: Final = __import__("re").compile(r"_+")


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
        │   ├── config.toml         ← config_file
        │   ├── compile_state.json  ← compile_state_file
        │   ├── last_wiki.json      ← wiki_cache_file
        │   ├── compiled_wiki.json  ← compiled_wiki_file
        │   ├── search.db           ← search_db_file
        │   ├── memory.db           ← memory_db_file
        │   ├── memory.jsonl        ┆ memory_log_file
        │   ├── memory.json         ← memory_store_file
        │   ├── verification.json   ← verification_file
        │   ├── private.json        ← private_file
        │   ├── downloaded.json     ← downloaded_file
    │   ├── last_organize.json  ← organize_report_file
    │   ├── lint_history.jsonl  ← lint_history_file
    │   ├── references.json     ← references_file
    │   ├── attachments.jsonl   ← attachments_manifest_file
    │   └── state.json          ← state_file
        ├── sources/                ← sources_dir
        ├── attachments/            ← attachments_dir (per-source)
        ├── wiki/                   ← output_dir
        │   └── index.md            ← index_file
    """

    base_dir: Path
    sources_dir_name: str = DEFAULT_SOURCES_DIR_NAME
    output_dir_name: str = DEFAULT_OUTPUT_DIR_NAME

    def __post_init__(self) -> None:
        """Normalize base_dir to an absolute, resolved path.

        Accepts ``None``, ``"."``, a relative str, or a ``Path`` — making
        :class:`VaultPaths` the single constructor that turns any input into
        a canonical vault root. No other code should call ``Path.cwd()`` or
        ``Path(x).resolve()`` on a wiki root.
        """
        resolved = Path.cwd().resolve() if self.base_dir is None or self.base_dir == Path("") else Path(self.base_dir).expanduser().resolve()
        object.__setattr__(self, "base_dir", resolved)

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

    @property
    def attachments_dir(self) -> Path:
        """Per-source attachment directories (attachments)."""
        return self.base_dir / ATTACHMENTS_DIR_NAME

    def source_attachments_dir(self, source_stem: str) -> Path:
        """Attachments directory for a specific source by slug."""
        return self.attachments_dir / source_stem

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
    def config_file(self) -> Path:
        """Project-level config file (``.knox/config.toml``)."""
        return self.knox_dir / CONFIG_FILE

    @property
    def compile_state_file(self) -> Path:
        """Persistent tier-state checkpoint file."""
        return self.knox_dir / COMPILE_STATE_FILE

    @property
    def wiki_cache_file(self) -> Path:
        """Serialised ``CompiledWiki`` snapshot."""
        return self.knox_dir / WIKI_CACHE_FILE

    @property
    def compiled_wiki_file(self) -> Path:
        """Full compiled wiki JSON (used by pipeline internally)."""
        return self.knox_dir / COMPILED_WIKI_FILE

    @property
    def search_db_file(self) -> Path:
        """SQLite search index database."""
        return self.knox_dir / SEARCH_DB_FILE

    @property
    def memory_db_file(self) -> Path:
        """SQLite memory store database."""
        return self.knox_dir / MEMORY_DB_FILE

    @property
    def memory_log_file(self) -> Path:
        """Append-only memory log."""
        return self.knox_dir / MEMORY_LOG_FILE

    @property
    def memory_store_file(self) -> Path:
        """Memory JSON export."""
        return self.knox_dir / MEMORY_STORE_FILE

    @property
    def verification_file(self) -> Path:
        """Verification results JSON."""
        return self.knox_dir / VERIFICATION_FILE

    @property
    def private_file(self) -> Path:
        """Private page manifest."""
        return self.knox_dir / PRIVATE_FILE

    @property
    def downloaded_file(self) -> Path:
        """URL download manifest (inside cache)."""
        return self.cache_dir / DOWNLOADED_FILE

    @property
    def organize_report_file(self) -> Path:
        """Last ``knox organize`` report."""
        return self.knox_dir / ORGANIZE_REPORT_FILE

    @property
    def references_file(self) -> Path:
        """Bibliographic references store (CSL JSON-aligned)."""
        return self.knox_dir / REFERENCES_FILE

    @property
    def assets_file(self) -> Path:
        """Separated structured-asset store (tables/figures, JSON)."""
        return self.knox_dir / ASSETS_FILE

    @property
    def attachments_manifest_file(self) -> Path:
        """Attachment manifest (JSONL)."""
        return self.knox_dir / ATTACHMENTS_MANIFEST

    @property
    def state_file(self) -> Path:
        """General state file."""
        return self.knox_dir / STATE_FILE

    @property
    def log_file(self) -> Path:
        """Pipeline log file (prepended after each stage)."""
        return self.knox_dir / LOG_FILE

    @property
    def lint_history_file(self) -> Path:
        """Append-only JSONL log of lint runs."""
        return self.knox_dir / LINT_HISTORY_FILE

    @property
    def runs_dir(self) -> Path:
        """Append-only JSONL audit trail of every CLI command run."""
        return self.knox_dir / RUNS_FILE

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


class _DefaultWikiDir:
    """Holds the server-wide default wiki directory (set by MCP server)."""

    def __init__(self) -> None:
        self._path: str | None = None

    def set(self, path: str) -> None:
        self._path = path

    def get(self) -> str | None:
        return self._path


def attachment_slug(source_rel: Path | str) -> str:
    """Derive a deterministic flat directory name from a source's relative path.

    The slug is the relative path (without extension) with:
    - slashes replaced by ``_``
    - lowercased
    - non-alphanumeric (except ``_`` and ``-``) collapsed to ``_``
    - leading/trailing underscores stripped

    Examples::

        >>> attachment_slug("report.pdf")
        'report'
        >>> attachment_slug("papers/thesis.pdf")
        'papers_thesis'
        >>> attachment_slug("_captured/quick-note.md")
        'captured_quick-note'
        >>> attachment_slug("reports/Q1/report.pdf")
        'reports_q1_report'
    """
    p = Path(source_rel)
    # Use stem (no extension), join parts with underscore
    parts = list(p.parent.parts) + ([p.stem] if p.stem else [])
    raw = "_".join(parts).lower()
    # Collapse non-alphanumeric to underscore
    raw = _SLUG_INVALID_RE.sub("_", raw)
    # Collapse multiple underscores
    raw = _SLUG_UNDERSCORE_RE.sub("_", raw)
    return raw.strip("_")


def is_protected_source(path: Path, sources_dir: Path) -> bool:
    """True if *path* lives under a protected tool-managed subdir of *sources_dir*.

    Files in ``sources/_captured/`` or ``sources/_distilled/`` are inputs produced
    by ``capture``/``learn``/``distill``. ``organize`` must never flatten or
    relocate them — they are left in place. Compile still processes them.
    """
    try:
        rel = path.resolve().relative_to(sources_dir.resolve())
    except ValueError:
        return False
    return any(part in PROTECTED_SOURCE_SUBDIRS for part in rel.parts[:-1])


def rel_to_vault(path: Path | str, base_dir: Path) -> str:
    """Convert an absolute path to a vault-relative POSIX string.

    This is the canonical relativization helper for every persistence
    boundary that writes a source path into a ``.knox/`` artifact
    (references ``citing_source``, assets ``source``, generation
    checkpoints, search index, wiki cache). Keeping paths relative to
    the vault ``base_dir`` is what makes a vault relocatable — see
    Antipattern 6 in ``src/knox/AGENTS.md``.

    - Accepts a :class:`~pathlib.Path` or a string.
    - Returns a POSIX-normalized relative path (forward slashes) when
      *path* is under *base_dir*.
    - Returns the path as-is (POSIX-normalized) when it is not under
      *base_dir* — so out-of-vault sources degrade gracefully instead of
      crashing. Callers that must guarantee a vault-relative result should
      check the result with :meth:`Path.is_absolute`.
    - Empty input returns ``""``.
    """
    if not path:
        return ""
    p = Path(path)
    try:
        return p.relative_to(base_dir).as_posix()
    except ValueError:
        return p.as_posix()


def set_default_wiki_dir(path: str) -> None:
    """Set the server-wide default wiki directory (used by MCP server)."""
    _default_wiki_dir().set(path)


def vault_paths(wiki_dir: str | Path | None = None) -> VaultPaths:
    """Resolve VaultPaths from the most specific source.

    Priority: explicit arg > server default (set by MCP) >
    ``KNOX_WIKI_DIR`` env var > current working directory.

    This is the single source of truth for resolving vault paths across
    all code paths (CLI, MCP, API).
    """
    if wiki_dir is None:
        default = _default_wiki_dir().get()
        # KNOX_WIKI_DIR is defined in the config-map (keys.PATHS_WIKI_DIR); read
        # directly here because vault resolution predates config discovery.
        wiki_dir = default if default is not None else os.environ.get(EnvVar.WIKI_DIR, ".")
    return VaultPaths(Path(wiki_dir).resolve())


@cache
def _default_wiki_dir() -> _DefaultWikiDir:
    """Return the singleton default-wiki-dir holder."""
    return _DefaultWikiDir()
