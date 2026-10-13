"""Single source of truth for all kompend path component names and vault paths.

Path *names* are ``Final[str]`` constants (``KOMP_DIR``, ``CACHE_SUBDIR``, …).
Resolved paths for a specific vault live in the :class:`VaultPaths` dataclass,
which takes a ``base_dir`` and exposes every artifact path as a property.

Usage — names (for scaffolding / string-level operations)::

    from kompend.core.paths import KOMP_DIR, CACHE_SUBDIR

    kompend_dir = target / KOMP_DIR

Usage — resolved paths (preferred for all runtime code)::

    from kompend.core.paths import VaultPaths

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

from kompend.core.env import EnvVar

# ── Hidden kompend artifacts folder ────────────────────────────────────────
KOMP_DIR: Final[str] = ".kompend"

# ── Single non-vector store (SQLite) ────────────────────────────────────
KOMP_DB_FILE: Final[str] = "kompend.db"

# ── Capture (learn/capture/distill inputs live under sources/<subdir>) ──
CAPTURED_SUBDIR: Final[str] = "_captured"
LEARNED_SUBDIR: Final[str] = "_learned"
DISTILLED_SUBDIR: Final[str] = "_distilled"
# Archived (consolidated/distilled-away) inputs live under sources/_archive/<kind>/
ARCHIVE_SUBDIR: Final[str] = "_archive"
# Subdirectories under sources/ that `kompend organize` must NEVER flatten or
# relocate. These are tool-managed input zones, not topically-categorized
# material. Compile still processes them (they are legitimate sources);
# organize leaves them in place.
PROTECTED_SOURCE_SUBDIRS: Final[frozenset[str]] = frozenset({CAPTURED_SUBDIR, LEARNED_SUBDIR, DISTILLED_SUBDIR, ARCHIVE_SUBDIR})

# ── Config ──────────────────────────────────────────────────────────────
CONFIG_FILE: Final[str] = "config.toml"
KOMP_CONFIG_FILE: Final[str] = f"{KOMP_DIR}/{CONFIG_FILE}"
PROJECT_CONFIG_FILE: Final[str] = "kompend.toml"
#: Files that mark a directory as a kompend vault root. Used by
# :func:`vault_paths` to fail fast when pointed at a non-vault directory.
VAULT_MARKERS: Final[tuple[str, ...]] = (PROJECT_CONFIG_FILE, KOMP_CONFIG_FILE)
RUMDL_CONFIG_FILE: Final[str] = f"{KOMP_DIR}/rumdl.toml"

# ── Cache ───────────────────────────────────────────────────────────────
CACHE_SUBDIR: Final[str] = "cache"
CACHE_DIR: Final[str] = f"{KOMP_DIR}/{CACHE_SUBDIR}"
IMAGES_SUBDIR: Final[str] = "images"
EXTRACT_SUBDIR: Final[str] = "extract"

# ── Wiki cache (serialised CompiledWiki) ────────────────────────────────
WIKI_CACHE_FILE: Final[str] = "last_wiki.json"
COMPILED_WIKI_FILE: Final[str] = "compiled_wiki.json"
INDEX_FILE: Final[str] = "index.md"

# ── Review queue ────────────────────────────────────────────────────────
REVIEW_DIR: Final[str] = "review"
# ── Per-source summary pages (wiki/summaries/) ─────────────────────────
SUMMARIES_SUBDIR: Final[str] = "summaries"
# ── Full-source exports (exported/) ──────────────────────────────
EXPORTED_DIR_NAME: Final[str] = "exported"
# ── Verification summary page (wiki/quality.md) ─────────────────────────────
QUALITY_FILE: Final[str] = "quality.md"

# ── Vector store ────────────────────────────────────────────────────────
VECTORS_DIR: Final[str] = "vectors"
#: Legacy loose-JSON vector manifest; superseded by the ``vector_manifest`` row.
VECTORS_MANIFEST_FILE: Final[str] = "vectors.manifest.json"

# ── Download tracking ───────────────────────────────────────────────────
DOWNLOADED_FILE: Final[str] = "downloaded.json"

# ── Memory ──────────────────────────────────────────────────────────────
MEMORY_DB_FILE: Final[str] = "memory.db"  # zvec collection dir (rebuildable from the log)
MEMORY_LOG_FILE: Final[str] = "memory.jsonl"  # legacy — superseded by kompend.db
MEMORY_STORE_FILE: Final[str] = "memory.json"  # legacy — superseded by kompend.db

# ── Provenance ──────────────────────────────────────────────────────────
PROVENANCE_DIR: Final[str] = "provenance"  # legacy — superseded by kompend.db

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

# ── Entity-type registry (stable per-concept entity kinds) ───────────────
ENTITY_TYPES_FILE: Final[str] = "entity_types.json"

# ── Assets (separated tables/figures) ────────────────────────────────────
ASSETS_FILE: Final[str] = "assets.json"

# ── Attachments (user-provided supplementary files per source) ───────────
ATTACHMENTS_DIR_NAME: Final[str] = "attachments"
ATTACHMENTS_MANIFEST: Final[str] = "attachments.jsonl"  # legacy — manifest now backed by kompend.db

# ── Dedup (content-level duplicate detection) ────────────────────────────
DEDUP_DIR: Final[str] = "dedup"

# ── Default configurable directory names ────────────────────────────────
DEFAULT_SOURCES_DIR_NAME: Final[str] = "sources"
DEFAULT_OUTPUT_DIR_NAME: Final[str] = "wiki"

#: Canonical page-kind → output-subfolder mapping. Single source of truth for
#: both the vault writers and the lint fixer (which computes the path a
#: resolved wikilink points at).
KIND_FOLDERS: Final[dict[str, str]] = {
    "concept": "concepts",
    "comparison": "concepts",
    "entity": "entities",
    "summary": "concepts",
}

# ── Attachment slug generation ───────────────────────────────────────────
_SLUG_INVALID_RE: Final = __import__("re").compile(r"[^a-z0-9_-]+")
_SLUG_UNDERSCORE_RE: Final = __import__("re").compile(r"_+")


#: Filename extensions that ``safe_stem`` is allowed to strip. Anything
#: outside this set is preserved verbatim — protects ITU-T filenames like
#: ``T-REC-P.10-201906-I!Amd1!PDF-E.pdf`` where ``.stem`` would mistakenly
#: chop off ``T-REC-P`` because Python treats ``.10-201906-…`` as the
#: extension when ``.pdf`` is absent. See p863 lint roundup 2026-09-10.
_SAFE_STEM_EXTENSIONS: Final[frozenset[str]] = frozenset(
    {
        ".pdf",
        ".html",
        ".htm",
        ".md",
        ".markdown",
        ".txt",
        ".csv",
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".xml",
        ".rst",
        ".tex",
        ".ltx",
        ".eml",
        ".zip",
        ".tar",
        ".gz",
        ".7z",
        ".rar",
        ".docx",
        ".pptx",
        ".xlsx",
        ".doc",
        ".ppt",
        ".xls",
        ".epub",
    }
)


@dataclass(frozen=True)
class VaultPaths:
    """All resolved paths for a kompend vault.

    Given a ``base_dir`` (the project root), every artifact path is available
    as a read-only property.  This eliminates the repeated
    ``base_dir / KOMP_DIR / SOMETHING`` pattern scattered across modules.

    Configurable directory names (``sources``, ``wiki``) come from
    :data:`DEFAULTS` and can be overridden via the constructor.

    Directory layout::

        base_dir/
        ├── .kompend/                  ← kompend_dir
        │   ├── cache/              ← cache_dir
        │   │   ├── images/         ← images_dir
        │   │   ├── extract/        ← extract_dir (legacy)
        │   │   └── downloaded.json ← downloaded_file (legacy)
        │   ├── dedup/              ← dedup_dir (legacy)
        │   ├── provenance/         ← provenance_dir (legacy)
        │   ├── review/             ← review_dir
        │   ├── vectors/            ← vectors_dir (zvec directory store)
        │   ├── kompend.db             ← kompend_db_file (single non-vector store)
        │   │                            └ vector_manifest row = zvec id sidecar
        │   ├── config.toml         ← config_file
        │   ├── rumdl.toml          ← RUMDL_CONFIG_FILE
        │   ├── compile_state.json  ← compile_state_file (legacy)
        │   ├── last_wiki.json      ← wiki_cache_file (legacy)
        │   ├── compiled_wiki.json  ← compiled_wiki_file (legacy)
        │   ├── memory.db           ← memory_db_file (zvec dir, NOT SQLite)
        │   ├── memory.jsonl        ┆ memory_log_file (legacy)
        │   ├── memory.json         ← memory_store_file (legacy)
        │   ├── verification.json   ← verification_file (legacy)
        │   ├── private.json        ← private_file (legacy)
        │   ├── last_organize.json  ← organize_report_file (legacy)
        │   ├── lint_history.jsonl  ← lint_history_file (legacy)
        │   ├── runs.jsonl          ← runs_file (legacy)
        │   ├── log.md              ← log_file
        │   ├── references.json     ← references_file (legacy)
        │   ├── entity_types.json   ← ENTITY_TYPES_FILE (legacy)
        │   ├── assets.json         ← assets_file (legacy)
        │   ├── attachments.jsonl   ← attachments_manifest_file (legacy)
        │   └── state.json          ← state_file (unused placeholder)
        ├── sources/                ← sources_dir
        ├── attachments/            ← attachments_dir (per-source)
        ├── exported/               ← full-source markdown exports (on demand)
        ├── wiki/                   ← output_dir
        │   ├── summaries/          ← per-source summary pages
        │   └── index.md            ← index_file

    Entries marked ``(legacy)`` are flat-file artifacts superseded by the single
    ``kompend.db`` store and are no longer written or read. Their path names remain
    only so existing vaults can be detected and cleaned
    (:func:`kompend.core.store.legacy.obsolete_artifacts`).
    """

    base_dir: Path
    sources_dir_name: str = DEFAULT_SOURCES_DIR_NAME
    output_dir_name: str = DEFAULT_OUTPUT_DIR_NAME

    def __init__(
        self,
        base_dir: Path | str | None = None,
        sources_dir_name: str = DEFAULT_SOURCES_DIR_NAME,
        output_dir_name: str = DEFAULT_OUTPUT_DIR_NAME,
    ) -> None:
        """Turn any input (``None``/``str``/``Path``) into a canonical vault root.

        ``None`` or ``""`` means the current working directory. No other code
        should call ``Path.cwd()`` or ``Path(x).resolve()`` on a wiki root.
        """
        resolved = Path.cwd().resolve() if base_dir is None or base_dir == Path("") else Path(base_dir).expanduser().resolve()
        object.__setattr__(self, "base_dir", resolved)
        object.__setattr__(self, "sources_dir_name", sources_dir_name)
        object.__setattr__(self, "output_dir_name", output_dir_name)

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
    def kompend_dir(self) -> Path:
        """Internal ``.kompend`` state directory."""
        return self.base_dir / KOMP_DIR

    @property
    def attachments_dir(self) -> Path:
        """Per-source attachment directories (attachments)."""
        return self.base_dir / ATTACHMENTS_DIR_NAME

    def source_attachments_dir(self, source_stem: str) -> Path:
        """Attachments directory for a specific source by slug."""
        return self.attachments_dir / source_stem

    # ── Sub-directories of .kompend/ ───────────────────────────────────────

    @property
    def cache_dir(self) -> Path:
        """Document parse / compilation cache directory."""
        return self.kompend_dir / CACHE_SUBDIR

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
        return self.kompend_dir / DEDUP_DIR

    @property
    def provenance_dir(self) -> Path:
        """Path handle for provenance — the store resolves it to :attr:`kompend_db_file` (``provenance``); the flat directory is no longer written."""
        return self.kompend_dir / PROVENANCE_DIR

    @property
    def review_dir(self) -> Path:
        """Review queue directory."""
        return self.kompend_dir / REVIEW_DIR

    @property
    def vectors_dir(self) -> Path:
        """Embedding vector store directory."""
        return self.kompend_dir / VECTORS_DIR

    @property
    def vectors_manifest_file(self) -> Path:
        """Loose JSON vector manifest (legacy — the row in :attr:`kompend_db_file`
        replaces it; this path exists only so the file can be reported and
        removed, never read).
        """
        return self.kompend_dir / VECTORS_MANIFEST_FILE

    # ── Files inside .kompend/ ─────────────────────────────────────────────

    @property
    def kompend_db_file(self) -> Path:
        """Single SQLite store for all non-vector ``.kompend/`` state."""
        return self.kompend_dir / KOMP_DB_FILE

    @property
    def config_file(self) -> Path:
        """Project-level config file (``.kompend/config.toml``)."""
        return self.kompend_dir / CONFIG_FILE

    @property
    def compile_state_file(self) -> Path:
        """Path handle for tier state — the store resolves it to :attr:`kompend_db_file` (``tier_state``); the flat file is no longer written."""
        return self.kompend_dir / COMPILE_STATE_FILE

    @property
    def wiki_cache_file(self) -> Path:
        """Path handle for the compiled-wiki snapshot — resolves to :attr:`kompend_db_file` (``wiki_snapshot``); flat file no longer written."""
        return self.kompend_dir / WIKI_CACHE_FILE

    @property
    def compiled_wiki_file(self) -> Path:
        """Path handle for the compiled-wiki snapshot — resolves to :attr:`kompend_db_file` (``wiki_snapshot``); flat file no longer written."""
        return self.kompend_dir / COMPILED_WIKI_FILE

    @property
    def memory_db_file(self) -> Path:
        """Semantic query-memory collection directory (zvec, not SQLite).

        Despite the ``.db`` suffix this is a zvec directory store opened via
        ``memory.store.MemoryStore._ensure_collection``; the rebuildable
        chronological twin is ``memory_log_file``.
        """
        return self.kompend_dir / MEMORY_DB_FILE

    @property
    def memory_log_file(self) -> Path:
        """Path handle for the query-memory log — the store resolves it to :attr:`kompend_db_file` (``memory_log``); the flat file is no longer written."""
        return self.kompend_dir / MEMORY_LOG_FILE

    @property
    def memory_store_file(self) -> Path:
        """Path handle for memory lifecycle state — the store resolves it to :attr:`kompend_db_file` (``memory_states``); the flat file is no longer written."""
        return self.kompend_dir / MEMORY_STORE_FILE

    @property
    def verification_file(self) -> Path:
        """Path handle for the verification report — the store resolves it to :attr:`kompend_db_file` (``verification``); the flat file is no longer written."""
        return self.kompend_dir / VERIFICATION_FILE

    @property
    def private_file(self) -> Path:
        """Path handle for the private-page index — the store resolves it to :attr:`kompend_db_file` (``private_pages``); the flat file is no longer written."""
        return self.kompend_dir / PRIVATE_FILE

    @property
    def downloaded_file(self) -> Path:
        """URL download manifest (inside cache)."""
        return self.cache_dir / DOWNLOADED_FILE

    @property
    def organize_report_file(self) -> Path:
        """Last ``kompend organize`` report."""
        return self.kompend_dir / ORGANIZE_REPORT_FILE

    @property
    def references_file(self) -> Path:
        """Path handle for bibliographic references — the store resolves it to
        :attr:`kompend_db_file` (``bib_references``); the flat file is no longer
        written.
        """
        return self.kompend_dir / REFERENCES_FILE

    @property
    def entity_types_file(self) -> Path:
        """Path handle for the entity-kind registry — the store resolves it to :attr:`kompend_db_file` (``entity_types``); the flat file is no longer written.

        Stable per-concept entity-kind registry.

        Keyed by ``normalize_concept_key(name)``; values are the sanitized
        folder names (``person``, ``organization``, ``standard``, …) used
        by the vault writer for ``entities/<type>/<id>.md`` paths.
        Persisted across recompiles so a concept's folder does not
        flicker between runs when the model gives a slightly different
        type for the same name. See p863 wiki roundup 2026-09-10.
        """
        return self.kompend_dir / ENTITY_TYPES_FILE

    @property
    def assets_file(self) -> Path:
        """Path handle for structured assets — the store resolves it to :attr:`kompend_db_file` (``assets``); the flat file is no longer written."""
        return self.kompend_dir / ASSETS_FILE

    @property
    def attachments_manifest_file(self) -> Path:
        """Path handle for the attachment manifest — resolves to :attr:`kompend_db_file` (``attachment_manifest``); flat file no longer written."""
        return self.kompend_dir / ATTACHMENTS_MANIFEST

    @property
    def state_file(self) -> Path:
        """Unused legacy placeholder — no producer or consumer."""
        return self.kompend_dir / STATE_FILE

    @property
    def log_file(self) -> Path:
        """Rendered markdown pipeline log report (written on demand by ``kompend log``)."""
        return self.kompend_dir / LOG_FILE

    @property
    def lint_history_file(self) -> Path:
        """Path handle for lint-run history — the store resolves it to :attr:`kompend_db_file` (``lint_runs``); the flat file is no longer written."""
        return self.kompend_dir / LINT_HISTORY_FILE

    @property
    def runs_file(self) -> Path:
        """Path handle for the command audit trail — the store resolves it to :attr:`kompend_db_file` (``runs``); the flat file is no longer written."""
        return self.kompend_dir / RUNS_FILE

    # ── Files inside output_dir/ ────────────────────────────────────────

    @property
    def index_file(self) -> Path:
        """Wiki entry point (table of contents)."""
        return self.output_dir / INDEX_FILE

    # ── Files at project root ───────────────────────────────────────────

    @property
    def project_config_file(self) -> Path:
        """User project config file (``kompend.toml`` at root)."""
        return self.base_dir / PROJECT_CONFIG_FILE

    # ── Directory lifecycle ─────────────────────────────────────────────

    def ensure_dirs(self) -> None:
        """Create all standard directories if they do not exist.

        Creates: ``sources_dir``, ``output_dir``, ``kompend_dir``, ``cache_dir``,
        ``images_dir``, ``extract_dir``, ``provenance_dir``, ``review_dir``,
        ``vectors_dir``.
        """
        for directory in (
            self.sources_dir,
            self.output_dir,
            self.kompend_dir,
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


def attachment_slugs_for_source(source_rel: Path | str) -> set[str]:
    """All slug forms that could refer to a source file.

    Attachments may be registered with either the full relative path
    (``attachment_slug("papers/thesis.pdf") == "papers_thesis"``) or the
    bare filename (``attachment_slug("thesis.pdf") == "thesis"``) — the
    ``kompend attachments add`` command passes the user's string verbatim.
    Orphan detection and auto-adoption must match BOTH forms, otherwise
    every attachment for a source under a subdirectory is falsely flagged
    as orphaned (p863 wiki roundup 2026-09-10: 248 files deleted by the
    post-compile lint's ``--fix`` because ``p862_...`` never matched
    ``t-rec-p_862-...``).
    """
    rel = Path(source_rel)
    return {attachment_slug(rel), attachment_slug(rel.name)}


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
    boundary that writes a source path into a ``.kompend/`` artifact
    (references ``citing_source``, assets ``source``, generation
    checkpoints, search index, wiki cache). Keeping paths relative to
    the vault ``base_dir`` is what makes a vault relocatable — see
    Antipattern 6 in ``src/kompend/AGENTS.md``.

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


def safe_stem(path: Path | str) -> str:
    """Return the basename of *path* with a recognized extension stripped.

    Unlike :attr:`pathlib.Path.stem` — which treats the last ``.xxx`` as the
    extension regardless of whether it's a real one — this only strips
    extensions from :data:`_SAFE_STEM_EXTENSIONS`. Multi-dot ITU filenames
    survive intact::

        Path("T-REC-P.10-201906-I!Amd1!PDF-E.pdf").stem
            → "T-REC-P.10-201906-I!Amd1!PDF-E"      (correct for this one)
        safe_stem("T-REC-P.10-201906-I!Amd1!PDF-E.pdf")
            → "T-REC-P.10-201906-I!Amd1!PDF-E"      (correct)
        Path("T-REC-P.10-201906-I!Amd1!PDF-E").stem
            → "T-REC-P"                            (WRONG — chops parent slug)
        safe_stem("T-REC-P.10-201906-I!Amd1!PDF-E")
            → "T-REC-P.10-201906-I!Amd1!PDF-E"      (preserved verbatim)

    Single source of truth for every provenance / label / slug extraction
    (memory #44). Replaces ad-hoc ``Path(...).stem`` calls — those silently
    corrupted ``source_files`` and ``sources[].id`` on the p863 wiki.
    """
    if not path:
        return ""
    name = Path(str(path)).name  # basename only — drops the directory
    if not name:
        return ""
    # Walk from the end looking for a SAFE suffix to strip.
    for ext in _SAFE_STEM_EXTENSIONS:
        if name.lower().endswith(ext) and len(name) > len(ext):
            return name[: -len(ext)]
    return name


def set_default_wiki_dir(path: str) -> None:
    """Set the server-wide default wiki directory (used by MCP server)."""
    _default_wiki_dir().set(path)


def vault_paths(wiki_dir: str | Path | None = None, *, require_vault: bool = True) -> VaultPaths:
    """Resolve VaultPaths from the most specific source.

    Priority: explicit arg > server default (set by MCP) >
    ``KOMP_WIKI_DIR`` env var > current working directory.
    This is the single source of truth for resolving vault paths across
    all code paths (CLI, MCP, API).

    Args:
        wiki_dir: explicit vault root, or ``None`` for env/CWD resolution.
        require_vault: fail fast when the resolved directory is not a kompend
            vault root. Only config-introspection paths (``kompend config get``)
            may opt out — they read defaults, not vault data.

    Raises:
        LookupError: when ``require_vault`` is set and the resolved directory
            has no ``kompend.toml`` / ``.kompend/config.toml``. Silent operation on
            arbitrary directories masks misconfiguration — config, vector
            index, and LLM would all silently degrade.
    """
    if wiki_dir is None:
        default = _default_wiki_dir().get()
        # KOMP_WIKI_DIR is defined in the config-map (keys.PATHS_WIKI_DIR); read
        # directly here because vault resolution predates config discovery.
        wiki_dir = default if default is not None else os.environ.get(EnvVar.WIKI_DIR, ".")
    base_dir = Path(wiki_dir).expanduser().resolve()
    if require_vault and not any((base_dir / marker).exists() for marker in VAULT_MARKERS):
        raise LookupError(
            f"{base_dir} is not a kompend vault root: no {VAULT_MARKERS[0]} or "
            f"{VAULT_MARKERS[1]} found. Run 'kompend init' to create a vault, or "
            "point --wiki-dir at the directory containing .kompend/."
        )
    return VaultPaths(base_dir)


@cache
def _default_wiki_dir() -> _DefaultWikiDir:
    """Return the singleton default-wiki-dir holder."""
    return _DefaultWikiDir()
