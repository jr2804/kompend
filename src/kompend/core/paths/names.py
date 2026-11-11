"""Canonical names for every on-disk artifact a kompend vault holds.

Path *names* are ``Final[str]`` constants (``KOMP_DIR``, ``CACHE_SUBDIR``, …).
Nothing here resolves a path against a vault root; :class:`VaultPaths` does
that, and :mod:`kompend.core.paths.resolve` holds the derivation helpers.

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

from dataclasses import dataclass
from pathlib import Path
from typing import Final

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
#: :func:`kompend.core.paths.vault_paths` to fail fast when pointed at a
#: non-vault directory.
VAULT_MARKERS: Final[tuple[str, ...]] = (PROJECT_CONFIG_FILE, KOMP_CONFIG_FILE)
#: The version-control rail file at the vault root. Its absence is what makes a
#: directory look unversioned, so the gitignore scaffolding and the
#: ``kompend gitignore`` command read the name from here, never inline.
GITIGNORE_FILE: Final[str] = ".gitignore"
RUMDL_CONFIG_FILE: Final[str] = f"{KOMP_DIR}/rumdl.toml"
#: parsecraft's own configuration file inside a vault. parsecraft owns its
#: routing vocabulary, so the file sits in the same tool directory as kompend's
#: ``config.toml`` and is read by parsecraft's engine, not by kompend.
PARSECRAFT_CONFIG_FILE: Final[str] = f"{KOMP_DIR}/parsecraft.toml"
#: parsecraft's project-layer file name. Its engine reads the one in the
#: current directory above the vault file and below the environment.
PARSECRAFT_PROJECT_FILE: Final[str] = "parsecraft.toml"

# ── Failure repro bundles ───────────────────────────────────────────────
TMP_SUBDIR: Final[str] = "tmp"
FAILURES_SUBDIR: Final[str] = "failures"
#: One bundle per failed/stalled parse: ``.kompend/tmp/failures/<hash>/``.
FAILURE_BUNDLES_DIR: Final[str] = f"{KOMP_DIR}/{TMP_SUBDIR}/{FAILURES_SUBDIR}"

# ── Cache ───────────────────────────────────────────────────────────────
CACHE_SUBDIR: Final[str] = "cache"
CACHE_DIR: Final[str] = f"{KOMP_DIR}/{CACHE_SUBDIR}"
IMAGES_SUBDIR: Final[str] = "images"
EXTRACT_SUBDIR: Final[str] = "extract"

# ── Wiki cache (serialised CompiledWiki) ────────────────────────────────
WIKI_CACHE_FILE: Final[str] = "last_wiki.json"
COMPILED_WIKI_FILE: Final[str] = "compiled_wiki.json"
INDEX_FILE: Final[str] = "index.md"
#: Corpus bibliography page (wiki/references.md) — distinct from the
#: machine-readable ``REFERENCES_FILE`` cache.
REFERENCES_MD: Final[str] = "references.md"
#: Thematic corpus overview page (wiki/overview.md).
OVERVIEW_MD: Final[str] = "overview.md"

# ── Review queue ────────────────────────────────────────────────────────
REVIEW_DIR: Final[str] = "review"
# ── Per-source summary pages (wiki/summaries/) ─────────────────────────
SUMMARIES_SUBDIR: Final[str] = "summaries"
# ── Full-source exports (exported/) ──────────────────────────────
EXPORTED_DIR_NAME: Final[str] = "exported"
# ── Verification summary page (wiki/quality.md) ─────────────────────────────
QUALITY_FILE: Final[str] = "quality.md"
#: Single-file markdown export of the whole wiki (output_dir/wiki.md).
WIKI_MARKDOWN_FILE: Final[str] = "wiki.md"

# ── Vector store ────────────────────────────────────────────────────────
#: The vector branch: one SQLite file holding the vectors, the lexical index
#: and the filterable metadata. Separate from the vault store and
#: regenerable — the manifest of the last build lives in ``kompend.db``.
VECTORS_DB_FILE: Final[str] = "vectors.db"
#: Loose-JSON vector manifest the ``vector_manifest`` row replaced; kept so
#: :func:`kompend.core.store.legacy.obsolete_artifacts` can find the file.
VECTORS_MANIFEST_FILE: Final[str] = "vectors.manifest.json"

# ── Download tracking ───────────────────────────────────────────────────
DOWNLOADED_FILE: Final[str] = "downloaded.json"

# ── Memory ──────────────────────────────────────────────────────────────
MEMORY_DB_FILE: Final[str] = "memory.db"  # sqlite-vec index over the log (rebuildable)
MEMORY_LOG_FILE: Final[str] = "memory.jsonl"  # obsolete artifact, see VaultPaths docstring
MEMORY_STORE_FILE: Final[str] = "memory.json"  # obsolete artifact, see VaultPaths docstring
#: Subdirectory under ``sources/`` that holds the synthesized memory context.
MEMORY_SUBDIR: Final[str] = "_memory"
#: Synthesized memory context page (sources/_memory/system_context.md).
MEMORY_CONTEXT_FILE: Final[str] = "system_context.md"
#: Sub-directory of ``.kompend/`` for the agent notebook (``compile.llm.memory_store = "vault"``):
#: one directory per agent, each holding the agent-written ``MEMORY.md`` index plus
#: the topic files behind the agents' memory tools.
AGENT_MEMORY_SUBDIR: Final[str] = "agent-memory"

# ── Step persistence ────────────────────────────────────────────────────
#: SQLite file backing the harness ``StepPersistence`` capability (generation
#: run transcripts + snapshots). Deliberately NOT a table in
#: :attr:`KOMP_DB_FILE`: that store is recreated clean-cut on schema version
#: bumps, which would wipe foreign tables.
STEP_DB_FILE: Final[str] = "steps.db"

# ── Provenance ──────────────────────────────────────────────────────────
PROVENANCE_DIR: Final[str] = "provenance"  # obsolete artifact, see VaultPaths docstring

# ── Verification ────────────────────────────────────────────────────────
VERIFICATION_FILE: Final[str] = "verification.json"

# ── Logging ─────────────────────────────────────────────────────────────
LOG_FILE: Final[str] = "log.md"

# Flight-recorder run logs (see kompend.logging.RunLogHandler): one file per
# CLI run, named by the run id minted in core/reporting/runlog.py.
LOGS_SUBDIR: Final[str] = "logs"
RUN_LOG_FILE_TEMPLATE: Final[str] = "run-{run_id}.log"


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
ATTACHMENTS_MANIFEST: Final[str] = "attachments.jsonl"  # obsolete artifact, see VaultPaths docstring

# ── Dedup (content-level duplicate detection) ────────────────────────────
DEDUP_DIR: Final[str] = "dedup"

# ── Default configurable directory names ────────────────────────────────
DEFAULT_SOURCES_DIR_NAME: Final[str] = "sources"
DEFAULT_OUTPUT_DIR_NAME: Final[str] = "wiki"

#: Canonical kind→folder mapping lives next to its key vocabulary:
#: :data:`kompend.core.models.kinds.KIND_FOLDERS` (``core.models`` imports
#: ``core.paths``, so it cannot be defined here).


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
        │   │   └── images/         ← images_dir
        │   ├── review/             ← review_dir
        │   ├── vectors.db          ← vectors_db_file (sqlite-vec vector branch)
        │   ├── kompend.db          ← kompend_db_file (single non-vector store)
        │   │                            └ vector_manifest row = index manifest
        │   ├── memory.db           ← memory_db_file (sqlite-vec index over the log)
        │   ├── config.toml         ← config_file
        │   ├── rumdl.toml          ← RUMDL_CONFIG_FILE
        │   └── log.md              ← log_file
        ├── sources/                ← sources_dir
        ├── attachments/            ← attachments_dir (per-source)
        ├── exported/               ← full-source markdown exports (on demand)
        └── wiki/                   ← output_dir
            ├── summaries/          ← per-source summary pages
            └── index.md            ← index_file

    Properties whose docstring names a ``kompend.db`` row (``wiki_snapshot``,
    ``tier_state``, ``provenance``, …) resolve a path that store replaced, so an
    existing vault can be detected and cleaned
    (:func:`kompend.core.store.legacy.obsolete_artifacts`); nothing writes them.
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
        """Provenance — ``provenance`` row in :attr:`kompend_db_file`."""
        return self.kompend_dir / PROVENANCE_DIR

    @property
    def agent_memory_dir(self) -> Path:
        """Agent notebook directory (``compile.llm.memory_store = "vault"``)."""
        return self.kompend_dir / AGENT_MEMORY_SUBDIR

    @property
    def logs_dir(self) -> Path:
        """Flight-recorder run logs directory (created on demand by the handler)."""
        return self.kompend_dir / LOGS_SUBDIR

    @property
    def tmp_dir(self) -> Path:
        """Scratch directory inside the vault (repro bundles and one-off dumps)."""
        return self.kompend_dir / TMP_SUBDIR

    @property
    def failure_bundles_dir(self) -> Path:
        """Repro bundles for failed or stalled parses, one directory per source hash."""
        return self.tmp_dir / FAILURES_SUBDIR

    def run_log_file(self, run_id: str) -> Path:
        """Flight-recorder log for one CLI run (``run-<run_id>.log``)."""
        return self.logs_dir / run_log_file_name(run_id)

    @property
    def step_db_file(self) -> Path:
        """SQLite file for harness step persistence (generation transcripts)."""
        return self.kompend_dir / STEP_DB_FILE

    @property
    def review_dir(self) -> Path:
        """Review queue directory."""
        return self.kompend_dir / REVIEW_DIR

    @property
    def vectors_db_file(self) -> Path:
        """Embedding vector store file (the regenerable vector branch)."""
        return self.kompend_dir / VECTORS_DB_FILE

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
    def parsecraft_config_file(self) -> Path:
        """Parsecraft's routing config inside the vault (``.kompend/parsecraft.toml``)."""
        return self.base_dir / PARSECRAFT_CONFIG_FILE

    @property
    def compile_state_file(self) -> Path:
        """Tier state — ``tier_state`` row in :attr:`kompend_db_file`."""
        return self.kompend_dir / COMPILE_STATE_FILE

    @property
    def wiki_cache_file(self) -> Path:
        """The compiled-wiki snapshot — ``wiki_snapshot`` row in :attr:`kompend_db_file`."""
        return self.kompend_dir / WIKI_CACHE_FILE

    @property
    def compiled_wiki_file(self) -> Path:
        """The compiled-wiki snapshot — ``wiki_snapshot`` row in :attr:`kompend_db_file`."""
        return self.kompend_dir / COMPILED_WIKI_FILE

    @property
    def memory_db_file(self) -> Path:
        """Semantic query-memory index file (sqlite-vec over ``memory_log``).

        Despite the ``.db`` suffix this is a SQLite file opened via
        ``memory.store.MemoryStore._ensure_store``; the rebuildable
        chronological twin is ``memory_log_file``.
        """
        return self.kompend_dir / MEMORY_DB_FILE

    @property
    def memory_log_file(self) -> Path:
        """The query-memory log — ``memory_log`` row in :attr:`kompend_db_file`."""
        return self.kompend_dir / MEMORY_LOG_FILE

    @property
    def memory_store_file(self) -> Path:
        """Memory lifecycle state — ``memory_states`` row in :attr:`kompend_db_file`."""
        return self.kompend_dir / MEMORY_STORE_FILE

    @property
    def verification_file(self) -> Path:
        """The verification report — ``verification`` row in :attr:`kompend_db_file`."""
        return self.kompend_dir / VERIFICATION_FILE

    @property
    def private_file(self) -> Path:
        """The private-page index — ``private_pages`` row in :attr:`kompend_db_file`."""
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
        """Bibliographic references — ``bib_references`` row in
        :attr:`kompend_db_file`.
        """
        return self.kompend_dir / REFERENCES_FILE

    @property
    def entity_types_file(self) -> Path:
        """Entity-kind registry — ``entity_types`` row in :attr:`kompend_db_file`.

        Stable per-concept entity-kind registry.

        Keyed by ``normalize_concept_key(name)``; values are the sanitized
        folder names (``person``, ``organization``, ``standard``, …) used
        by the vault writer for ``entities/<type>/<id>.md`` paths.
        Persisted across recompiles so a concept's folder does not
        flicker between runs when the model gives a slightly different
        type for the same name.
        """
        return self.kompend_dir / ENTITY_TYPES_FILE

    @property
    def assets_file(self) -> Path:
        """Structured assets — ``assets`` row in :attr:`kompend_db_file`."""
        return self.kompend_dir / ASSETS_FILE

    @property
    def attachments_manifest_file(self) -> Path:
        """The attachment manifest — ``attachment_manifest`` row in :attr:`kompend_db_file`."""
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
        """Lint-run history — ``lint_runs`` row in :attr:`kompend_db_file`."""
        return self.kompend_dir / LINT_HISTORY_FILE

    @property
    def runs_file(self) -> Path:
        """The command audit trail — ``runs`` row in :attr:`kompend_db_file`."""
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
        ``images_dir``, ``extract_dir``, ``provenance_dir``, ``review_dir``.

        The vector branch is a *file* (``vectors.db``); the store creates it
        on demand, so no directory is pre-created for it.
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
        ):
            directory.mkdir(parents=True, exist_ok=True)


def run_log_file_name(run_id: str) -> str:
    """File name of the flight-recorder log for *run_id*.

    The template is data (``.agents/FILES.md`` documents it), so one function
    owns the substitution; callers never format it themselves. ``str.format``
    is kept over an f-string because the template stays inspectable.
    """
    return RUN_LOG_FILE_TEMPLATE.format(run_id=run_id)
