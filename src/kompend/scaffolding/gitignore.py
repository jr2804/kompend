"""The vault ``.gitignore``: which artifacts belong in version control.

A kompend vault mixes three kinds of files, and only one of them has to be
committed for the wiki to work:

* **Inputs and settings** — ``sources/``, ``kompend.toml``,
  ``.kompend/config.toml``, ``.kompend/rumdl.toml``, ``.kompend/parsecraft.toml``
  and the root ``*.md`` files. Nothing regenerates these.
* **The vault store** — ``.kompend/kompend.db``. Its cache tables rebuild, but
  it also holds rows that exist nowhere else (the memory log, review
  decisions, bibliographic references, the entity-kind registry). Committing it
  is safe only while no kompend process runs, because SQLite runs in WAL mode.
* **Derived state** — the vector and memory indexes, generation transcripts,
  caches, run logs, failure bundles, dedup fingerprints and the legacy flat
  JSON files whose content moved into the store. ``kompend`` rebuilds all of
  them from the sources and the store, so they never belong in a repository.

:func:`render_gitignore` renders the managed block from the path constants, so
the ignore list cannot drift from the artifact names. :func:`merge_gitignore`
replaces that block in an existing file and leaves every other line — the
user's own rules — untouched.
"""

from __future__ import annotations

import re
from importlib import resources
from pathlib import Path
from typing import Any, Final

import jinja2

from kompend.core.paths import (
    AGENT_MEMORY_SUBDIR,
    ASSETS_FILE,
    ATTACHMENTS_DIR_NAME,
    ATTACHMENTS_MANIFEST,
    CACHE_SUBDIR,
    COMPILE_STATE_FILE,
    CONFIG_FILE,
    DEDUP_DIR,
    ENTITY_TYPES_FILE,
    EXPORTED_DIR_NAME,
    KOMP_DB_FILE,
    KOMP_DIR,
    LINT_HISTORY_FILE,
    LOG_FILE,
    LOGS_SUBDIR,
    MEMORY_DB_FILE,
    MEMORY_LOG_FILE,
    MEMORY_STORE_FILE,
    ORGANIZE_REPORT_FILE,
    PRIVATE_FILE,
    PROVENANCE_DIR,
    REFERENCES_FILE,
    REVIEW_DIR,
    RUNS_FILE,
    STATE_FILE,
    STEP_DB_FILE,
    TMP_SUBDIR,
    VECTORS_DB_FILE,
    VECTORS_MANIFEST_FILE,
    VERIFICATION_FILE,
)

TEMPLATES_PACKAGE: Final[str] = "kompend.scaffolding.templates"

#: Opening line of the block kompend owns inside ``.gitignore``.
BLOCK_BEGIN: Final[str] = "# >>> kompend gitignore (managed — regenerate with `kompend gitignore`) >>>"
#: Closing line of that block.
BLOCK_END: Final[str] = "# <<< kompend gitignore (managed) <<<"

_BLOCK_RE: Final[re.Pattern[str]] = re.compile(
    rf"{re.escape(BLOCK_BEGIN)}.*?{re.escape(BLOCK_END)}\n?",
    re.DOTALL,
)

TEMPLATE_GITIGNORE: Final[str] = "gitignore.j2"


def render_gitignore(*, sources_dir: str, output_dir: str) -> str:
    """Render the full ``.gitignore`` — header prose, managed block, optional block.

    *sources_dir* and *output_dir* are the vault's configured source and
    compiled-wiki directories. The compiled wiki is ignored by default because
    ``kompend compile`` rewrites it wholesale; the sources are never ignored.
    """
    return _render_template(
        sources_dir=sources_dir,
        output_dir=output_dir,
        kompend_dir=KOMP_DIR,
        kompend_db=KOMP_DB_FILE,
        config_file=CONFIG_FILE,
        cache_subdir=CACHE_SUBDIR,
        logs_subdir=LOGS_SUBDIR,
        tmp_subdir=TMP_SUBDIR,
        vectors_db=VECTORS_DB_FILE,
        memory_db=MEMORY_DB_FILE,
        steps_db=STEP_DB_FILE,
        provenance_dir=PROVENANCE_DIR,
        dedup_dir=DEDUP_DIR,
        compile_state_file=COMPILE_STATE_FILE,
        state_file=STATE_FILE,
        organize_report_file=ORGANIZE_REPORT_FILE,
        vectors_manifest_file=VECTORS_MANIFEST_FILE,
        memory_log_file=MEMORY_LOG_FILE,
        memory_store_file=MEMORY_STORE_FILE,
        attachments_manifest=ATTACHMENTS_MANIFEST,
        verification_file=VERIFICATION_FILE,
        references_file=REFERENCES_FILE,
        entity_types_file=ENTITY_TYPES_FILE,
        assets_file=ASSETS_FILE,
        private_file=PRIVATE_FILE,
        lint_history_file=LINT_HISTORY_FILE,
        runs_file=RUNS_FILE,
        attachments_dir=ATTACHMENTS_DIR_NAME,
        exported_dir=EXPORTED_DIR_NAME,
        review_dir=REVIEW_DIR,
        agent_memory_dir=AGENT_MEMORY_SUBDIR,
        log_file=LOG_FILE,
    )


def _render_template(**variables: Any) -> str:
    """Render the shipped ``gitignore.j2`` with *variables*.

    Lives here rather than in :mod:`kompend.scaffolding._core` so the scaffold can
    import this module without a cycle: the template and the names it needs are
    one contract, and the scaffold only consumes the rendered text.
    """
    source = Path(str(resources.files(TEMPLATES_PACKAGE).joinpath(TEMPLATE_GITIGNORE)))
    content = source.read_text(encoding="utf-8")
    return jinja2.Template(content, keep_trailing_newline=True).render(**variables)


def dir_name(value: object, *, fallback: str) -> str:
    """The vault-relative name of a configured directory path.

    ``paths.sources_dir`` and ``paths.output_dir`` resolve to absolute paths,
    but the template spells rules relative to the vault root, so only the last
    segment matters. A missing value falls back to the documented default.
    """
    if not isinstance(value, str) or not value:
        return fallback
    return Path(value).name or fallback


def merge_gitignore(existing: str, rendered: str) -> str:
    """Replace the managed block in *existing* with the one from *rendered*.

    Every line outside the block survives verbatim, including the user's own
    rules above and below it. A file without a block gets one appended, with a
    blank line separating it from the user's last rule.
    """
    block = managed_block(rendered)
    if block is None:
        msg = "the rendered gitignore has no managed block"
        raise ValueError(msg)

    if managed_block(existing) is None:
        body = existing.strip()
        if not body:
            return rendered
        return f"{body}\n\n{block}"

    merged = _BLOCK_RE.sub(lambda _match: block, existing, count=1)
    return merged if merged.endswith("\n") else f"{merged}\n"


def is_current(existing: str, rendered: str) -> bool:
    """True when *existing* carries exactly the managed block *rendered* proposes."""
    return managed_block(existing) == managed_block(rendered)


def managed_block(text: str) -> str | None:
    """Return the managed block inside *text*, or ``None`` when it is absent."""
    match = _BLOCK_RE.search(text)
    return match.group(0) if match is not None else None
