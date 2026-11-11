"""Structural lint family: the attachments tree.

Owns the two rules that inspect ``attachments/`` on disk — files nobody
links to (``unreferenced-attachments``) and files whose slug matches no
live source (``orphan-attachments``) — plus the two public helpers that
answer the same questions outside the lint run (``kompend attachments
verify`` shares :func:`orphan_attachment_dirs`, so the slug logic has one
implementation).

Both rules are destructive: their ``fix()`` deletes files and tombstones
the manifest row, and the lint runner gates them behind its
``--fix-destructive`` opt-in.
"""

from __future__ import annotations

import contextlib
import os
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar

from markdown_it import MarkdownIt

from kompend.core.models import CompiledWiki
from kompend.core.models.attachments import AttachmentManifest
from kompend.core.paths import VaultPaths, attachment_slugs_for_source, is_protected_source
from kompend.core.pipeline.state import TierStateStore
from kompend.lint.base import BaseLintRule, LintResult, LintRuleOutcome, Severity
from kompend.logging import get_logger

#: The markdown reader behind :func:`_link_targets`. ``commonmark`` resolves
#: inline destinations (with titles and balanced parentheses), angle-bracket
#: destinations and reference definitions; ``html`` keeps raw HTML as its own
#: tokens so ``<img src>`` / ``<a href>`` stay readable. Fenced and inline code
#: never produce a link token, so a documentation example cannot protect a
#: file the wiki no longer shows.
_MARKDOWN = MarkdownIt("commonmark", {"html": True})

#: ``src``/``href`` inside a raw HTML tag, double-, single- or unquoted.
_HTML_TARGET_RE = re.compile(
    r"""\b(?:src|href)\s*=\s*(?:"(?P<double>[^"]*)"|'(?P<single>[^']*)'|(?P<bare>[^\s"'=<>`]+))""",
    re.IGNORECASE,
)

logger = get_logger(__name__)


class UnreferencedAttachmentsRule(BaseLintRule):
    """Detect published media in a live source directory that no page references.

    The vault writer copies every referenced figure out of ``.kompend/`` into
    ``attachments/<source-slug>/`` (markdown-only invariant). Nothing removes
    those copies when a page is rewritten to reference fewer images or when a
    page that referenced them is gone, so a live source directory keeps copies
    the wiki no longer shows. (A directory whose slug maps to no source at all
    is :class:`OrphanAttachmentsRule`'s territory — this rule skips those, so
    the two never report the same file.)

    A file is prunable only when it is byte-identical to an extracted image in
    ``.kompend/cache/images/``, the copy source. That keeps the rule lossless: it
    deletes only what a later compile can recreate, and never a file a user
    placed there (whose bytes came from somewhere else — hand-added, or written
    by an earlier run whose cache entry is gone). Auto-adoption registers writer
    copies in the manifest too, so the manifest row is consulted only for
    tombstoning what gets deleted, never for the decision.
    """

    name = "unreferenced-attachments"
    description = "Published media in a live source directory that no page references"

    #: Deletes files on disk — same opt-in contract as
    #: :class:`OrphanAttachmentsRule`.
    destructive: ClassVar[bool] = True

    # Populated by check() so fix() does not need sources_dir threaded through.
    _last_sources_dir: str | None = None

    def check(self, wiki: CompiledWiki, *, sources_dir: str | None = None) -> list[LintResult]:
        self._last_sources_dir = sources_dir
        if sources_dir is None:
            return [LintResult(rule=self.name, outcome=LintRuleOutcome.SKIP, message="No sources directory provided")]

        base_dir = Path(sources_dir).parent
        vp = VaultPaths(base_dir)
        if not vp.attachments_dir.is_dir():
            return [LintResult(rule=self.name, outcome=LintRuleOutcome.SKIP, message="No attachments directory")]

        stale = unreferenced_attachment_files(base_dir)
        if not stale:
            return [LintResult(rule=self.name, outcome=LintRuleOutcome.PASS, message="No unreferenced attachments")]
        return [
            LintResult(
                rule=self.name,
                outcome=LintRuleOutcome.FAIL,
                severity=Severity.WARNING,
                message=(
                    f"Unreferenced attachment '{rel}' — no page links it and the manifest does not "
                    "register it (recompile after a page change, or delete it with --fix-destructive)"
                ),
                details={"path": rel},
            )
            for rel in stale
        ]

    def fix(self, wiki: CompiledWiki, results: list[LintResult]) -> list[LintResult]:
        """Delete unreferenced, unregistered attachment files (destructive).

        Mirrors :meth:`OrphanAttachmentsRule.fix`: the lint runner's
        ``--fix-destructive`` opt-in is the only trigger, and a failure leaves
        the result FAIL with the error recorded. Empty slug directories are
        pruned afterwards.
        """
        sources_dir = self._last_sources_dir
        if not results or sources_dir is None:
            return list(results)

        base_dir = Path(sources_dir).parent
        vp = VaultPaths(base_dir)
        manifest = AttachmentManifest(vp.attachments_manifest_file)
        updated: list[LintResult] = []
        deleted = 0
        for r in results:
            if r.outcome != LintRuleOutcome.FAIL:
                updated.append(r)
                continue
            rel_path = (r.details or {}).get("path")
            if not rel_path:
                updated.append(r)
                continue
            abs_path = base_dir / rel_path
            try:
                if abs_path.is_file():
                    abs_path.unlink()
                    deleted += 1
                # Tombstone the manifest row (auto-adoption registered writer
                # copies too) so the manifest does not keep pointing at a file
                # that is gone — the row keeps its history, marked deleted.
                manifest.remove(rel_path)
                r.fix_applied = True
                r.message = f"Deleted unreferenced attachment '{rel_path}'"
            except OSError as exc:
                logger.warning("Could not delete unreferenced attachment %s: %s", abs_path, exc)
                r.message = f"Could not delete unreferenced attachment '{rel_path}': {exc}"
                r.details = {**(r.details or {}), "error": str(exc)}
            updated.append(r)

        if deleted:
            logger.info("Deleted %d unreferenced attachment file(s)", deleted)
            _prune_empty_dirs(vp.attachments_dir)
        return updated


class OrphanAttachmentsRule(BaseLintRule):
    """Detect attachment files whose slug matches no source in the vault.

    Files under ``attachments/<slug>/`` whose slug is derivable from a live
    source are auto-adopted at compile start (manifest entry created, flags
    derived from config). Files whose slug matches NO source cannot be
    recovered by that pass — the parent source is gone or was renamed — so
    they are flagged here for manual intervention. ``--fix`` deletes them
    on disk + tombstones them in the ``attachment_manifest`` table (in ``.kompend/kompend.db``); the caller
    must opt in via the lint ``--fix`` flag.

    A file a page still links to is never an orphan, whatever its slug says.
    A vault carries slug forms from older kompend versions (the slug rule
    changed when sources moved into subdirectories), and the compiled pages
    still link those directories. The parent source is alive — only the
    directory name is history — so deleting the file would break the wiki.
    The rule keeps such files (reported as one PASS summary) and flags only
    the files no page links to.
    """

    name = "orphan-attachments"
    description = "Attachment files with no matching source (unrecoverable by auto-adoption)"

    #: Deletes files on disk — never auto-run by the post-compile lint or
    #: default ``kompend lint check``. Only an explicit ``fix_destructive``
    #: opt-in triggers deletion (p863 wiki roundup 2026-09-10).
    destructive: ClassVar[bool] = True

    # Stored on the instance by check() so fix() can locate the manifest
    # without the lint runner needing to thread ``sources_dir`` through.
    _last_sources_dir: str | None = None

    def check(self, wiki: CompiledWiki, *, sources_dir: str | None = None) -> list[LintResult]:
        self._last_sources_dir = sources_dir
        results: list[LintResult] = []
        if sources_dir is None:
            results.append(LintResult(rule=self.name, outcome=LintRuleOutcome.SKIP, message="No sources directory provided"))
            return results

        base_dir = Path(sources_dir).parent
        vp = VaultPaths(base_dir)
        attachments_root = vp.attachments_dir
        if not attachments_root.is_dir():
            results.append(LintResult(rule=self.name, outcome=LintRuleOutcome.SKIP, message="No attachments directory"))
            return results

        # A compile must have run at least once — before that, orphan state
        # is meaningless (nothing was ever adoptable).
        if not TierStateStore(vp.compile_state_file).load().entries:
            results.append(LintResult(rule=self.name, outcome=LintRuleOutcome.SKIP, message="No compile state — run compile first"))
            return results

        sources_root = Path(sources_dir)
        orphans = orphan_attachment_dirs(base_dir, sources_root)
        # A page link outranks the slug: the file is in use even when its
        # directory name comes from an older slug rule.
        referenced = _referenced_attachment_paths(base_dir)
        linked = [rel for rel in orphans if rel in referenced]
        orphans = [rel for rel in orphans if rel not in referenced]

        kept_note = (
            f" — {len(linked)} attachment file(s) in legacy slug directories are still linked from the wiki "
            "and were kept (migrate the links before deleting the directories)"
            if linked
            else ""
        )

        if orphans:
            for rel in orphans:
                results.append(
                    LintResult(
                        rule=self.name,
                        outcome=LintRuleOutcome.FAIL,
                        severity=Severity.WARNING,
                        message=(
                            f"Orphaned attachment '{rel}' — slug matches no source in sources/ "
                            "(re-add the source, move the file under the correct slug, or delete it)"
                        ),
                        details={"path": rel},
                    )
                )
            if linked:
                results.append(
                    LintResult(
                        rule=self.name,
                        outcome=LintRuleOutcome.PASS,
                        message=(
                            f"{len(linked)} attachment file(s) in legacy slug directories are still linked "
                            "from the wiki and were kept — migrate the links before deleting the directories"
                        ),
                        details={"kept_linked": len(linked)},
                    )
                )
        else:
            results.append(
                LintResult(
                    rule=self.name,
                    outcome=LintRuleOutcome.PASS,
                    message=f"No orphaned attachments{kept_note}",
                    details={"kept_linked": len(linked)} if linked else {},
                )
            )
        return results

    def fix(
        self,
        wiki: CompiledWiki,
        results: list[LintResult],
    ) -> list[LintResult]:
        """Delete orphan attachment files from disk + tombstone in the manifest.

        Destructive. Each FAIL result becomes a removed file + a tombstone
        entry in the ``attachment_manifest`` table (in ``.kompend/kompend.db``). Empty slug directories are
        also removed after their last file is deleted. Failures (permission
        errors, missing manifest) leave the result outcome FAIL but record
        the error in ``details["error"]`` and log at WARNING.

        The lint runner's ``--fix`` flag is the user's opt-in to this
        destructive action. There is no dry-run.

        Reads ``sources_dir`` from the instance attribute populated by the
        most recent :meth:`check` call — the lint runner calls them in
        sequence (``check`` → ``fix``) on the same rule instance.
        """
        sources_dir = self._last_sources_dir
        if not results or sources_dir is None:
            return list(results)

        base_dir = Path(sources_dir).parent
        vp = VaultPaths(base_dir)
        manifest = AttachmentManifest(vp.attachments_manifest_file)

        updated: list[LintResult] = []
        for r in results:
            if r.outcome != LintRuleOutcome.FAIL:
                updated.append(r)
                continue
            rel_path = (r.details or {}).get("path")
            if not rel_path:
                updated.append(r)
                continue
            abs_path = base_dir / rel_path
            try:
                if abs_path.is_file():
                    abs_path.unlink()
                manifest.remove(rel_path)
                r.fix_applied = True
                r.message = f"Deleted orphan attachment '{rel_path}' (was unrecoverable)"
                logger.info("Deleted orphan attachment: %s", rel_path)
            except OSError as exc:
                logger.warning("Could not delete orphan attachment %s: %s", abs_path, exc)
                r.message = f"Could not delete orphan attachment '{rel_path}': {exc}"
                r.details = {**(r.details or {}), "error": str(exc)}
            updated.append(r)

        # Clean up empty slug directories.
        if vp.attachments_dir.is_dir():
            for slug_dir in sorted(vp.attachments_dir.iterdir()):
                if slug_dir.is_dir() and not any(slug_dir.iterdir()):
                    with contextlib.suppress(OSError):
                        slug_dir.rmdir()
        return updated


#: The rules this family contributes to the structural registry.
ATTACHMENT_RULES: tuple[BaseLintRule, ...] = (
    OrphanAttachmentsRule(),
    UnreferencedAttachmentsRule(),
)


def unreferenced_attachment_files(base_dir: Path) -> list[str]:
    """Vault-relative regenerable media in live source dirs that no page links.

    Page links are the source of truth (markdown-first invariant): a file the
    written pages do not reference is not part of the wiki. Only directories
    whose slug matches a live source are considered — dead slugs are
    :class:`OrphanAttachmentsRule`'s job — and only files byte-identical to an
    extracted image (:func:`_is_cached_extraction`), so nothing the wiki cannot
    recreate is ever reported.
    """
    vp = VaultPaths(base_dir)
    if not vp.attachments_dir.is_dir():
        return []

    referenced = _referenced_attachment_paths(base_dir)
    live_slugs = _live_attachment_slugs(base_dir)
    cache_dir = vp.images_dir

    stale: list[str] = []
    for path in sorted(vp.attachments_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(base_dir).as_posix()
        slug = path.relative_to(vp.attachments_dir).as_posix().split("/", 1)[0]
        if slug not in live_slugs:
            continue  # orphan-attachments owns directories with no live source
        if rel in referenced:
            continue
        if not _is_cached_extraction(path, cache_dir):
            continue  # not a regenerable copy — leave it alone
        stale.append(rel)
    return stale


def orphan_attachment_dirs(base_dir: Path, sources_dir: Path) -> list[str]:
    """Attachment files whose directory slug matches no live source.

    Returns **vault-relative** paths (``attachments/<slug>/<file>``). Shared by
    :class:`OrphanAttachmentsRule` and ``kompend attachments verify`` so the slug
    logic has exactly one implementation. Matches both slug forms (full
    relative path and bare filename) — see :func:`attachment_slugs_for_source`.
    """
    attachments_root = VaultPaths(base_dir).attachments_dir
    if not attachments_root.is_dir():
        return []

    sources_root = Path(sources_dir)
    live_slugs: set[str] = set()
    if sources_root.is_dir():
        for p in sources_root.rglob("*"):
            if p.is_file() and not is_protected_source(p, sources_root):
                live_slugs.update(attachment_slugs_for_source(p.relative_to(sources_root).as_posix()))

    orphans: list[str] = []
    for slug_dir in sorted(attachments_root.iterdir()):
        if not slug_dir.is_dir() or slug_dir.name in live_slugs:
            continue
        for f in sorted(slug_dir.rglob("*")):
            if f.is_file():
                orphans.append(f.relative_to(base_dir).as_posix())
    return orphans


def _live_attachment_slugs(base_dir: Path) -> set[str]:
    """Slug forms of every live source (both full-path and bare-name forms)."""
    vp = VaultPaths(base_dir)
    if not vp.sources_dir.is_dir():
        return set()
    slugs: set[str] = set()
    for source in vp.sources_dir.rglob("*"):
        if source.is_file():
            slugs |= attachment_slugs_for_source(source.relative_to(vp.sources_dir))
    return slugs


def _is_cached_extraction(path: Path, cache_dir: Path) -> bool:
    """Is *path* a byte-identical copy of an image the parser extracted?

    Compares content, not just the name: same-named files diverge when a source
    is re-parsed, and a re-parse invalidates the copy anyway. Cheap enough —
    the caller only asks about files no page references.
    """
    original = cache_dir / path.name
    try:
        if not original.is_file() or original.stat().st_size != path.stat().st_size:
            return False
        return original.read_bytes() == path.read_bytes()
    except OSError:
        return False


def _referenced_attachment_paths(base_dir: Path) -> set[str]:
    """Every vault-relative ``attachments/...`` path linked from a markdown file.

    Reads the written pages, not the in-memory graph: the writer rewrites image
    paths to relative links at write time, so only the files on disk show what
    the wiki actually references. Links resolve page-relative first (what the
    wiki writer emits) and vault-relative as a fallback (what an export uses).
    """
    referenced: set[str] = set()
    for md in base_dir.rglob("*.md"):
        parts = md.relative_to(base_dir).parts[:-1]
        if any(part in {".kompend", "sources", "attachments"} for part in parts):
            continue
        try:
            text = md.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for raw in _link_targets(text):
            if "attachments/" not in raw:
                continue
            link = raw.strip().strip("<>").split("#", 1)[0]
            for candidate in (md.parent / link, base_dir / link):
                rel = _vault_relative(base_dir, candidate)
                if rel is not None:
                    referenced.add(rel)
    return referenced


def _link_targets(text: str) -> list[str]:
    """The destination of every link, image and raw-HTML reference in *text*.

    Read with a markdown parser, not a pattern: the forms a hand-rolled reader
    got wrong (a link title, an angle-bracket destination, a reference
    definition, ``<img src>``) each used to resolve to nothing, so the figure
    behind them looked unreferenced and the destructive rules deleted a page's
    own media. Code spans and fences yield no link tokens here, so a
    documentation example cannot protect a file either.

    A link target may hold balanced parentheses — a figure named
    ``Perceptual_Objective_Listening_Quality_Assessment_(POLQA),_…jpg`` has
    them — and the parser resolves that case too.
    """
    targets: list[str] = []
    env: dict[str, Any] = {}
    for token in _MARKDOWN.parse(text, env):
        if token.type == "inline":
            for child in token.children or ():
                if child.type == "image" and child.attrGet("src"):
                    targets.append(str(child.attrGet("src")))
                elif child.type == "link_open" and child.attrGet("href"):
                    targets.append(str(child.attrGet("href")))
                elif child.type == "html_inline":
                    targets.extend(_html_targets(child.content))
        elif token.type == "html_block":
            targets.extend(_html_targets(token.content))
    # A reference definition authors the reference even before a link uses it:
    # the definition is page content, and the destructive rules must not delete
    # what the author declared.
    for definition in (env.get("references") or {}).values():
        href = definition.get("href") if isinstance(definition, dict) else None
        if href:
            targets.append(str(href))
    return targets


def _html_targets(fragment: str) -> Iterator[str]:
    """The ``src``/``href`` values in one raw HTML fragment."""
    for match in _HTML_TARGET_RE.finditer(fragment):
        value = match.group("double") or match.group("single") or match.group("bare")
        if value:
            yield value


def _prune_empty_dirs(attachments_root: Path) -> None:
    """Remove directories left empty under ``attachments/`` (deepest first)."""
    for d in sorted((p for p in attachments_root.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
        with contextlib.suppress(OSError):
            d.rmdir()


def _vault_relative(base_dir: Path, candidate: Path) -> str | None:
    """``candidate`` as a vault-relative posix path, or ``None`` when outside."""
    try:
        return Path(os.path.normpath(candidate)).relative_to(base_dir).as_posix()
    except ValueError:
        return None
