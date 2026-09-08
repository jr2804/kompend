"""Consolidate redundant captured/learned/distilled notes.

Finds near-duplicate notes across ``sources/_captured/``, ``sources/_learned/``
and ``sources/_distilled/`` using the shared text-matching stack
(:mod:`knox.core.text.matching`): rapidfuzz token similarity (T2) plus an
optional embedding tier (T3) via ``SemanticMatcher``. Duplicate groups keep
the newest (or longest) file; the rest are archived to
``sources/_archive/<kind>/`` by default.

Only tool-managed input zones are touched — real sources are never moved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from knox.capture.distill import CapturedNote, _load_captured_notes
from knox.core.paths import ARCHIVE_SUBDIR, CAPTURED_SUBDIR, DISTILLED_SUBDIR, LEARNED_SUBDIR
from knox.core.text.matching import SemanticMatcher, similarity
from knox.logging import get_logger

logger = get_logger(__name__)

#: rapidfuzz-scale (0–100) threshold for "these two notes are duplicates".
_DUP_SIMILARITY_THRESHOLD = 85.0
#: cosine threshold for the embedding tier (0–1).
_DUP_EMBED_THRESHOLD = 0.75


@dataclass
class DuplicateGroup:
    """A set of notes judged to be near-duplicates of each other."""

    keeper: CapturedNote
    duplicates: list[CapturedNote] = field(default_factory=list)

    @property
    def all_notes(self) -> list[CapturedNote]:
        return [self.keeper, *self.duplicates]


def _zone_kind(note: CapturedNote) -> str:
    """Map a note's source zone to its archive subdir name."""
    if note.source_file is None:
        return "unknown"
    parts = note.source_file.parts
    if CAPTURED_SUBDIR in parts:
        return "capture"
    if LEARNED_SUBDIR in parts:
        return "learn"
    if DISTILLED_SUBDIR in parts:
        return "distilled"
    return "unknown"


def _note_text(note: CapturedNote) -> str:
    """Comparable text for a note: title + body (frontmatter excluded)."""
    return f"{note.title}\n{note.content}"


def _pick_keeper(group: list[CapturedNote]) -> CapturedNote:
    """Keep the longest note; ties go to the newest (captured_at)."""
    return max(group, key=lambda n: (len(n.content), n.captured_at or ""))


def find_duplicates(
    notes: list[CapturedNote],
    *,
    matcher: SemanticMatcher | None = None,
    similarity_threshold: float = _DUP_SIMILARITY_THRESHOLD,
    embed_threshold: float = _DUP_EMBED_THRESHOLD,
) -> list[DuplicateGroup]:
    """Group *notes* into near-duplicate clusters.

    T2 (rapidfuzz) runs first; notes scoring at or above
    *similarity_threshold* are grouped. Notes that remain unmatched are
    checked against the embedding tier when *matcher* is available and
    scores at or above *embed_threshold*.

    Returns one :class:`DuplicateGroup` per cluster (singletons are
    excluded — only groups with 2+ members are returned).
    """
    groups: list[DuplicateGroup] = []
    used: set[int] = set()

    texts = [_note_text(n) for n in notes]
    for i, note in enumerate(notes):
        if i in used:
            continue
        cluster = [note]
        for j in range(i + 1, len(notes)):
            if j in used:
                continue
            score = similarity(texts[i], texts[j])
            if score >= similarity_threshold:
                cluster.append(notes[j])
                used.add(j)
        if len(cluster) > 1:
            groups.append(DuplicateGroup(keeper=_pick_keeper(cluster), duplicates=[n for n in cluster if n is not _pick_keeper(cluster)]))
            used.add(i)

    # Embedding tier: catch near-duplicates the string tier missed.
    if matcher is not None and matcher.available:
        unmatched = [n for i, n in enumerate(notes) if i not in used]
        if len(unmatched) > 1:
            matcher.candidates = {str(i): _note_text(n) for i, n in enumerate(unmatched)}
            matcher.__post_init__()
            for i, note in enumerate(unmatched):
                if i in {x for g in groups for x in (g.keeper, *g.duplicates)}:
                    continue
                hit = matcher.match(_note_text(note))
                if hit is None:
                    continue
                j = int(hit[0])
                if j <= i or j >= len(unmatched):
                    continue
                other = unmatched[j]
                if hit[1] >= embed_threshold:
                    group = next((g for g in groups if other in g.all_notes), None)
                    if group is not None:
                        group.duplicates.append(note)
                    else:
                        groups.append(DuplicateGroup(keeper=_pick_keeper([note, other]), duplicates=[note if other is _pick_keeper([note, other]) else other]))
    return groups


def consolidate(
    sources_dir: Path,
    *,
    matcher: SemanticMatcher | None = None,
    dry_run: bool = False,
    archive: bool = True,
) -> list[DuplicateGroup]:
    """Find and archive redundant notes under *sources_dir*.

    Scans ``_captured/``, ``_learned/`` and ``_distilled/``, groups
    near-duplicates, and archives every non-keeper to
    ``sources/_archive/<kind>/``. With *dry_run* nothing is moved — the
    groups are only reported. With *archive=False* duplicates are left in
    place (report-only).

    Returns the duplicate groups found.
    """
    captured_dir = sources_dir / CAPTURED_SUBDIR
    learned_dir = sources_dir / LEARNED_SUBDIR
    distilled_dir = sources_dir / DISTILLED_SUBDIR
    notes: list[CapturedNote] = []
    for d in (captured_dir, learned_dir, distilled_dir):
        if d.is_dir():
            notes.extend(_load_captured_notes(d))

    groups = find_duplicates(notes, matcher=matcher)
    if not groups:
        logger.info("consolidate: no duplicate notes found")
        return groups

    archive_root = sources_dir / ARCHIVE_SUBDIR
    for group in groups:
        for dup in group.duplicates:
            if dup.source_file is None or not dup.source_file.exists():
                continue
            if dry_run or not archive:
                keeper = group.keeper.source_file.name if group.keeper.source_file else "?"
                logger.info("consolidate: would archive %s (dup of %s)", dup.source_file.name, keeper)
                continue
            kind = _zone_kind(dup)
            dest_dir = archive_root / kind
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / dup.source_file.name
            if dest.exists():
                dest.unlink()
            dup.source_file.rename(dest)
            logger.info("consolidate: archived %s -> %s", dup.source_file.name, dest)
    return groups
