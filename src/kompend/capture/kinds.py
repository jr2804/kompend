"""Vocabularies for the capture subsystem.

Two different vocabularies met in one field before this module existed: the
``type`` of a source note is either the *extraction kind* the capture agent
assigned (``decision``, ``discovery``, …) or the *zone* the note lives in
(``learn``, ``distilled``). Archiving added a third reading of the same
strings. Both now have one home each.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Final

from kompend.core.paths import CAPTURED_SUBDIR, DISTILLED_SUBDIR, LEARNED_SUBDIR

__all__ = [
    "CAPTURE_SOURCE_ID",
    "UNGROUPED_TAG",
    "CaptureKind",
    "SourceZone",
]

#: First tag a note without tags is grouped under when distilling.
#: Persisted as the distilled page's tag, so the spelling is a contract.
UNGROUPED_TAG: Final[str] = "general"

#: ``source_id`` the capture agent runs under (agent/cache identity, not a
#: frontmatter key).
CAPTURE_SOURCE_ID: Final[str] = "capture"


class CaptureKind(StrEnum):
    """Extraction kinds the capture agent may assign to an item.

    The prompt template (``capture/capture_system.md.j2``) offers exactly
    these kinds; ``tests/unit/test_capture_vocabulary.py`` pins the two
    against each other so a prompt edit cannot silently invent a kind the
    wiki has never heard of.

    ``NOTE`` is the fallback: the agent may leave ``kind`` empty, and the
    non-agent path stores raw text without asking anyone.
    """

    DECISION = "decision"
    DISCOVERY = "discovery"
    CORRECTION = "correction"
    REFERENCE = "reference"
    NOTE = "note"

    @classmethod
    def of(cls, value: str | None) -> CaptureKind:
        """Map an agent-reported kind onto the vocabulary.

        An unrecognized kind becomes :attr:`NOTE` — the item is still worth
        keeping, and the page type is written to frontmatter, so refusing the
        whole capture over a vocabulary drift would lose the content.
        """
        normalized = (value or "").strip().lower()
        member = cls._value2member_map_.get(normalized)
        return member if isinstance(member, cls) else cls.NOTE


class SourceZone(StrEnum):
    """Where a source note comes from, and where it is archived.

    The values are contract in three places at once: the subdirectory a
    note is archived under (``sources/_archive/<zone>/``), the ``source``
    provenance value in a note's frontmatter, and the OKF ``type`` label of
    the zone-written pages (``learn``, ``distilled``).

    ``MANUAL`` is not an archive zone: it is the provenance of a note a user
    wrote through ``kompend learn``. It lives here because it answers the
    same question as the other members — who produced this note.
    """

    CAPTURE = "capture"
    LEARN = "learn"
    MANUAL = "manual"
    DISTILLED = "distilled"
    UNKNOWN = "unknown"

    @property
    def source_subdir(self) -> str:
        """Source subdirectory this zone's notes are written to.

        ``MANUAL`` shares the learned zone: both are user-supplied facts, and
        only the frontmatter ``source`` value tells them apart.
        """
        match self:
            case SourceZone.CAPTURE:
                return CAPTURED_SUBDIR
            case SourceZone.LEARN | SourceZone.MANUAL:
                return LEARNED_SUBDIR
            case SourceZone.DISTILLED:
                return DISTILLED_SUBDIR
            case SourceZone.UNKNOWN:
                return ""

    @property
    def archive_subdir(self) -> str:
        """Subdirectory archived notes of this zone are moved under.

        This is the member value — notes already on disk live in
        ``sources/_archive/<zone>/`` and the layout is a contract, so the
        archive name is deliberately shorter than :attr:`source_subdir`.
        """
        return str(self)

    @classmethod
    def of_path(cls, path: Path | None) -> SourceZone:
        """Map a note's path onto the zone it was written by.

        A note outside the three tool-managed zones has no zone of its own;
        it is reported as :attr:`UNKNOWN` rather than guessed at, so an
        unforeseen layout cannot masquerade as a capture.
        """
        if path is None:
            return cls.UNKNOWN
        parts = path.parts
        for zone in (cls.CAPTURE, cls.LEARN, cls.DISTILLED):
            if zone.source_subdir in parts:
                return zone
        return cls.UNKNOWN
