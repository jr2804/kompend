"""Kind-vocabulary → output-folder mapping.

Lives next to :class:`~kompend.core.models.enums.NodeType` (the vocabulary
it is keyed by) rather than in :mod:`kompend.core.paths`, because
``core.paths`` cannot import ``core.models`` (``models.attachments``
imports ``core.paths`` — importing both ways would cycle). Kinds without
an entry (``overview`` …) render at the vault root.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final

from kompend.core.models.enums import NodeType


class KindFolder(StrEnum):
    """Output subfolder per rendered page kind.

    Single source of truth for the vault writers, the lint fixer (which
    computes the path a resolved wikilink points at), and the pipeline
    status view.
    """

    CONCEPTS = "concepts"
    ENTITIES = "entities"


#: Canonical page-kind → output-subfolder mapping, keyed by
#: :class:`~kompend.core.models.enums.NodeType` member values. Lookups with
#: a plain ``str`` kind work (``NodeType`` is a ``StrEnum``).
KIND_FOLDERS: Final[dict[str, str]] = {
    NodeType.CONCEPT.value: KindFolder.CONCEPTS.value,
    NodeType.COMPARISON.value: KindFolder.CONCEPTS.value,
    NodeType.ENTITY.value: KindFolder.ENTITIES.value,
    NodeType.SUMMARY.value: KindFolder.CONCEPTS.value,
}
