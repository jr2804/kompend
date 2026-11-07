"""Vault page layout — where a rendered page lands inside the bundle.

Only the *page* layout lives here: the kind→folder vocabulary and the
bundle-relative path a page is written to. Artifact names (``index.md``,
``log.md``, the output directory, …) belong to :mod:`kompend.core.paths`;
this module derives a path from them and never invents one.

``MarkdownWriter._write_index`` renders its catalog links from its own
template instead: an empty page slug yields an ``index.md`` entry that no
page file backs, and unifying the two would change the written index.

The mixin keeps the documented ``VaultWriter._KIND_FOLDERS`` /
``VaultWriter._folder_for_kind`` entry points working while the definition
lives in one place.
"""

from __future__ import annotations

from typing import ClassVar, Final

from kompend.core.models import KIND_FOLDERS
from kompend.core.models.enums import NodeType

__all__ = ["UNSLUGGED_PAGE", "PageLayoutMixin", "bundle_path", "page_stem"]

#: Filename stem used when a node carries no page slug.
UNSLUGGED_PAGE: Final[str] = "untitled"


def page_stem(slug: str) -> str:
    """Filename stem for a page slug, falling back to :data:`UNSLUGGED_PAGE`."""
    return slug or UNSLUGGED_PAGE


def bundle_path(folder: str, slug: str) -> str:
    """Bundle-relative markdown path for the page *slug* in *folder*."""
    name = page_stem(slug)
    return f"{folder}/{name}.md" if folder else f"{name}.md"


class PageLayoutMixin:
    """Vault-relative page routing shared by the writers and the index pages."""

    # node_type -> subfolder under the vault root. Kinds not listed here
    # (overview, procedure, section, unknown) stay at the root so the
    # layout never splits a type the compiler didn't anticipate.
    # Canonical kind→folder map lives in core.models.kinds (shared with the
    # lint fixer).
    _KIND_FOLDERS: ClassVar[dict[str, str]] = KIND_FOLDERS

    @classmethod
    def _folder_for_kind(cls, node_type: NodeType, sub: str = "") -> str:
        """Return the subfolder for a page kind, or '' to stay at the root.

        Entity pages route into ``entities/<sub>/`` — the sub-type
        (person/organization/standard/...) from node metadata.
        """
        folder = cls._KIND_FOLDERS.get(node_type.value, "")
        if folder and sub:
            return f"{folder}/{sub}"
        return folder
