"""OKF frontmatter rendering shared by the vault writers and the capture writers.

Lives apart from :mod:`kompend.output.vault` so that :mod:`kompend.capture` can
render conformant frontmatter without importing the vault writer's store and
pipeline dependencies. ``vault.py`` re-exports these names, so the existing
import paths stay valid.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Final

import yaml

# OKF spec version every writer targets.
OKF_VERSION: Final[str] = "0.2"


class NoAliasDumper(yaml.Dumper):
    """YAML Dumper that disables anchor/alias emission."""

    def ignore_aliases(self, data: object) -> bool:  # noqa: PLR6301
        return True


def okf_frontmatter_yaml(frontmatter: dict[str, Any]) -> str:
    """Render frontmatter as YAML without anchors/aliases.

    Keys are rendered as plain strings: the OKF key contract is a
    ``StrEnum`` (:class:`kompend.core.models.enums.OkfField`), and a YAML
    dumper would otherwise emit a python-object tag for the enum member
    instead of the key name.
    """
    plain = {str(key): _wire_value(value) for key, value in frontmatter.items()}
    yaml_str = yaml.dump(
        plain,
        Dumper=NoAliasDumper,
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=True,
    ).strip()
    return f"---\n{yaml_str}\n---\n\n"


def _wire_value(value: Any) -> Any:
    """Replace a vocabulary member with the string it renders as.

    The OKF frontmatter is a wire format: a ``StrEnum`` reaches the YAML dumper
    as a Python object and renders as a ``!!python/object/apply`` tag, which no
    reader can load. ``str()`` of such a member is exactly its wire value.
    """
    return str(value) if isinstance(value, StrEnum) else value


def source_frontmatter(
    *,
    page_type: str,
    timestamp: str,
    timestamp_key: str,
    tags: list[str],
    title: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the OKF frontmatter mapping for a source-corpus page.

    ``okf_version`` is included on every source page: source notes are part of the
    same corpus as the compiled wiki and must declare the spec they were written
    against, or a reader cannot tell an older note from a conformant one.

    Values are returned as a mapping for :func:`okf_frontmatter_yaml` to render,
    never interpolated into a hand-built YAML string — a title or tag containing a
    colon or newline would produce invalid frontmatter.

    ``page_type`` and ``timestamp_key`` go through ``str()`` because callers pass
    vocabulary members: a ``StrEnum`` reaches the YAML dumper as a Python object
    and renders as a ``!!python/object/apply`` tag, while ``str()`` of one is its
    wire value.
    """
    frontmatter: dict[str, Any] = {
        "okf_version": OKF_VERSION,
        "type": str(page_type),
        str(timestamp_key): timestamp,
        "tags": list(tags),  # nocheck: frontmatter dict, not wiki metadata
    }
    if title is not None:
        frontmatter["title"] = title
    if extra:
        frontmatter.update(extra)
    return frontmatter
