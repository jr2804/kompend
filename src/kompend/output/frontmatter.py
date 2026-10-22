"""OKF frontmatter rendering shared by the vault writers and the capture writers.

Lives apart from :mod:`kompend.output.vault` so that :mod:`kompend.capture` can
render conformant frontmatter without importing the vault writer's store and
pipeline dependencies. ``vault.py`` re-exports these names, so the existing
import paths stay valid.
"""

from __future__ import annotations

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
    plain = {str(key): value for key, value in frontmatter.items()}
    yaml_str = yaml.dump(
        plain,
        Dumper=NoAliasDumper,
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=True,
    ).strip()
    return f"---\n{yaml_str}\n---\n\n"


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

    ``okf_version`` is included on every source page, which is the point of the
    migration: source notes are part of the same corpus as the compiled wiki and
    must declare the spec they were written against, or a reader cannot tell a
    pre-migration note from a conformant one.

    Values are returned as a mapping for :func:`okf_frontmatter_yaml` to render,
    rather than interpolated into a hand-built YAML string — a title or tag
    containing a colon or newline used to produce invalid frontmatter.
    """
    frontmatter: dict[str, Any] = {
        "okf_version": OKF_VERSION,
        "type": page_type,
        timestamp_key: timestamp,
        "tags": list(tags),  # nocheck: frontmatter dict, not wiki metadata
    }
    if title is not None:
        frontmatter["title"] = title
    if extra:
        frontmatter.update(extra)
    return frontmatter
