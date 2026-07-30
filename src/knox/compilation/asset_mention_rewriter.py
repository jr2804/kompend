"""Rewrite source-relative figure/table mentions into stable asset-id references.

After structured assets are separated (see
:mod:`knox.compilation.asset_registry`), the extraction model still emits
prose like *"as shown in Figure 2"* or *"Table 3 demonstrates"*. This module
replaces those source-relative mentions with stable asset-id references
(``¶ fig-9c2e`` / ``¶ tab-a1f3``) so downstream consumers (generation, OKF
footnote rendering) can resolve them against the
:class:`~knox.core.pipeline.assets.AssetStore`.

Only mentions with a matching asset are rewritten; unmatched mentions
(e.g. a forward reference to a figure that wasn't detected as structured) are
left untouched so the prose stays readable.
"""

from __future__ import annotations

import re

from knox.compilation.asset_registry import AssetIdMap
from knox.core.models.enums import AssetKind

__all__ = ["rewrite_asset_mentions"]

#: Regex matching ``Figure 2``, ``Fig. 2``, ``Fig 2``, ``Table 3``, ``Tab. 3``.
#: Case-insensitive; captures the kind (fig/tab) and the number.
_MENTION_RE = re.compile(
    r"\b(Fig(?:ure|\.)?|Tab(?:le|\.)?)\s+(\d+)\b",
    flags=re.IGNORECASE,
)

def rewrite_asset_mentions(text: str, id_map: AssetIdMap) -> str:
    """Replace ``Figure N`` / ``Table N`` mentions with ``¶ <asset-id>``.

    Parameters
    ----------
    text:
        Prose that may contain source-relative figure/table mentions.
    id_map:
        Per-source asset id map (built by
        :func:`knox.compilation.asset_registry.assign_asset_ids`).

    Returns:
    -------
    str
        The prose with matching mentions rewritten. Unmatched mentions are
        left unchanged.
    """

    def _replace(m: re.Match[str]) -> str:
        kind = _kind_for(m.group(1))
        if kind is None:
            return m.group(0)
        number = int(m.group(2))
        asset_id = id_map.asset_for_label(kind, number)
        if asset_id is None:
            return m.group(0)
        return f"¶ {asset_id}"

    return _MENTION_RE.sub(_replace, text)


def _kind_for(match_kind: str) -> AssetKind | None:
    """Return the :class:`AssetKind` for a regex kind group, or ``None``.

    The enum accepts both canonical names (``figure``/``table``) and aliases
    (``fig``/``tab``) via :class:`aenum.MultiValueEnum`.
    """
    try:
        return AssetKind(match_kind.lower().rstrip("."))
    except ValueError:
        return None
