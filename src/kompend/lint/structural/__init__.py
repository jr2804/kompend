"""Structural lint rules — deterministic, zero LLM cost.

Each rule is a class that satisfies the :class:`LintRule` protocol.
Rules inspect a :class:`CompiledWiki` (and optionally the sources
directory on disk) and return :class:`LintResult` findings.

The rules live in one module per family — ``links``, ``images``,
``structure``, ``metadata``, ``sources``, ``attachments``, ``references``
and ``markdown``. This module is the registry: it owns
``STRUCTURAL_RULES`` and ``RULE_BY_NAME`` and re-exports the rule
classes, so ``kompend.lint.structural`` stays the one import path.
"""

from __future__ import annotations

from typing import Any

from kompend.lint.base import BaseLintRule
from kompend.lint.base import Severity as Severity
from kompend.lint.structural.attachments import OrphanAttachmentsRule as OrphanAttachmentsRule
from kompend.lint.structural.attachments import UnreferencedAttachmentsRule as UnreferencedAttachmentsRule
from kompend.lint.structural.attachments import orphan_attachment_dirs as orphan_attachment_dirs
from kompend.lint.structural.attachments import unreferenced_attachment_files as unreferenced_attachment_files
from kompend.lint.structural.images import ImageEmbedRule as ImageEmbedRule
from kompend.lint.structural.links import BrokenWikilinksRule as BrokenWikilinksRule
from kompend.lint.structural.links import KompendReferenceRule as KompendReferenceRule
from kompend.lint.structural.links import _node_lookup_index as _node_lookup_index
from kompend.lint.structural.markdown import RumdlFinding as RumdlFinding
from kompend.lint.structural.markdown import RumdlRule as RumdlRule
from kompend.lint.structural.metadata import MissingFrontmatterRule as MissingFrontmatterRule
from kompend.lint.structural.references import DuplicateReferencesRule as DuplicateReferencesRule
from kompend.lint.structural.sources import SourceNotFoundRule as SourceNotFoundRule
from kompend.lint.structural.sources import StalePageRule as StalePageRule
from kompend.lint.structural.structure import DuplicateTitleRule as DuplicateTitleRule
from kompend.lint.structural.structure import EmptyPageRule as EmptyPageRule
from kompend.lint.structural.structure import HeadingSkipRule as HeadingSkipRule
from kompend.lint.structural.structure import OrphanPageRule as OrphanPageRule

__all__ = [
    "RULE_BY_NAME",
    "STRUCTURAL_RULES",
    "BrokenWikilinksRule",
    "DuplicateReferencesRule",
    "DuplicateTitleRule",
    "EmptyPageRule",
    "HeadingSkipRule",
    "ImageEmbedRule",
    "KompendReferenceRule",
    "MissingFrontmatterRule",
    "OrphanAttachmentsRule",
    "OrphanPageRule",
    "RumdlFinding",
    "RumdlRule",
    "Severity",
    "SourceNotFoundRule",
    "StalePageRule",
    "UnreferencedAttachmentsRule",
    "orphan_attachment_dirs",
    "unreferenced_attachment_files",
]

# ── Registry ────────────────────────────────────────────────────────────

STRUCTURAL_RULES: list[BaseLintRule] = [
    ImageEmbedRule(),
    BrokenWikilinksRule(),
    KompendReferenceRule(),
    MissingFrontmatterRule(),
    OrphanPageRule(),
    DuplicateTitleRule(),
    EmptyPageRule(),
    HeadingSkipRule(),
    StalePageRule(),
    SourceNotFoundRule(),
    OrphanAttachmentsRule(),
    UnreferencedAttachmentsRule(),
]

RULE_BY_NAME: dict[str, Any] = {r.name: r for r in STRUCTURAL_RULES}
# RumdlRule and DuplicateReferencesRule are not singletons — register factory
# references so they can be looked up by name and instantiated with config
# values in LintRunner._resolve_rules.
RULE_BY_NAME["rumdl"] = RumdlRule
RULE_BY_NAME["duplicate-references"] = DuplicateReferencesRule
