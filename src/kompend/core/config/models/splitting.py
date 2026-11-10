"""Page-range source splitting (``[splitting]``).

A multi-document packet — one PDF holding a contract, its amendments and an
appendix — is one file with one identity, so every later stage treats its three
documents as one source: one chunk tree, one set of page kinds, one set of
citations. Splitting recognises the document boundaries the packet's pages
already carry and cuts the *parsed* document into virtual child sources before
chunking, so each document gets its own identity, its own cache entries and its
own compilation.

The block is opt-in because a split changes identity: source stems, cache keys,
state rows and provenance all gain a ``#p005-009`` page-range suffix. With
``enabled = false`` no splitter is built and Tier 1 behaves exactly as before.
"""

from __future__ import annotations

from pydantic import Field

from kompend.core.config.models._base import _KompendBaseModel
from kompend.core.keys import EnvVar


class SplittingConfig(_KompendBaseModel):
    """Detect and cut multi-document packets along page ranges.

    Boundaries come from the decision model in :attr:`judge_model` (or, when it
    is unset, from ``[decision]``). There is no deterministic stand-in: a page
    count cannot tell an amendment from a chapter, and a heuristic that guessed
    would fragment real documents — so with no judge reachable the source is
    kept whole and the decline is recorded.
    """

    enabled: bool = Field(
        default=False,
        validation_alias=EnvVar.SPLITTING_ENABLED,
        description=(
            "Cut multi-document packets into one virtual source per document before chunking. "
            "Opt-in: a split changes every child's stem (`packet#p005-009`), and therefore its "
            "cache keys, state row and provenance. Off, Tier 1 is byte-identical to a run "
            "without splitting."
        ),
    )
    judge_model: str | None = Field(
        default=None,
        validation_alias=EnvVar.SPLITTING_JUDGE_MODEL,
        description=(
            "Full `provider/model` string of the decision model that finds document boundaries, "
            "for example `ollama/clef` or `typesafe/jev-latest`. Unset falls back to "
            "`decision.model` (with `decision.enabled`); with neither set nothing is split and "
            "the decline is recorded in the ledger."
        ),
    )
    judge_timeout: float = Field(
        default=20.0,
        gt=0.0,
        validation_alias=EnvVar.SPLITTING_JUDGE_TIMEOUT,
        description=(
            "Seconds one boundary question may take. The judge answers about a whole packet, so "
            "this sits above `decision.timeout` (10 s); on expiry the source is kept whole."
        ),
    )
    rules_file: str | None = Field(
        default=None,
        validation_alias=EnvVar.SPLITTING_RULES_FILE,
        description=(
            "Vault-relative YAML file of document categories (`categories: [{id, description}]`) "
            "the judge may assign, or an absolute path. Unset uses the built-in rules shipped "
            "with kompend (`templates/splitting/rules.yaml`). A missing or unparseable file "
            "declines the split — categories are never guessed."
        ),
    )
    min_pages: int = Field(
        default=6,
        ge=2,
        validation_alias=EnvVar.SPLITTING_MIN_PAGES,
        description=(
            "Fewest pages a source must have for a split to be considered. A packet below this "
            "floor cannot plausibly hide two documents, so no judge call is spent on it. `2` "
            "allows any source with more than one page."
        ),
    )
    max_page_chars: int = Field(
        default=12_000,
        ge=1,
        validation_alias=EnvVar.SPLITTING_MAX_PAGE_CHARS,
        description=(
            "Most characters one page may contribute to the judge's request. A page above it "
            "declines the whole split instead of being cut: a truncated page is a page the judge "
            "did not see, and the doctrine forbids cropping. Raise it when scanned pages "
            "(garbled OCR, dense tables) exceed the default."
        ),
    )


__all__ = ["SplittingConfig"]
