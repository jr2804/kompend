"""Structural lint family: duplicate bibliographic references across sources.

Loads ``.kompend/references.json`` and flags references cited by more than
one source — exact (DOI, title+author, raw) or fuzzy (embedding cosine
similarity). Informational findings: duplicates are expected in
multi-source corpora.
"""

from __future__ import annotations

from pathlib import Path

from kompend.core.models import CompiledWiki
from kompend.core.paths import REFERENCES_FILE, VaultPaths
from kompend.core.pipeline.references import ReferenceStore, consolidate_references
from kompend.lint.base import BaseLintRule, LintResult, LintRuleOutcome, Severity
from kompend.lint.details import DuplicateReferenceDetail, LintField, NearDuplicateReferenceDetail
from kompend.types import JSONDict

#: Max chars of a reference title used for display labels in lint findings.
_REF_LABEL_TITLE_MAX = 40

#: Max chars of a duplicate title kept in a duplicate finding as evidence.
_DUPLICATE_TITLE_MAX = 60

#: Max chars of a duplicated raw reference kept in a duplicate finding.
_DUPLICATE_RAW_MAX = 80


class DuplicateReferencesRule(BaseLintRule):
    """Detect duplicate bibliographic references across sources.

    Loads ``.kompend/references.json`` and checks for references that appear
    in multiple source files — same DOI, same title+first-author pair, or
    identical ``raw`` string. When ``embedding_model`` is set, also
    detects near-duplicates via cosine similarity on title embeddings
    (same pattern as concept merging in ``assembly.py``).

    Duplicates are informational (not errors) because they are expected
    in multi-source corpora.
    """

    name = "duplicate-references"
    description = "Bibliographic references cited by more than one source"

    def __init__(
        self,
        *,
        embedding_model: str = "",
        fuzzy_threshold: float = 0.85,
    ) -> None:
        """Initialize the rule.

        Parameters
        ----------
        embedding_model:
            When non-empty, enables fuzzy dedup via embedding cosine similarity.
        fuzzy_threshold:
            Similarity threshold for fuzzy matching (0.0–1.0).
        """
        self.embedding_model = embedding_model
        self.fuzzy_threshold = fuzzy_threshold

    def check(self, wiki: CompiledWiki, *, sources_dir: str | None = None) -> list[LintResult]:
        results: list[LintResult] = []
        if sources_dir is None:
            results.append(LintResult(rule=self.name, outcome=LintRuleOutcome.SKIP, message="No sources directory provided"))
            return results

        base_dir = Path(sources_dir).parent
        store = ReferenceStore(VaultPaths(base_dir).references_file)
        raw_data = store.to_dict()
        if not raw_data:
            msg = f"No {REFERENCES_FILE} found — run 'kompend compile' with reference extraction"
            results.append(LintResult(rule=self.name, outcome=LintRuleOutcome.SKIP, message=msg))
            return results

        duplicates = self._find_exact_duplicates(raw_data)
        near_dupes = self._find_fuzzy_duplicates(store)

        if duplicates or near_dupes:
            all_findings = duplicates + near_dupes
            msg = f"{len(all_findings)} reference finding(s) ({len(duplicates)} exact, {len(near_dupes)} fuzzy) across {store.source_count()} source(s)"
            results.append(
                LintResult(
                    rule=self.name,
                    outcome=LintRuleOutcome.FAIL,
                    severity=Severity.INFO,
                    message=msg,
                    details={LintField.DUPLICATES: duplicates, LintField.NEAR_DUPLICATES: near_dupes},
                )
            )
        else:
            msg = f"No cross-source duplicate references ({store.total_count()} refs across {store.source_count()} source(s))"
            results.append(LintResult(rule=self.name, outcome=LintRuleOutcome.PASS, message=msg))
        return results

    def _find_fuzzy_duplicates(self, store: ReferenceStore) -> list[NearDuplicateReferenceDetail]:
        """Find near-duplicate references via embedding cosine similarity."""
        near_dupes: list[NearDuplicateReferenceDetail] = []
        if not self.embedding_model:
            return near_dupes

        all_refs = store.all_references()
        consolidated = consolidate_references(
            all_refs,
            self.embedding_model,
            threshold=self.fuzzy_threshold,
        )
        fuzzy_removed = len(all_refs) - len(consolidated)
        if fuzzy_removed > 0:
            near_dupes.append(
                {
                    "type": "fuzzy",
                    "removed": fuzzy_removed,
                    "threshold": self.fuzzy_threshold,
                    "remaining": len(consolidated),
                }
            )
        return near_dupes

    @staticmethod
    def _find_exact_duplicates(
        raw_data: dict[str, list[JSONDict]],
    ) -> list[DuplicateReferenceDetail]:
        """Find exact-duplicate references (DOI, title+author, raw)."""
        by_doi: dict[str, list[tuple[str, str]]] = {}
        by_title_author: dict[str, list[tuple[str, str]]] = {}
        by_raw: list[tuple[str, str, str]] = []

        for source, ref_list in raw_data.items():
            source_name = Path(source).name
            for ref_dict in ref_list:
                title_field = _text_field(ref_dict, "title")
                label = _text_field(ref_dict, "id") or title_field[:_REF_LABEL_TITLE_MAX]
                doi = _text_field(ref_dict, "doi").strip().lower()
                title = title_field.strip().lower()
                raw = _text_field(ref_dict, "raw").strip()
                first_family = _first_author_family(ref_dict)

                if doi:
                    by_doi.setdefault(doi, []).append((source_name, label))
                if title and first_family:
                    key = f"{first_family.lower()}|{title}"
                    by_title_author.setdefault(key, []).append((source_name, label))
                if raw:
                    by_raw.append((raw.lower(), source_name, label))

        duplicates: list[DuplicateReferenceDetail] = []

        for doi, entries in by_doi.items():
            sources_seen = {s for s, _ in entries}
            if len(sources_seen) > 1:
                refs = [lbl for _, lbl in entries]
                duplicates.append({"type": "doi", "doi": doi, "sources": sorted(sources_seen), "refs": refs})

        for key, entries in by_title_author.items():
            sources_seen = {s for s, _ in entries}
            if len(sources_seen) > 1:
                family, title = key.split("|", 1)
                refs = [lbl for _, lbl in entries]
                duplicates.append(
                    {"type": "title-author", "author": family, "title": title[:_DUPLICATE_TITLE_MAX], "sources": sorted(sources_seen), "refs": refs}
                )

        raw_seen: dict[str, list[str]] = {}
        for raw_text, source_name, _ in by_raw:
            raw_seen.setdefault(raw_text, []).append(source_name)
        for raw_text, sources_for_raw in raw_seen.items():
            unique_sources = set(sources_for_raw)
            if len(unique_sources) > 1:
                duplicates.append({"type": "raw", "raw": raw_text[:_DUPLICATE_RAW_MAX], "sources": sorted(unique_sources)})

        return duplicates


def _text_field(reference: JSONDict, key: str) -> str:
    """One CSL field read as text, empty when the payload holds another type.

    A stored reference is CSL JSON — every value is a ``JSONValue``, so a field
    that should be a string has to be narrowed before it can be stripped or
    sliced. Reading it as text and defaulting to empty keeps a malformed record
    from raising mid-scan; the finding it produces is still worth reporting.
    """
    value = reference.get(key)
    return value if isinstance(value, str) else ""


def _first_author_family(reference: JSONDict) -> str:
    """The first author's family name, empty when the payload carries none."""
    authors = reference.get("author")
    if not isinstance(authors, list) or not authors:
        return ""
    first = authors[0]
    if not isinstance(first, dict):
        return ""
    family = first.get("family")
    return family if isinstance(family, str) else ""
