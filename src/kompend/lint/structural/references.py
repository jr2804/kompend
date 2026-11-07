"""Structural lint family: duplicate bibliographic references across sources.

Loads ``.kompend/references.json`` and flags references cited by more than
one source — exact (DOI, title+author, raw) or fuzzy (embedding cosine
similarity). Informational findings: duplicates are expected in
multi-source corpora.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kompend.core.models import CompiledWiki
from kompend.core.paths import REFERENCES_FILE, VaultPaths
from kompend.core.pipeline.references import ReferenceStore, consolidate_references
from kompend.lint.base import BaseLintRule, LintResult, LintRuleOutcome, Severity

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
                    details={"duplicates": duplicates, "near_duplicates": near_dupes},
                )
            )
        else:
            msg = f"No cross-source duplicate references ({store.total_count()} refs across {store.source_count()} source(s))"
            results.append(LintResult(rule=self.name, outcome=LintRuleOutcome.PASS, message=msg))
        return results

    def _find_fuzzy_duplicates(self, store: ReferenceStore) -> list[dict[str, Any]]:
        """Find near-duplicate references via embedding cosine similarity."""
        near_dupes: list[dict[str, Any]] = []
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
        raw_data: dict[str, list[dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        """Find exact-duplicate references (DOI, title+author, raw)."""
        by_doi: dict[str, list[tuple[str, str]]] = {}
        by_title_author: dict[str, list[tuple[str, str]]] = {}
        by_raw: list[tuple[str, str, str]] = []

        for source, ref_list in raw_data.items():
            source_name = Path(source).name
            for ref_dict in ref_list:
                label = ref_dict.get("id") or ref_dict.get("title", "")[:_REF_LABEL_TITLE_MAX]
                doi = ref_dict.get("doi", "").strip().lower()
                title = ref_dict.get("title", "").strip().lower()
                raw = ref_dict.get("raw", "").strip()
                authors = ref_dict.get("author", [])
                first_family = ""
                if authors and isinstance(authors, list):
                    first_family = authors[0].get("family", "") if isinstance(authors[0], dict) else ""

                if doi:
                    by_doi.setdefault(doi, []).append((source_name, label))
                if title and first_family:
                    key = f"{first_family.lower()}|{title}"
                    by_title_author.setdefault(key, []).append((source_name, label))
                if raw:
                    by_raw.append((raw.lower(), source_name, label))

        duplicates: list[dict[str, Any]] = []

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
