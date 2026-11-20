# Concept merge — corpus-relative thresholds (2026-11-20)

**Bead:** `knox-4ntp` (child of the epic `knox-tot0`).

## Context

An absolute similarity constant is a statement about a distribution, not about
a concept: the same cut that separates a document revision from its sibling
on a five-revision corpus merges a distinct sub-topic on a thousand-document
corpus, and the failure is silent because the merge succeeds. The knob that
does not scale is therefore only allowed to select candidates; the amount of
merging has to come from the corpus.

## What landed

`docs/architecture/decisions.md` records the decision under **Compilation**:
the threshold selects the candidate pairs, and the corpus-relative caps decide
how much of the corpus those candidates may remove —
`concept_merge_max_reduction` (the fraction of the pool one merge may
remove), `concept_merge_max_sources` (the sources one merged concept may
hold), and `concept_merge_similarity_margin` (a fraction of the cluster's own
best pair, so it means the same on every corpus). The derivations of the pair
count and the round bound are in `2026-11-20-concept-merge-top-k.md`.

## Verification

`uv run mkdocs build --strict` — the decision log is in the site navigation,
so the entry is built with the documentation.
