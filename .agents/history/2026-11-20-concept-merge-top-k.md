# Concept merge — the per-cluster top-K merge (2026-11-20)

**Bead:** `knox-0ybl` (child of the epic `knox-tot0`).

## Context

Run `bc35f97e` compiled a corpus of five revisions of the same documentation:
230 of 247 concepts landed in one cluster. The reduction guard
(`concept_merge_max_reduction = 0.5`) refused the whole merge, so 249
near-duplicates fed 32 generation segments — the refusal was right, the
outcome was still wrong.

## What landed

`flows/assembly.py` merges an oversized cluster pairwise instead of refusing
it wholesale (`_top_k_merge`):

- **Which pairs** — the most similar ones, and only those within
  `compile.budget.concept_merge_similarity_margin` of the cluster's own best
  pair. The margin is relative to the cluster's measured similarity, so no
  absolute similarity constant decides a merge.
- **K — the pairs one cluster may merge** — the cluster's share of what is
  left of the pool-wide budget (`concept_merge_max_reduction * pool_size`),
  shared by size among the oversized clusters, capped by the source cap:
  `K = min(budget_share, m - ceil(S / concept_merge_max_sources))`, where
  `m` is the cluster's size and `S` the distinct sources it holds.
- **Rounds** — at most `ceil(log2(m))` rounds (a group at most doubles per
  round), and a round that merges fewer than `ceil(K / rounds)` pairs ends
  the collapse.

Every bound is a ratio or a count derived from the pool, from the cluster, or
from the similarity the run measured — never an absolute number. The merged
concept's sources stay within `concept_merge_max_sources`, because a pair
that would break the cap is not admissible.

## Verification

`tests/unit/test_concept_merge.py::TestOversizedCluster` — four topics with
three revisions each inside one cluster of a twenty-concept pool: the
revisions collapse into four concepts, the eight unrelated fillers are
untouched, and the refusal is logged. The guard tests that pinned the
wholesale refusal now pin the budget instead: the pool shrinks by exactly
the share the guard allows.
