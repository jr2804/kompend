# Concept merge — the refusal logs the cluster-size distribution (2026-11-20)

**Bead:** `knox-154k` (child of the epic `knox-tot0`).

## Context

The refusal line named the refused cluster ("merging 230 concepts holding 5
sources") and nothing else. A run that refuses a merge therefore says nothing
about the shape of the redundancy around it — how many clusters there were,
how large the next ones were, how many sources each held — so the next run
starts without the picture of what the merge wanted to do.

## What landed

`flows/assembly.py::_log_refusal` now names the distribution with the refusal:
the clusters are taken from the largest down until they hold more than half
the pool, each as "<size> concepts/<sources> sources", next to the pool size
and the cluster count. The head of the distribution shows the redundancy that
explains the merge instead of every cluster of a large corpus.

The refusal path stays one path: the wholesale refusal and the partially
applied merge log through the same helper, with the same distribution.

## Verification

`tests/unit/test_concept_merge.py::TestRefusalLogging` — a pool of nine
concepts in three clusters; the refusal names the largest cluster
("4 concepts/4 sources") and the next one ("3 concepts/3 sources") next to
"pool 9 concepts in 3 clusters".
