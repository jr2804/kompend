---
---

# Content Duplicate Detection

## Overview

When ingesting documents from diverse sources (different URLs, different file
formats), the same paper may appear multiple times — an author preprint on
arXiv, a published version in conference proceedings, or a local copy in a
different format. Binary SHA256 hashing cannot detect these duplicates because
the bytes differ even when the content is identical.

kompend provides configurable, content-level duplicate detection that runs
during the ingest phase, **after parsing but before chunking**. It compares
parsed text content using two complementary methods, producing a verdict for
each document.

## How It Works

```text
download → parse → DEDUP → chunk → resolve → compile → output
```

1. Each document is parsed to text and cached (regardless of original format).
2. In a separate pass after all files are parsed, the batch is fingerprinted
   (embedding vectors and/or MinHash signatures). This avoids GPU memory
   contention between GPU-bound parsers and the embedding model.
3. A single batch comparison runs — each new document is compared against
   all previously ingested documents **and** against other documents in the
   same batch. Comparisons are visible as a progress bar during ingestion:
   ``dedup 12/45``, where the total is ``N × M + N×(N−1)÷2``.
4. A verdict is produced based on configurable thresholds.
5. Fingerprints and verdicts are stored in ``.kompend/dedup/fingerprints.json``.
6. During the next ``kompend compile``, the pipeline reads the store:
   - **identical** documents are silently skipped — no chunking, no LLM calls
   - **similar** documents get a ``[!warning]`` callout with a link to the
     closest match injected into the compiled page
   - **different** documents are processed normally

## Configuration

All settings are configurable via **TOML config file**, **environment variable**,
or **CLI arg** (when the ingest command supports it).

### TOML Reference

```toml
[dedup]
# Enable content-level duplicate detection
# enabled = false

# Comma-separated matchers to use: "embedding", "minhash", or both
# matchers = "embedding,minhash"

# Cosine similarity threshold for "identical" verdict (>95%)
# threshold_identical = 0.95

# Cosine similarity threshold for "similar" verdict (>80%)
# threshold_similar = 0.80

# Embedding model override (requires an explicit provider prefix; no local
# fallback). An empty string "" is treated the same as None / not set — the
# embedding matcher is skipped.
# model = ""

# Jaccard threshold for MinHash LSH index
# minhash_lsh_threshold = 0.8
```

### Environment Variables

| Variable | Overrides |
|---|---|
| `KOMP_DEDUP_ENABLED` | `dedup.enabled` |
| `KOMP_DEDUP_THRESHOLD_IDENTICAL` | `dedup.threshold_identical` |
| `KOMP_DEDUP_THRESHOLD_SIMILAR` | `dedup.threshold_similar` |
| `KOMP_DEDUP_MODEL` | `dedup.model` |
| `KOMP_DEDUP_MATCHERS` | `dedup.matchers` |
| `KOMP_DEDUP_MINHASH_LSH_THRESHOLD` | `dedup.minhash_lsh_threshold` |

## Matcher Details

### Embedding Matcher

Uses [sentence-transformers](https://www.sbert.net/) (or any pydantic-ai-compatible
embedding provider) to compute document vectors, then compares via cosine
similarity. The model must specify an explicit provider prefix (e.g.
`ollama:granite-embedding:278m`); there is no built-in local model.

- **Model:** Configurable; must include a provider prefix.
- **Chunking:** Long documents are split into word chunks and mean-pooled into
  a single vector.
- **Strengths:** Captures semantic similarity — same paper in different formats
  (PDF vs HTML vs DOCX) produces similar embeddings.
- **Blind spots:** Chunking long documents introduces centroid drift. A 50-page
  paper chunked into pieces averages signal across topics, potentially reducing
  similarity between versions with different section ordering or significant
  additions. Different languages will be near-orthogonal.

### MinHash Matcher

Uses [datasketch](https://github.com/ekzhu/datasketch) to generate MinHash
signatures and estimate Jaccard similarity. Supports LSH indexing for
sub-linear candidate lookup on large corpora.

- **Tokenisation:** Word-level 5-grams by default.
- **Strengths:** Fast, compact signatures. Excellent at detecting exact or
  near-exact text reuse. LSH enables scaling to thousands of documents.
- **Blind spots:** Purely lexical — rewording, synonym substitution, or
  sentence restructuring will lower scores even for semantically identical
  content. Boilerplate text (licenses, templates) shared across different
  documents can produce false positives.

### Combined Mode

When `matchers = "embedding,minhash"`, both methods run and the **maximum**
of the two scores is used to determine the verdict:

```python
combined = max(embedding_score, minhash_score)
```

Both scores and the winning method are stored in document metadata. Using
`max` is conservative — a document is flagged if *either* method detects
similarity. This is the correct tradeoff: false positives (flagging a unique
document) produce metadata notes for review, while false negatives (missing a
duplicate) waste compute.

## What Cannot Be Detected

Even combined, neither method handles these cases well:

- **Same paper in different languages** — embedding vectors for different
  languages are near-orthogonal; MinHash shingles share no overlap.
- **Heavily revised content** — a paper where the author has restructured
  sections, rewritten large passages, or added significant new material may
  score below the `similar` threshold, even though a human might recognise
  the lineage.
- **Small overlap + large additions** — adding 80% new content to an existing
  paper dilutes fingerprints.

These cases are *correctly* treated as different content — if a human reviewer
would not confidently call them duplicates, the system should not either.

## Verdicts

| Verdict | Threshold | Behaviour |
|---|---|---|
| `identical` | ≥ `threshold_identical` (default 0.95) | Document is cached but excluded from chunking/compilation. Metadata references the original kept source. |
| `similar` | ≥ `threshold_similar` (default 0.80) | Processed normally. Metadata includes a duplication warning and a link to the nearest match. |
| `different` | < `threshold_similar` | Normal processing, no dedup handling. |

## Metadata

After dedup runs, each document in the persistent store
(`.kompend/dedup/fingerprints.json`) has an entry like:

```json
{
  "verdict": "identical",
  "closest_match": "paper-title-slug",
  "closest_score": 0.97,
  "closest_method": "minhash",
  "embedding_score": 0.95,
  "minhash_score": 0.97
}
```

The `verdict` field is what the compile pipeline reads to decide whether to
skip or process the document.

If a newly ingested document scores higher against an existing document than
that existing document's previous closest match, the existing document's
metadata is updated — no recomputation needed. During compile, similar
documents get a Markdown callout injected into the page:

```markdown
> [!warning] Possible duplicate
> This document is similar to **paper-title-slug** (score=0.83).
```

This callout is added to `parsed.metadata["_dedup_warning"]` and rendered
by the Markdown output writer.

## Performance

- **Embedding:** provider-dependent latency; scales as O(n·m) — fine for
  typical usage (tens to low hundreds of sources). For very large corpora,
  consider enabling only MinHash with LSH.
- **MinHash:** < 10ms per document pair. LSH indexing provides sub-linear
  candidate lookup for corpora > 1000 documents.
- **Storage:** ~3 KB per document (embedding vector + MinHash signature).

## Storage Invalidation

The fingerprint store includes a version stamp computed from all relevant
config keys (`dedup.matchers`, `dedup.model`, `dedup.minhash_lsh_threshold`,
`compile.embedding.model`). If the config changes, the store is detected as
stale and fingerprints are rebuilt from cached parsed documents — no
re-downloading or re-parsing is required.
