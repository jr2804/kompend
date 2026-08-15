# Test Data

```text
data/
├── sources/     ← Batch 1 — initial raw documents + URLs
│   ├── *.pdf
│   └── urls.txt
├── batch2/      ← Batch 2 — incremental sources added in a second pass
│   ├── *.pdf    ← (continuous workflow demo: drop new PDFs/DOCX here)
│   └── urls.txt
└── README.md
```

## `sources/` (Batch 1)

The initial document set. Tests in `test_parse_quality.py` and
`test_real_corpus.py` scan this directory for PDFs to parse, and
`scripts/build/build-wiki.ps1` ingests it for the first compile pass. Download
URLs for the first pass live in `sources/urls.txt`.

Keep files small (< 5 MB each) so the repo doesn't bloat.

## `batch2/` (Batch 2)

Incremental sources for the **continuous workflow demo**. Same domain
as `sources/` (speech intelligibility, listening effort, LEAP, hearing
aids) but added on the *second* compile pass — after the wiki already
exists. This demonstrates the incremental / knowledge-growth workflow:

```text
ingest batch2/ → compile --start-at-tier chunk → lint → history
```

Drop additional PDFs / DOCX here and list any download URLs in
`batch2/urls.txt`. Files stay out of the first-pass tests so the
baseline benchmarks are unaffected.

## `batch2/urls.txt`

One URL per line. Lines starting with `#` are comments.
Used by the second (incremental) ingest pass in the continuous-workflow demo.
