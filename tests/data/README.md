# Test Data

```text
data/
├── sources/     ← raw documents (PDF, DOCX, MD…) for parse benchmarks
│   └── *.pdf
└── urls.txt     ← URLs for download integration tests (one per line, # comments)
```

## `sources/`

Place real document samples here. Tests in `test_parse_quality.py` and
`test_real_corpus.py` scan this directory for PDFs to parse.

Keep files small (< 5 MB each) so the repo doesn't bloat.

## `urls.txt`

One URL per line. Lines starting with `#` are comments.
These URLs are used by the URL download integration tests.
