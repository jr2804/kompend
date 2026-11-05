# sqlite-vec RAG backend (2026-11-05)

Bead: `knox-zzg4` — *Replace zvec RAG backend with sqlite-vec + FlashRank
reranker*.

## Why zvec had to go

`import zvec` dies with SIGILL on a CPU without AVX: the wheel's ELF
constructor runs VEX-encoded instructions before any runtime dispatch
(`vmovd xmm1, edx` at offset 0x36a32a). There is no fallback and no
environment variable that helps — `ARROW_USER_SIMD_LEVEL=NONE` changes
nothing, because the crash is in the static initialiser, not in a dispatched
kernel. zvec 0.7.0 behaves the same, and there is no source distribution on
PyPI to rebuild.

The failure was not cosmetic: `tests/conftest.py` imports the vector store, so
the **entire unit suite could not even be collected** on such a machine
(exit 132). The store also pulled the Proxima engine and RocksDB into every
CLI start, and the whole thing had to be rebuilt for a schema change.

## What replaced it

**sqlite-vec** (Apache 2.0, `asg017/sqlite-vec`) — a vector extension for
SQLite. The vectors, the lexical index and the filterable metadata live in one
file, `.kompend/vectors.db`:

- `vec_items` — a `vec0` virtual table with `distance_metric=cosine`, so
  `1 - distance` is exactly the cosine similarity and the previous backend's
  score convention carries over unchanged.
- `vec_fts` — an FTS5 table over `title` + `content_preview` (the lexical
  recall stays preview-bounded; page bodies stay in the markdown).
- `vec_item_tags` / `vec_item_attributes` / `vec_item_attr_keys` — side tables
  with indexes, so the filters are indexed `EXISTS` subqueries rather than a
  filter string.
- `vec_meta` — the store's own fingerprint (backend, dimension, schema); a
  mismatch drops the file and rebuilds it.

The dependency itself is `sqlite-vec` plus **`sqlite-vec-client`** for the
connection layer only (the extension load and the pragma set). Its
`ConnectionPool` is available for a threaded caller, but kompend does not need
it yet: every store instance owns one connection and its caller closes it, so
no connection is ever shared across threads. Its CRUD layer was not adopted: its fixed schema (`rowid, text, metadata BLOB,
text_embedding BLOB`) has no FTS5 index, no drift guard, and stores the
filterable metadata as one JSON blob — a scan per filter instead of an indexed
lookup.

## The decisions worth remembering

**A sidecar file, not a table in `kompend.db`.** The issue proposed putting
the vec0 table next to the manifest row. `core/store/connection.py` recreates
the whole vault store when `SCHEMA_VERSION` mismatches, so the vector index
would die with any unrelated schema bump and force a full re-embed of the
corpus. The store DOX already keeps the embedding branch separate and
regenerable — a file replaces the directory that was there before. The
manifest row stays in `kompend.db`, and gained a `backend` column: a
zvec-era manifest must not validate a sqlite-vec build (the ids, hashes,
model and embed version could all match while the vectors are unreadable).

**Filters became parameterised SQL.** The zvec filter was a string built by
`_build_filter` with `_escape_filter_value` escaping the values. It is now
`EXISTS` subqueries over the side tables, with every value bound as a
parameter — the string-building and its `S608` waiver are gone. The canonical
semantics stay in `search/filters.py` (tags are an OR of ANDs; attributes are
an AND of `(key, value)` checks), and a test pins the SQL to those
predicates.

**Fusion moved to Python.** sqlite-vec has no server-side reranker, so the
weighted reciprocal rank fusion (k=60, weights `[vector, title, preview]`) runs
in the store. The policy is unchanged; the candidate pools are not identical
to zvec's single fused pool, so the behaviour is pinned by tests rather than
claimed.

**A user query is not FTS5 syntax.** FTS5 reads `-` as NOT and `:` as a column
filter, so `context-window` fails to parse ("no such column: window"). The
query is split into word characters and each term is quoted before it reaches
FTS5.

**Exact search, and the ANN question left open.** zvec indexed the embedding
field with HNSW; `vec0` is brute-force exact. Measured on the slowest
supported CPU (a Celeron J6413 with no AVX): ~9 ms at 10k nodes, ~111 ms at
100k. The alternatives were measured, not guessed:

| | sqlite-vec (exact) | usearch 2.26.6 (HNSW) |
|---|---|---|
| build 10k | 2.1 s | 130.4 s |
| search k=10 @10k | 9.4 ms | 10.3 ms |
| recall@10 @10k | 10/10 | 8/10 |
| build 100k | 3.2 s | 1056 s (17.6 min) |
| search k=10 @100k | 111 ms | 11.7 ms |
| recall@10 @100k | 10/10 | 3/10 |
| usearch's own exact scan @100k | — | 530 ms |

usearch works on a non-AVX CPU (it dispatches SIMD at runtime and falls back
to the scalar kernel), but on that hardware its HNSW build costs 60–330× the
insert and its query is no faster at wiki scale. At 100k it does buy latency
(11.7 ms vs 111 ms) — but its own exact scan is 5× slower than sqlite-vec's
brute force, so it would be an *accelerator over the same vectors*, never a
replacement for the store. `hnswlib` is sdist-only and needs a C++ compiler;
`SFEirc/HNSW-Vector-Search` is header-only C++ with the same burden;
`vectorlite` is Swift/Apple-only. The store therefore implements
`VectorStoreLike` and the ANN path stays open by design: a second
implementation slots in without touching the retrievers. Revisit at ~50k nodes
or when a latency budget rejects 111 ms @100k.

**The module split.** `zvec_store.py` mixed three concerns. It is now
`vector_store.py` (the field contract, the embedding providers, the
composition rules — backend-independent) and `sqlite_vec_store.py` (the
backend). The protocol boundary held: the retriever tests inject fakes and
did not change.

## The optional cross-encoder

`search.reranker.backend` selects the re-rank stage: `bi-encoder` (the
previous behaviour), `flashrank` (a cross-encoder that reads the query and
the page body together), or `none`. FlashRank is an optional extra
(`kompend[rerank]`) because it pulls onnxruntime; the import is deferred
through `lazy_loader`, so a vault that never enables the stage never pays for
it. The stage is best-effort enrichment: a missing model leaves the
candidates unchanged.

## What it cost

`zvec` was the last native vector engine in the tree. `kompend` now runs on
any CPU, the unit suite collects and runs everywhere, and the vector branch is
one SQLite file the reader can open with any SQLite client.
