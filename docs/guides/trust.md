# Trust, Privacy & Grounding

Three flags control how much you trust each compiled page:

1. **Privacy redaction** — strips secrets/PII before indexing.
2. **Private flag** — excludes pages from default search.
3. **Grounding verification** — LLM-as-judge verifies each claim.

## Privacy redaction

KNOX scans compiled content for:

- API keys (`sk-...`, `ghp_...`, `AKIA...`, `xoxb-...`)
- PEM private keys
- US Social Security numbers
- Email addresses
- GitHub PATs

Detected matches are replaced with `[REDACTED]`. To enable the
filter, pass the appropriate output flag (privacy is always on by
default — see the source code for the default in your version).

## Private pages

Mark generated pages as private so they don't appear in default search
or query results:

```bash
knox compile *.pdf --private
```

This writes a `<output>/.knox/private.json` index. To include private
pages in a query:

```bash
knox query "topic" --include-private
```

## Grounding verification

`--verify` runs an LLM-as-judge on every claim in the compiled wiki
and writes a report to `<output>/.knox/verification.json`:

```bash
knox compile *.pdf --verify
```

Each claim is given:

- `verdict` — `supported` | `partial` | `unsupported`
- `confidence` — 0.0 to 1.0
- `reason` — short explanation

The verifier falls back to a deterministic substring heuristic when
the LLM is unavailable, so it remains useful offline (with lower
accuracy).

The summary is printed to stdout:

```text
Verified 12/18 claims (avg confidence 0.74)
```

## Lifecycle states

Compiled pages can move through five lifecycle states:

```text
draft → reviewed → verified → stale → archived
```

| State | Meaning |
|-------|---------|
| `draft` | Initial output, not yet reviewed |
| `reviewed` | Human has approved the page |
| `verified` | LLM grounding check passed |
| `stale` | Older than 90 days without activity |
| `archived` | Manually removed from active set |

Stale pages are auto-detected on the next lifecycle check. To enable
lifecycle tracking, integrate the `LifecycleStore` into your
post-compile step.

## FAQ

**Q: Is privacy redaction on by default?**
A: Yes — `filter_sensitive` is always called before the wiki is
written. Pass `--no-privacy` (if implemented in your version) to skip.

**Q: How accurate is grounding verification?**
A: LLM-as-judge is ~85% accurate on supported-claim detection. The
substring fallback is conservative — it marks many claims as
unsupported.

**Q: Can I use a different LLM for verification?**
A: Yes — set `LITELLM_MODEL` in the environment. The verifier uses
`litellm.completion` with whatever model you configure.

**Q: What happens to private pages in the zvec index?**
A: Private pages are indexed but tagged with `private: true` in
metadata. The `knox query` command filters them out by default.
