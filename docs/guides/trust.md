# Trust, Privacy & Grounding

Three flags control how much you trust each compiled page:

1. **Privacy redaction** — strips secrets/PII before indexing.
2. **Private flag** — excludes pages from default search.
3. **Grounding verification** — LLM-as-judge verifies each claim.

## Privacy redaction

KOMPEND scans compiled content for:

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
kompend compile *.pdf --private
```

This records a private-page index in `.kompend/kompend.db`. To include private
pages in a query:

```bash
kompend query "topic" --include-private
```

## Grounding verification

`--verify` runs an LLM-as-judge on every claim in the compiled wiki
and records a report in the verification table of `.kompend/kompend.db`. It is **on by default**;
pass `--no-verify` to skip the cost:

```bash
kompend compile *.pdf            # verify runs by default
kompend compile *.pdf --no-verify
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

## Memory lifecycle

Compiled pages are tracked through a memory lifecycle that models
confidence decay, tier promotion, and supersession. The store lives
in the `memory_states` table of `base_dir/.kompend/kompend.db` and is updated
automatically on each compile when `compile.memory_enabled` is `True` (the
default).

### Tiers

```text
WORKING → EPISODIC → SEMANTIC → PROCEDURAL
```

| Tier | Promotion threshold | Meaning |
|------|---------------------|---------|
| `WORKING` | initial | Freshly compiled, untested |
| `EPISODIC` | 2 reviews | Survived multiple compiles |
| `SEMANTIC` | 5 reviews | Stable, high-confidence knowledge |
| `PROCEDURAL` | 10 reviews | Core, rarely-changing reference |

### Confidence decay

Each page has a confidence score (0.5 → 1.0, asymptotic). Retention
decays exponentially:

```text
retention = confidence × e^(−elapsed_days / half_life)
```

The half-life defaults to 90 days (`compile.memory_half_life_days`).
Pages whose retention falls below the decay threshold
(`compile.memory_decay_threshold`, default 0.3) are flagged as stale
by the `stale-page` lint rule.

### Supersession

When a page is recompiled under a different node ID (e.g., heading
restructuring), the old page is marked `superseded_by` the new one.
Superseded pages are excluded from stale detection and reported in
`wiki.metadata["memory"]["superseded"]`.

## FAQ

**Q: Is privacy redaction on by default?**
A: Yes — `filter_sensitive` is always called before the wiki is
written. Pass `--no-privacy` (if implemented in your version) to skip.

**Q: How accurate is grounding verification?**
A: LLM-as-judge is ~85% accurate on supported-claim detection. The
substring fallback is conservative — it marks many claims as
unsupported.

**Q: Can I use a different LLM for verification?**
A: Yes — set `KOMP_LLM_DEFAULT_MODEL`, or give the `verify` role its own model
in `[compile.llm.role_models]`. The verifier runs through KOMPEND's configured
pydantic-ai agent like every other LLM step.

**Q: What happens to private pages in the vector index?**
A: Private pages are indexed but tagged with `private: true` in
metadata. The `kompend query` command filters them out by default.
