# Lint Reference

KOMPEND lint checks compiled wikis for structural and semantic quality issues.

## Quick Start

```bash
# Run all structural + LLM-powered rules (LLM on by default)
kompend lint all

# Structural rules only (no LLM calls)
kompend lint check            # deterministic rules only

# Auto-fix supported issues
kompend lint all --fix

# Run only specific rules
kompend lint check --rule broken-wikilinks --rule orphan-page

# Machine-readable output for CI
kompend lint all --json
```

LLM-powered rules use `compile.llm.default_model`; with no model configured
they are skipped with a note (structural rules still run).

## Commands

`kompend lint` is a command group. Each lint phase is its own sub-command;
`kompend lint all` runs the whole pipeline in one flow with a single weighted
progress bar (deterministic steps weight 1/page, LLM steps 100/page).

| Sub-command | Description | LLM | Fix |
|------------|-------------|-----|-----|
| `kompend lint all` | Full pipeline: check → format → semantic → gaps → remediate, one progress bar | both | `--fix/--no-fix` (default on) |
| `kompend lint check` | Deterministic structural rules (broken-wikilinks, aliases, frontmatter, orphans, duplicates, stale, orphan-attachments, ...) | no | `--fix/--no-fix` (default on) |
| `kompend lint format` | rumdl formatting pass | no | `--check` for CI (exit 1 when changes needed) |
| `kompend lint semantic` | LLM rules: factual-consistency, clarity-score, coverage-gap, contradiction-detection | yes | report only |
| `kompend lint gaps` | Gap analysis + semantic source-coverage verification + weighted priorities | embeddings | `--fix` runs remediation |
| `kompend lint remediate` | LLM-generate missing pages from cached extractions | yes | writes pages (`--dry-run` to preview) |
| `kompend lint history` | Query past lint runs (scores, trends, details) | — | — |
| `kompend lint prune` | Remove old history entries | — | — |

### `kompend lint check`

```bash
kompend lint check                  # structural rules, fix what's possible
kompend lint check --no-fix         # report only
kompend lint check --rule broken-wikilinks --rule orphan-page
```

### `kompend lint all`

```bash
kompend lint all                    # check + format + semantic + gaps + remediate
kompend lint all --no-fix           # everything, but write nothing
```

LLM steps (`semantic`, `remediate`) are skipped gracefully with a note when
no model is configured. The model is `compile.lint.model`, falling back to
`compile.llm.default_model`.

## Lint during compile

`kompend compile` runs lint automatically after the wiki is built. Two
independent flags control what gets run (both default to on):

| Flag | Env var | Effect |
|------|---------|--------|
| `--lint / --no-lint` | `KOMP_COMPILE_LINT` | Deterministic structural lint only |
| `--lint-all / --no-lint-all` | `KOMP_COMPILE_LINT_ALL` | Full pipeline (deterministic + LLM + remediate); implies `--lint` |

`--no-lint` and `--no-lint-all` are identical — both disable any linting.
Fixes are always applied when lint runs during compile (no `--no-fix`
option; compile is meant to leave the wiki in a clean state).
Disable per-run with `--no-lint-all`, or set `KOMP_COMPILE_LINT_ALL=0`
in the environment:

```bash
kompend compile ./sources --no-lint-all        # no lint at all
KOMP_COMPILE_LINT=0 kompend compile ./sources  # structural lint only (skips LLM steps)
```

### `kompend lint history`

```text
kompend lint history [OPTIONS]

Show past lint runs from .kompend/lint_history.jsonl.

Options:
  --last INT, -n     Show last N entries (default 10)
  --trend            Show score trend sparkline
  --json             Output in JSON format
```

Examples:

```bash
# Show last 10 lint runs
kompend lint history

# Show only the most recent run
kompend lint history -n 1

# Show score trend over time (Unicode sparkline)
kompend lint history --trend

# Machine-readable output for dashboards
kompend lint history --json
```

### `kompend lint gaps`

Analyze the latest lint run for knowledge gaps and recommend source
material:

```bash
kompend lint gaps
```

Gap recommendations cover missing topics, empty pages, isolated topics,
coverage gaps, and unresolved contradictions. `gaps` only recommends —
it never modifies the wiki.

### `kompend lint remediate`

Generate missing concept pages for coverage-gap findings:

```bash
kompend lint remediate           # generate pages (needs an LLM configured)
kompend lint remediate --dry-run # preview without writing
```

Coverage-gap findings are matched against cached phase-1 extractions;
matched concepts get concept pages generated from that source material and
written to the wiki output directory. Unmatched concepts are skipped — run
`kompend lint gaps` for recommendations on those.

### `kompend lint prune`

```text
kompend lint prune [OPTIONS]

Remove old entries from .kompend/lint_history.jsonl.
At least one of --keep-last or --max-age-days is required.

Options:
  --keep-last INT          Keep only the most recent N entries
  --max-age-days INT       Remove entries older than N days
```

Examples:

```bash
# Keep only the last 50 runs
kompend lint prune --keep-last 50

# Remove entries older than 90 days
kompend lint prune --max-age-days 90

# Combined: keep last 50 AND younger than 90 days
kompend lint prune --keep-last 50 --max-age-days 90
```

### `kompend compile --lint` (see [the table above](#lint-during-compile) for the full flag/env-var matrix)

The first section above is the canonical reference. This block shows the
short flags on the `kompend compile` command:

```bash
kompend compile ./sources                # --lint-all on (default): full pipeline incl. LLM
kompend compile ./sources --lint         # deterministic structural lint only
kompend compile ./sources --no-lint-all  # disable any linting
KOMP_COMPILE_LINT=0 kompend compile ./sources  # deterministic only
KOMP_COMPILE_LINT_ALL=0 kompend compile ./sources  # disable any linting via env
```

## Structural Rules (11)

These rules are deterministic. Nine run entirely in-process; `rumdl` shells
out to the bundled rumdl binary.

| Rule | Severity | Auto-fix | What it checks |
|------|----------|----------|---------------|
| `image-embed` | WARNING | ✅ | Image links written as `[[img.png]]` instead of `![[img.png]]` |
| `broken-wikilinks` | ERROR | ✅ | `[[wikilinks]]` pointing to non-existent pages (image embeds excluded) |
| `missing-frontmatter` | WARNING | ✅ | Pages missing required frontmatter fields (`title`) |
| `orphan-page` | WARNING | ❌ | Pages with no incoming `[[wikilinks]]` from other pages |
| `duplicate-title` | ERROR | ❌ | Two or more pages with the same or similar title (case-insensitive) |
| `empty-page` | WARNING | ❌ | Pages with no body content |
| `heading-skip` | WARNING | ❌ | Heading level jumps (e.g. h1 → h3 without h2) |
| `stale-page` | WARNING | ❌ | Pages whose source files changed since last compile |
| `source-not-found` | ERROR | ❌ | Pages referencing source files that don't exist on disk |
| `orphan-attachments` | WARNING | ❌ | Attachment files whose slug matches no source (unrecoverable by compile-time auto-adoption) |
| `unreferenced-attachments` | WARNING | ⚠️ destructive | Regenerable media in a live source dir that no page links (stale copies left by earlier writes; delete with `--fix-destructive`) |
| `rumdl` | WARNING | ✅ | Markdown style issues on all vault folders except `sources/` (MD024 duplicate headings, MD013 line length, …) |

### Auto-fix behavior

When `--fix` is passed:

- **image-embed**: Rewrites `[[img.png]]` → `![[img.png]]`
- **broken-wikilinks**: Removes `[[...]]` brackets, keeping the link text
- **missing-frontmatter**: Adds `title` from the node title
- **rumdl**: Applies `rumdl check --fix` to the flagged files (self-written wiki notes get reformatted; `sources/` is never touched)
- Other rules: Cannot be auto-fixed (require manual editing or re-compilation)

The `rumdl` rule checks every `*.md` file under the vault root — wiki output,
index pages, and self-written notes — excluding `sources/`, `.kompend/`, and
`attachments/`. Rule tuning comes from `compile.output.rumdl_config` in
`kompend.toml` (same key as render-time formatting); when unset, rumdl
auto-discovers a `.rumdl.toml` in the vault. The rule reports `SKIP` when
rumdl is not installed.

## LLM-Powered Rules (4)

These rules call an LLM and are on by default; use `--no-llm` to skip them.

| Rule | What it checks |
|------|---------------|
| `factual-consistency` | Claims in a page contradict source material |
| `clarity-score` | Readability, coherence, conciseness (1-10 score) |
| `coverage-gap` | Important concepts from sources missing in the wiki |
| `contradiction-detection` | Pairs of wiki pages that make contradictory claims, with confidence and severity |

LLM rules require a configured LLM. If no LLM is available, they report `SKIP`.

## Health Score

The health score is computed as:

```text
health_score = 100 × (PASS results) / (total results)
```

A perfect wiki scores 100%. `kompend lint all` reports the score but does **not**
gate on it — the exit code is 0 unless the run itself fails. To fail CI on a
low score, parse the JSON output:

```bash
kompend lint all --json > lint-report.json
python -c "import json,sys; sys.exit(0 if json.load(open('lint-report.json'))['health_score']>=80 else 1)"
```

## JSON Output

```bash
kompend lint all --json
```

Returns:

```json
{
  "health_score": 87.5,
  "total_checks": 8,
  "failures": 1,
  "passes": 7,
  "skips": 0,
  "results": [
    {
      "rule": "broken-wikilinks",
      "outcome": "fail",
      "severity": "error",
      "message": "Broken wikilink [[Missing]] in page 'Introduction'",
      "page_id": "node_0",
      "fix_applied": false,
      "details": {"target": "Missing"}
    }
  ]
}
```

## History

Every `kompend lint all` (or any lint sub-command) appends a structured entry to `.kompend/lint_history.jsonl`. Each entry includes:

| Field | Content |
|-------|---------|
| `timestamp` | ISO 8601 UTC |
| `duration_seconds` | Wall-clock time of the lint run |
| `options` | Active rules, `enable_llm`, `fix` |
| `llm` | Model name + token usage (or `null` for structural-only) |
| `config_snapshot` | Relevant compile settings (compiler, chunk size, model, fallback chain) |
| `wiki_meta` | Node count |
| `report` | Full `LintReport` (health score, per-rule results) |

Query with `kompend lint history` (see above). Prune old entries with `kompend lint prune`.

## CI Integration

### GitHub Actions

```yaml
- name: Lint compiled wiki
  run: |
    kompend compile ./sources
    kompend lint all --json > lint-report.json
    python -c "import json,sys; sys.exit(0 if json.load(open('lint-report.json'))['health_score']>=80 else 1)"
```

### Pre-commit

```yaml
- repo: local
  hooks:
    - id: kompend-lint
      name: kompend lint
      entry: kompend lint all --json
      language: system
      pass_filenames: false
```

The hook runs lint and writes JSON but, like `kompend lint all` itself, exits 0
unless the run fails. To make a pre-commit hook gate on the score, wrap the
JSON parse (see [Health Score](#health-score)) in a small script and point
`entry` at it.

## FAQ

### How do I add a custom lint rule?

Create a class that satisfies the `LintRule` protocol:

```python
from kompend.lint.base import LintResult, LintRuleOutcome


class MyCustomRule:
    name = "my-custom-rule"
    description = "What this rule checks"

    def check(self, wiki, *, sources_dir=None):
        # Return list of LintResult
        return [LintResult(rule=self.name, outcome=LintRuleOutcome.PASS)]

    def fix(self, wiki, results):
        return results  # or implement auto-fix
```

### Why does lint say I have broken wikilinks?

A `[[wikilink]]` points to a page title that doesn't exist in the compiled wiki. This can happen when:

- The linked page was not compiled (source file missing or parse failure)
- The link text has a typo or different capitalization
- The page was removed but links weren't updated

### What counts as a health score violation?

Any lint result with `outcome: "fail"` reduces the health score. The score is the percentage of passing checks out of total checks run.

### Why are some rules skipped?

Rules that require a sources directory (`stale-page`, `source-not-found`) will skip if the sources directory doesn't exist or is empty. LLM rules skip when no LLM client is configured.
