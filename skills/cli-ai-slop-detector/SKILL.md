---
name: cli-ai-slop-detector
description: AI-SLOP Detector is an evidence-based static analyzer for AI-generated code. Use when the user mentions "ai-slop-detector", "slop-detector", "slop detector", "AI slop", "deficit score", "self-calibration", "LEDA", "phantom imports", "disconnected pipelines", or wants to analyze code for 27 adversarial patterns across 5 categories (Placeholder, Structural, Cross-Language, Python Advanced, Phantom) with a 4D scoring model (LDR/ICR/DDC/Purity).
---

# AI-SLOP Detector

AI-SLOP Detector is an evidence-based static analyzer purpose-built to catch the specific class of defects that AI code generation reliably introduces — before they reach production.

Not a style linter. A structural-risk scanner for AI-assisted code.

## Install

```shell
pip install "ai-slop-detector>=3.7.3"
pip install "ai-slop-detector[all]"    # all extras including [js], [go]
pip install "ai-slop-detector[js]"      # JS/TS tree-sitter analysis
pip install "ai-slop-detector[go]"      # Go regex-based analysis
```

Or via uvx (no install required):

```shell
uvx ai-slop-detector mycode.py
```

## Quick Start

```shell
slop-detector --init                       # bootstrap .slopconfig.yaml + .gitignore
slop-detector mycode.py                    # single file
slop-detector --project ./src             # entire project
slop-detector mycode.py --json            # machine-readable output
slop-detector --project . --ci-mode hard --ci-report  # CI gate
```

## What It Detects

### 27 Patterns Across 5 Categories

| Category | Patterns |
|----------|----------|
| **Placeholder** | `empty_except`, `not_implemented`, `pass_placeholder`, `ellipsis_placeholder`, `return_none_placeholder`, `return_constant_stub`, `todo_comment`, `fixme_comment`, `hack_comment`, `xxx_comment`, `interface_only_class` |
| **Structural** | `bare_except`, `mutable_default_arg`, `star_import`, `global_statement` |
| **Cross-Language** | `javascript_array_push`, `java_equals_method`, `ruby_each`, `go_print`, `csharp_length`, `php_strlen` |
| **Python Advanced** | `god_function`, `dead_code`, `deep_nesting`, `lint_escape`, `function_clone_cluster`, `placeholder_variable_naming` |
| **Phantom** | `phantom_import` |

### The 4D Scoring Model

Each file is measured across four independent axes combined via weighted geometric mean:

| Metric | What It Measures |
|--------|-----------------|
| **LDR** (Logic Density Ratio) | `logic_lines / total_lines` — code vs. whitespace/comments |
| **ICR** (Inflation Check) | `jargon_density × complexity_modifier` — buzzword weight |
| **DDC** (Dependency Check) | `used_imports / total_imports` — import utilization |
| **Purity** | `exp(-0.5 × n_critical_patterns)` — AND-gate on critical pattern severity |

### Score Interpretation

| Score | Status | Action |
|-------|--------|--------|
| ≥ 70 | `CRITICAL_DEFICIT` | Immediate deep-dive — full patch guidance |
| 50–70 | `INFLATED_SIGNAL` | Full deep-dive — action required before merge |
| 30–50 | `SUSPICIOUS` | Run `/slop-file` on top 2 files first; confirm before escalating |
| < 30 | `CLEAN` | Skip Phase 2 — report clean, propose gate |

Default weights: `ldr=0.40 · inflation=0.30 · ddc=0.20 · purity=0.10`

## Self-Calibration

AI-SLOP Detector learns your codebase:

```shell
slop-detector . --self-calibrate               # see what history recommends
slop-detector . --self-calibrate --apply-calibration  # write to .slopconfig.yaml
```

- **Project-scoped** — `history.db` tags every record with `project_id` (sha256 of cwd)
- **Domain-aware** — 8 built-in profiles: `general`, `scientific/ml`, `scientific/numerical`, `web/api`, `library/sdk`, `cli/tool`, `bio`, `finance`
- **4D grid search** (±0.15 domain-anchored) optimizes all weight dimensions simultaneously
- Only applies when confidence gap between top two candidates exceeds **0.10**

## CI/CD Integration

### pre-commit

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/flamehaven01/AI-SLOP-Detector
    rev: v3.7.3
    hooks:
      - id: slop-detector          # hard gate — fails on CRITICAL_DEFICIT >= 70
```

### GitHub Actions

```yaml
# .github/workflows/quality-gate.yml
- name: AI-SLOP Gate
  run: |
    pip install "ai-slop-detector>=3.7.3"
    slop-detector --project . --ci-mode hard --ci-report
```

### Enforcement Modes

| Mode | Behavior |
|------|----------|
| `--ci-mode soft` | Informational, never fails build |
| `--ci-mode hard` | Fails: deficit_score >= 70, critical_patterns >= 3, inflation >= 1.5, ddc < 0.5 |
| `--ci-mode quarantine` | Escalates repeat offenders after 3 violations |

## Configuration

```yaml
# .slopconfig.yaml
weights:
  ldr: 0.40
  inflation: 0.30
  ddc: 0.20
  purity: 0.10

patterns:
  god_function:
    domain_overrides:
      - function_pattern: "check_node"   # AST walker — complex by design
        complexity_threshold: 30
        lines_threshold: 200

ignore:
  - "tests/**"
  - "**/__init__.py"
```

Initialize with domain auto-detection:

```shell
slop-detector --init                   # auto-detect domain, generate .slopconfig.yaml
slop-detector --init --domain web/api  # explicit domain override
```

## History Tracking

```shell
slop-detector mycode.py --show-history   # per-file trend
slop-detector --history-trends           # 7-day project aggregate
slop-detector --export-history data.jsonl
```

Every run auto-recorded to `~/.slop-detector/history.db`.

## CLI Commands Reference

| Command | Description |
|---------|-------------|
| `slop-detector <file>` | Analyze single file |
| `slop-detector --project <dir>` | Analyze entire project |
| `slop-detector --init` | Bootstrap config with domain auto-detection |
| `slop-detector --json` | Machine-readable JSON output |
| `slop-detector --ci-mode hard` | CI gate with hard fail |
| `slop-detector --self-calibrate` | Run calibration on history |
| `slop-detector --apply-calibration` | Write calibration to config |
| `slop-detector --show-history` | Per-file deficit trend |
| `slop-detector --history-trends` | 7-day project aggregate |
| `slop-detector --export-history <file>` | Export history to JSONL |
| `slop-detector --ignore "<pattern>"` | Exclude files matching pattern |
| `slop-detector --domain <profile>` | Override domain (8 profiles) |

## Key Principles

- **Structural-risk scanner**, not a style linter
- Catches **functionally empty, disconnected, or misleading** code
- 4D weighted geometric mean — a near-zero in any dimension pulls overall score down
- Self-calibrating via LEDA protocol — learns from your git commit behavior
- History database (`~/.slop-detector/history.db`) accumulates across all projects