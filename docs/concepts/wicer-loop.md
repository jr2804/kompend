---
icon: lucide/refresh-cw
---

# WiCER Loop

WiCER (Wiki Iterative Compile-Evaluate-Refine) is KNOX's self-correcting
compilation loop. Inspired by [arXiv:2605.07068v1](https://arxiv.org/pdf/2605.07068v1),
it recovers quality lost during blind compilation through targeted diagnosis
and refinement.

## The Problem: Blind Over-Compression

When an LLM compiles a large document into a wiki page in one pass, it
over-compresses by 2–3× beyond the target length. This is not gradual quality
loss — it is catastrophic. Key facts, figures, and nuances are silently dropped.

## The Solution: CEGAR-Style Loop

WiCER follows a Counterexample-Guided Abstraction Refinement (CEGAR) pattern:

``` mermaid
stateDiagram-v2
    [*] --> BlindCompile: Raw chunks
    BlindCompile --> Evaluate: Compiled wiki W₀
    Evaluate --> Diagnose: Score < threshold
    Evaluate --> [*]: Score ≥ threshold
    Diagnose --> Refine: Critical facts extracted
    Refine --> BlindCompile: Pinned facts as context
    Refine --> [*]: Max iterations reached
```

### Step A: Blind Compilation

The compiler produces an initial wiki `W₀` from resolved chunks in one pass.
This is fast but over-compresses — typically 2–3× shorter than the ideal.

### Step B: Evaluation

Probe Q&A pairs are generated per source document. The LLM-as-judge scores
answers from `W₀` on a 1–5 scale. Score-1 answers indicate catastrophic loss
— facts that exist in the source but are absent from the wiki.

### Step C: Diagnosis

For each score-1 failure, the system traces the lost fact back to its source
location. Critical facts are extracted as "preserve" items with their source
context. Generic pinning (just "include more detail") scores ~0.16; targeted
pinning (specific fact + location) scores ~0.95.

### Step D: Refinement

The compiler re-runs with pinned facts as "must-include" context. The loop
repeats until convergence (all probes score ≥ 3) or max iterations (default: 3).

## Empirical Results

| Metric | Value |
|--------|-------|
| Blind over-compression | 2–3× beyond target |
| Quality recovery after 1 iteration | ~70% |
| Quality recovery after 2 iterations | ~80% |
| Generic pinning effectiveness | ~0.16 |
| Targeted pinning effectiveness | ~0.95 |
| Max iterations (default) | 3 |

## Implementation

The WiCER compiler lives in `src/knox/compilation/wicer.py`:

| Component | Method | Purpose |
|-----------|--------|---------|
| `WiCERCompiler` | `compile()` | Main loop orchestrator |
| `_blind_compile()` | Initial one-shot compilation |  |
| `_evaluate()` | Probe Q&A generation + scoring |  |
| `_diagnose()` | Trace failures to source facts |  |
| `_refine()` | Re-compile with pinned facts |  |

The `GraphCompiler` in `src/knox/compilation/graph.py` provides the two-phase
(extract-then-generate) alternative without the iterative loop.

## Configuration

```toml
[compile.wicer]
max_iterations = 3
quality_threshold = 3.0  # minimum average probe score
probes_per_source = 3     # Q&A pairs per source document
```

## Further Reading

- [WiCER paper (arXiv:2605.07068v1)](https://arxiv.org/pdf/2605.07068v1)
- [Pipeline Tiers](pipeline-tiers.md) — Full tier reference
- [Architecture](../architecture/index.md) — 5-tier design
- [Decision Log](../architecture/decisions.md) — Why WiCER was chosen
