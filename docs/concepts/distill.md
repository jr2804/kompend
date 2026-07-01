# Distill

Captured notes and conversation fragments accumulate in `sources/_captured/`. The `distill` command synthesizes these raw fragments into coherent wiki source pages that are then compiled into the knowledge base.

## Flow

```mermaid
graph LR
    C[Capture / Learn] --> CAP[_captured/]
    CAP --> D[Distill]
    D --> DIS[_distilled/]
    DIS --> CO[Compile]
    CO --> W[Wiki]
```

## Commands

```bash
# Capture: LLM extracts decisions, discoveries, corrections from text
knox capture "We decided to use QKV projections with RoPE"

# Learn: store a verbatim fact
knox learn "The transformer uses d_model=512" --tag architecture

# Distill: synthesize all captures into wiki source pages
knox distill
```

## How it works

1. `distill` reads all `.md` files from `sources/_captured/`.
2. Captures are grouped by their first tag.
3. Each group is synthesized into a markdown source file via LLM (or a deterministic fallback if no LLM is available).
4. Output is written to `sources/_distilled/` with frontmatter tracking the source captures.
5. A subsequent `knox compile` integrates the distilled pages into the wiki.

## Output format

Each distilled page includes:

```yaml
---
type: distilled
title: "Attention Mechanisms"
synthesized_from:
  - capture-20260613T120000Z-00-attention-decision.md
  - learn-20260613T130000Z-attention-fact.md
tags:
  - attention
  - architecture
---
```

## See also

- [CLI Guide](../guides/cli.md) — capture, learn, distill commands
- [Wiki Structure](../guides/wiki-structure.md) — output directory layout
