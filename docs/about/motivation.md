# Motivation

## Why another compiled LLM-wiki?

Karpathy's [compiled LLM-wiki concept](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f)
spawned dozens of implementations. Almost all of them fall into one of two
camps — and almost none of them look sideways to see what the other is doing.

### Camp 1: local-or-bust

Privacy-first, fully offline, often Raspberry-Pi-compatible. Noble goals —
but the model fits in a thimble, hallucinates freely, and the project's idea
of "scaling up" is buying a better SD card. You are gently reminded that
*real* knowledge management doesn't need a frontier LLM, while you watch it
confuse two authors with the same surname for the third time.

### Camp 2: frontier-or-nothing

Gemini Ultra, Claude Max, Codex Pro — plus the corresponding agent framework,
editor, and deployment platform, thank you very much. The results are
genuinely impressive, provided you have the API budget of a small country
and are willing to abandon every tool you already own. There is exactly one
way to use this project, and it involves their MCP server, their CLI, their
IDE extension, and their preferred cloud.

### The blind spots

Both camps are technically impressive. Neither is looking left or right —
not out of malice, they're just genuinely unaware that anyone might operate
outside their chosen stack. They aren't designed that way.

**Blind spot № 1 — provider and tool agnosticism.**

KNOX is not inherently *better* than any of these projects — it just refuses
to pick a side. The same pipeline that runs on a laptop with an 8B local
model can be pointed at a frontier LLM with a million tokens of context
window, and it will make better use of whichever you give it. The compilation
pipeline is provider-agnostic, tool-agnostic, and performance-adaptive:
smaller models get crisper, more constrained tasks; larger models get the
freedom to shine. You keep your editor, your CLI, your MCP client, your CI
— KNOX is a library and a command, not a religion.

**Blind spot № 2 — document parsing quality.**

Both camps treat document parsing as a solved problem. Throw a PDF at an LLM
and hope it extracts the tables, right? For blog posts and news articles,
that works well enough. For technical reports, standards, specifications,
and scientific papers, the "markdown noise" from a botched figure, equation,
or table conversion drowns out the actual content. If your downstream
compilation is built on garbage extraction, no LLM — local or frontier —
can fix it.

That is why KNOX uses structure-preserving parsing
([parsecraft](https://pypi.org/project/parsecraft/), which routes each page
across docling, LiteParse, pandoc and native backends) as its foundation:
documents are parsed into a rich, typed intermediate representation that
keeps headings, tables, figures, equations, and code blocks intact before
anything is chunked, resolved, or compiled. Chunking never splits atomic
elements, and the compiler sees the actual structure — not the noise.

**Blind spot № 3 — multi-strategy, not single-path.**

If you're not picking sides, you shouldn't be picking single-path solutions
either. Most tools plant a flag on *one* technique — usually vanilla RAG —
and call it a day. KNOX applies and combines multiple approaches across
every tier:

- **Hybrid chunking** with atomic-element preservation — not naive paragraph
  splitting.
- **Multi-turn entity resolution** with typed relation extraction — not
  one-shot LLM prompts.
- **Graph-based and iterative compilation** — not stuff-and-pray.
- **Hybrid search fusion** blending FTS, vector, token-overlap, and raw
  sources via reciprocal rank fusion.
- **Confidence-ranked memory lifecycle** promoting facts from working to
  procedural memory as they prove themselves.

None of these is a silver bullet. Together, they cover cases that any single
approach misses.

---

*KNOX doesn't claim to be better. It claims to be more useful across a wider
range of realities — because no two users have the same hardware, the same
budget, the same documents, or the same tolerance for being told they're
holding it wrong.*
