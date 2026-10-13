# Query Memory & Context Synthesis

kompend includes a lightweight, native memory layer that allows the system to learn from past interactions. This feature enables long-term context retention, ensuring that the wiki evolves to better match your specific needs, terminology, and preferences over time.

## Overview

The memory layer operates on a dual-storage model for optimal performance and flexibility:

1. **Semantic Search (`zvec`)**: Past queries and answers are embedded and stored in `.kompend/memory.db`, allowing the system to retrieve highly relevant past interactions based on semantic similarity.
2. **Chronological Log**: An append-only log in the `memory_log` table of `.kompend/kompend.db` maintains a strict timeline of interactions, which is essential for periodic context synthesis.

## CLI Usage

### 1. Saving Queries During Search

When using `kompend query`, you can append the `--remember` flag. If the query is successfully synthesized by the LLM, both the query and the generated answer will be automatically saved to the memory store.

```bash
kompend query "What are the acoustic intelligibility requirements?" --remember
```

### 2. Manual Memory Management

You can interact with the memory store directly using the `kompend memory` command group.

**Add an entry manually:**

```bash
kompend memory add "How do I configure Ollama?" "Set the LLM model to ollama/..." --feedback "Prefer concise, step-by-step answers"
```

**Search past interactions:**

```bash
kompend memory search "Ollama configuration" --top-k 3
```

**Synthesize context:**
Periodically, you can instruct kompend to review recent memory entries and distill them into a living system context file. This file (`sources/_memory/system_context.md`) is automatically picked up during compilation and querying to guide the LLM's behavior.

```bash
kompend memory synthesize --limit 20
```

*Note: In the current implementation, synthesis generates a thematic summary of recent queries. Future iterations will use an LLM to generate deeper behavioral insights (e.g., "The user prefers bullet points and focuses heavily on acoustic metrics").*

## MCP Server Integration

For AI agents, the memory layer is fully exposed via the kompend MCP server, allowing autonomous agents to build and leverage long-term context without shell access.

Available MCP tools:

- **`memory_add`**: Save a query/answer/feedback triple.
- **`memory_search`**: Retrieve semantically similar past interactions to inform the current agent context.
- **`memory_synthesize`**: Trigger the periodic distillation of recent interactions into the `system_context.md` file.

### Example Agent Workflow

1. Agent answers a user query about the wiki.
2. Agent calls `memory_add` with the query, its answer, and any implicit/explicit user feedback.
3. On subsequent, related queries, the agent calls `memory_search` to see how similar questions were handled previously.
4. Periodically, the agent (or a background cron job) calls `memory_synthesize` to update the global system context.

## Enhanced RAG Pipeline

kompend includes optional RAG enhancement techniques that improve retrieval quality.
Each is a plug-in step configurable via `kompend.toml` under `[search]`:

| Feature | Config | Default | Description |
|---------|--------|---------|-------------|
| **Multi-Query Expansion** | `search.expansion.enabled` | `false` | N query variants + RRF merge (RAG-Fusion) |
| **CRAG** | `search.crag.enabled` | `false` | LLM relevance grading, drop noise |
| **Neural Re-ranker** | `search.reranker.enabled` | `false` | Cross-encoder post-fusion ranking |
| **HyDE** | `search.hyde.enabled` | `false` | Hypothetical answer embeddings |
| **Contextual Retrieval** | `search.contextual.enabled` | `false` | Situating context for chunk embeddings |

All features default to **disabled**. Enable only what you need.
See [RAG Pipeline](../concepts/rag-pipeline.md) for full documentation.

## Configuration

The memory layer respects the global wiki directory context. By default, it looks for the `.kompend/` directory in the current working directory (or the directory specified by `--wiki-dir` in the CLI).

No additional configuration is required to enable the memory layer, though ensuring your `compile.embedding.model` is configured correctly will provide the best semantic search results.
