# AGENTS.md — Rules for AI Assistants

You are working in the **knox** repository.

## Project

**KNOX** = Knowledge Organizer for eXtended documents.

A parser-agnostic, structure-preserving, LLM-aware knowledge base compiler.
Follows a 5-tier pipeline: Preprocessing -> Chunking -> Resolution -> Compilation -> Output.

## Conventions

- English-only for all code, docs, and commits
- Use `uv run` / `uv run -m` for all Python execution
- Conventional commits (`feat:`, `fix:`, `chore:`, `docs:`)
- All parsers implement `BaseParser`; all chunkers implement `BaseChunker`
- Online sources (URLs) go through `downloaders/` before `parsers/`
- MCP server is a thin wrapper — no business logic duplication
