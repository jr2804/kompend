from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from knox.core.models import CompiledWiki, WikiNode
from knox.core.template_loader import TemplateLoader
from knox.llm.client import LLMClient
from knox.search.index import build_chunk_index, search_fts

logger = logging.getLogger(__name__)

_TOKEN_SPLIT_RE: re.Pattern[str] = re.compile(r"[^a-zA-Z0-9_]+")


@dataclass
class SearchResult:
    article_id: str
    title: str
    snippet: str
    score: float
    source_file: str | None = None


@dataclass
class SynthesisResult:
    answer: str
    citations: list[dict[str, Any]] = field(default_factory=list)


class TokenOverlapRetriever:
    def __init__(self, wiki: CompiledWiki, index_path: Path | None = None) -> None:
        self._wiki: CompiledWiki = wiki
        self._template_loader: TemplateLoader = TemplateLoader()
        self._index_path: Path | None = index_path
        if index_path is not None and not index_path.exists():
            build_chunk_index(wiki, index_path)

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        if self._index_path and self._index_path.exists():
            fts_results = search_fts(query, self._index_path, top_k=top_k)
            if fts_results:
                results: list[SearchResult] = []
                for r in fts_results:
                    results.append(
                        SearchResult(
                            article_id=r["article_id"],
                            title=r["title"],
                            snippet=r["content"][:300],
                            score=r["score"],
                            source_file=r.get("source_file"),
                        )
                    )
                return results

        query_tokens: set[str] = set(_tokenize(query))
        if not query_tokens:
            return []

        scored: list[tuple[float, WikiNode]] = []
        for node in self._wiki.nodes:
            content: str = node.content or node.title
            node_tokens: set[str] = set(_tokenize(content))
            if not node_tokens:
                continue
            overlap: int = len(query_tokens & node_tokens)
            score: float = overlap / len(query_tokens)
            scored.append((score, node))

        scored.sort(key=lambda pair: pair[0], reverse=True)

        results: list[SearchResult] = []
        for score, node in scored[:top_k]:
            snippet: str = _extract_snippet(node.content, query_tokens) if node.content else node.title
            source_file: str | None = None
            if node.metadata and "source" in node.metadata:
                source_file = str(node.metadata["source"])
            results.append(
                SearchResult(
                    article_id=node.id,
                    title=node.title,
                    snippet=snippet,
                    score=score,
                    source_file=source_file,
                )
            )
        return results

    def expand_query(self, query: str, llm_client: LLMClient | None = None) -> list[str]:
        if llm_client is None:
            return [query]

        try:
            system_prompt = self._template_loader.render("search/expand_query.md.j2", query=query)
            result = llm_client.complete(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query},
                ],
                role="search.expand",
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            if result.json_data is None:
                return [query]

            expansions: list[str] = [query]
            for key in ("keyword_rewrites", "semantic_rewrites"):
                items = result.json_data.get(key, [])
                if isinstance(items, list):
                    expansions.extend(str(e) for e in items if isinstance(e, str))
            return expansions[:7]
        except Exception as exc:
            logger.warning("Query expansion failed: %s", exc)
            return [query]
        if not query_tokens:
            return []

        scored: list[tuple[float, WikiNode]] = []
        for node in self._wiki.nodes:
            content: str = node.content or node.title
            node_tokens: set[str] = set(_tokenize(content))
            if not node_tokens:
                continue
            overlap: int = len(query_tokens & node_tokens)
            score: float = overlap / len(query_tokens)
            scored.append((score, node))

        scored.sort(key=lambda pair: pair[0], reverse=True)

        results: list[SearchResult] = []
        for score, node in scored[:top_k]:
            snippet: str = _extract_snippet(node.content, query_tokens) if node.content else node.title
            source_file: str | None = None
            if node.metadata and "source" in node.metadata:
                source_file = str(node.metadata["source"])
            results.append(
                SearchResult(
                    article_id=node.id,
                    title=node.title,
                    snippet=snippet,
                    score=score,
                    source_file=source_file,
                )
            )
        return results

    def synthesize(
        self,
        query: str,
        results: list[SearchResult],
        llm_client: LLMClient | None = None,
    ) -> SynthesisResult:
        if llm_client is None:
            citations: list[dict[str, Any]] = []
            for r in results:
                citations.append(
                    {
                        "article": r.title,
                        "source": r.source_file,
                        "snippet": r.snippet,
                        "score": round(r.score, 3),
                    }
                )
            return SynthesisResult(answer="", citations=citations)

        context_lines: list[str] = []
        for i, r in enumerate(results, 1):
            context_lines.append(f"[{i}] {r.title}")
            if r.source_file:
                context_lines.append(f"    Source: {r.source_file}")
            context_lines.append(f"    {r.snippet}")
            context_lines.append("")

        context_text: str = "\n".join(context_lines)
        system_prompt: str = self._template_loader.render(
            "query/synthesize_system.md.j2",
            query=query,
            context=context_text,
        )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]

        result = llm_client.complete(messages, role="search")
        answer: str = result.content

        citations_out: list[dict[str, Any]] = []
        for r in results:
            citations_out.append(
                {
                    "article": r.title,
                    "source": r.source_file,
                    "snippet": r.snippet,
                    "score": round(r.score, 3),
                }
            )

        return SynthesisResult(answer=answer, citations=citations_out)

    def search_raw(self, query: str, sources_dir: Path, top_k: int = 5) -> list[SearchResult]:  # noqa: PLR6301
        query_tokens: set[str] = set(_tokenize(query))
        if not query_tokens or not sources_dir.is_dir():
            return []

        scored: list[tuple[float, Path, str]] = []
        for md_file in sources_dir.rglob("*.md"):
            try:
                content: str = md_file.read_text(encoding="utf-8", errors="replace")
            except Exception as exc:
                logger.warning("Skipping unreadable source %s: %s", md_file, exc)
                continue
            file_tokens: set[str] = set(_tokenize(content))
            if not file_tokens:
                continue
            overlap: int = len(query_tokens & file_tokens)
            score: float = overlap / len(query_tokens)
            scored.append((score, md_file, content))

        scored.sort(key=lambda pair: pair[0], reverse=True)

        results: list[SearchResult] = []
        for score, path, content in scored[:top_k]:
            snippet: str = (
                _extract_snippet(content, query_tokens)
                if content else path.name
            )
            results.append(
                SearchResult(
                    article_id=path.stem,
                    title=path.stem.replace("-", " ").replace("_", " ").title(),
                    snippet=snippet,
                    score=score,
                    source_file=str(path),
                )
            )
        return results

    def search_hybrid(
        self,
        query: str,
        sources_dir: Path | None = None,
        top_k: int = 5,
    ) -> list[SearchResult]:
        wiki_results: list[SearchResult] = self.search(query, top_k=top_k)
        raw_results: list[SearchResult] = (
            self.search_raw(query, sources_dir, top_k=top_k)
            if sources_dir and sources_dir.is_dir()
            else []
        )

        if not raw_results:
            return wiki_results
        if not wiki_results:
            for r in raw_results:
                r.score = round(r.score, 3)
            return raw_results[:top_k]

        merged: dict[str, SearchResult] = {}
        seen: set[str] = set()

        wiki_boost: float = 1.5
        for _rank, r in enumerate(wiki_results):
            r.score = round(r.score * wiki_boost, 3)
            key: str = r.article_id
            merged[key] = r
            seen.add(key)

        for _rank, r in enumerate(raw_results):
            key = r.article_id
            if key not in seen:
                r.score = round(r.score, 3)
                merged[key] = r

        sorted_results: list[SearchResult] = sorted(
            merged.values(), key=lambda x: x.score, reverse=True
        )
        return sorted_results[:top_k]


def _tokenize(text: str) -> list[str]:
    tokens: list[str] = _TOKEN_SPLIT_RE.split(text.lower())
    return [t for t in tokens if len(t) > 1]


def _extract_snippet(content: str, query_tokens: set[str], max_chars: int = 300) -> str:
    _TOKEN_SPLIT_RE.split(content)
    best_score: int = 0
    best_start: int = 0
    words: list[str] = _TOKEN_SPLIT_RE.split(content.lower())

    for i in range(len(words)):
        score: int = 0
        for j in range(i, min(i + 60, len(words))):
            if words[j] in query_tokens:
                score += 1
        if score > best_score:
            best_score = score
            _word_to_char_index(content, i)

    if best_score == 0:
        return content[:max_chars].rsplit(" ", 1)[0] + "..."

    start_char: int = _word_to_char_index(content, best_start)
    return content[start_char : start_char + max_chars].rsplit(" ", 1)[0] + "..."


def _word_to_char_index(text: str, word_index: int) -> int:
    pos: int = 0
    for _ in range(word_index):
        match: re.Match[str] | None = re.search(r"\S+\s*", text[pos:])
        if not match:
            break
        pos += match.end()
    return pos
