from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from knox.core.models import CompiledWiki, WikiNode
from knox.core.template_loader import TemplateLoader
from knox.llm.client import LLMClient

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
    def __init__(self, wiki: CompiledWiki) -> None:
        self._wiki: CompiledWiki = wiki
        self._template_loader: TemplateLoader = TemplateLoader()

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
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


def _tokenize(text: str) -> list[str]:
    tokens: list[str] = _TOKEN_SPLIT_RE.split(text.lower())
    return [t for t in tokens if len(t) > 1]


def _extract_snippet(content: str, query_tokens: set[str], max_chars: int = 300) -> str:
    sentences: list[str] = _TOKEN_SPLIT_RE.split(content)
    best_score: int = 0
    best_start: int = 0
    words: list[str] = _TOKEN_SPLIT_RE.split(content.lower())

    for i in range(len(words)):
        score: int = 0
        char_pos: int = 0
        for j in range(i, min(i + 60, len(words))):
            if words[j] in query_tokens:
                score += 1
        if score > best_score:
            best_score = score
            char_pos = _word_to_char_index(content, i)

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
