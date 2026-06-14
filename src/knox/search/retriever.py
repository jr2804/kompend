from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from knox.core.models import CompiledWiki, WikiNode
from knox.core.template_loader import TemplateLoader, get_template_loader
from knox.llm.client import LLMClient
from knox.rag.zvec_store import ZvecWikiStore
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
        self._template_loader: TemplateLoader = get_template_loader()
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
            snippet: str = _extract_snippet(content, query_tokens) if content else path.name
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
        raw_results: list[SearchResult] = self.search_raw(query, sources_dir, top_k=top_k) if sources_dir and sources_dir.is_dir() else []

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

        sorted_results: list[SearchResult] = sorted(merged.values(), key=lambda x: x.score, reverse=True)
        return sorted_results[:top_k]


class HybridRetriever:
    """Combine FTS, vector, token-overlap, and raw-source search via RRF."""

    _RRF_K: int = 60

    def __init__(
        self,
        wiki: CompiledWiki,
        index_path: Path | None = None,
        vector_store: ZvecWikiStore | None = None,
        sources_dir: Path | None = None,
    ) -> None:
        self._wiki = wiki
        self._index_path = index_path
        self._vector_store = vector_store
        self._sources_dir = sources_dir
        self._node_by_id: dict[str, WikiNode] = {n.id: n for n in wiki.nodes}

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        return self.search_hybrid(query, self._sources_dir, top_k=top_k)

    def search_wiki(self, query: str, top_k: int = 5) -> list[SearchResult]:
        query_tokens = _tokenize(query)
        fts_results = self._wiki_fts_results(query, top_k)
        vector_results = self._wiki_vector_results(query, top_k)
        token_results = self._wiki_token_results(query, query_tokens, top_k)
        return self._fuse_results(fts_results, vector_results, token_results, top_k=top_k)

    def search_raw(self, query: str, sources_dir: Path, top_k: int = 5) -> list[SearchResult]:
        return self._raw_bm25_results(query, sources_dir, top_k)

    def search_hybrid(
        self,
        query: str,
        sources_dir: Path | None = None,
        top_k: int = 5,
        expand_graph: bool = False,
    ) -> list[SearchResult]:
        wiki_results = self.search_wiki(query, top_k=top_k)
        raw_results: list[SearchResult] = []
        effective_sources_dir = sources_dir or self._sources_dir
        if effective_sources_dir is not None and effective_sources_dir.is_dir():
            raw_results = self._raw_bm25_results(query, effective_sources_dir, top_k=top_k)
        if not raw_results:
            results = wiki_results
        elif not wiki_results:
            results = raw_results[:top_k]
        else:
            results = self._fuse_results(wiki_results, raw_results, top_k=top_k)

        if expand_graph and results:
            results = self._graph_expansion_results(results, top_k=top_k)

        return results

    def _wiki_fts_results(self, query: str, top_k: int) -> list[SearchResult]:
        if self._index_path is None or not self._index_path.exists():
            return []
        try:
            rows = search_fts(query, self._index_path, top_k=top_k)
        except Exception as exc:
            logger.warning("FTS search failed: %s", exc)
            return []
        query_tokens = set(_tokenize(query))
        results: list[SearchResult] = []
        for row in rows:
            content = row.get("content") or ""
            snippet = _extract_snippet(content, query_tokens) if content else ""
            results.append(
                SearchResult(
                    article_id=row["article_id"],
                    title=row.get("title", ""),
                    snippet=snippet,
                    score=float(row.get("score", 0.0)),
                    source_file=row.get("source_file"),
                )
            )
        return results

    def _wiki_vector_results(self, query: str, top_k: int) -> list[SearchResult]:
        if self._vector_store is None or not self._vector_store.available:
            return []
        try:
            hits = self._vector_store.search(query, top_k=top_k)
        except Exception as exc:
            logger.warning("Vector search failed: %s", exc)
            return []
        query_tokens = set(_tokenize(query))
        results: list[SearchResult] = []
        for hit in hits:
            node_id = hit.get("id", "")
            node = self._node_by_id.get(node_id)
            title = hit.get("title") or (node.title if node else node_id)
            snippet = _extract_snippet(node.content, query_tokens) if node and node.content else hit.get("content_preview", "") or title
            source_file: str | None = hit.get("source_file")
            if source_file is None and node and node.metadata and node.metadata.get("source"):
                source_file = str(node.metadata["source"])
            results.append(
                SearchResult(
                    article_id=node_id,
                    title=title,
                    snippet=snippet,
                    score=float(hit.get("score", 0.0)),
                    source_file=source_file,
                )
            )
        return results

    def _wiki_token_results(
        self,
        query: str,
        query_tokens: list[str],
        top_k: int,
    ) -> list[SearchResult]:
        if not query_tokens:
            return []
        query_set = set(query_tokens)
        scored: list[tuple[float, WikiNode]] = []
        for node in self._wiki.nodes:
            content = node.content or node.title
            node_tokens = set(_tokenize(content))
            if not node_tokens:
                continue
            overlap = len(query_set & node_tokens)
            score = overlap / len(query_set)
            if score > 0:
                scored.append((score, node))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        results: list[SearchResult] = []
        for score, node in scored[:top_k]:
            snippet = _extract_snippet(node.content, query_set) if node.content else node.title
            source_file = None
            if node.metadata and node.metadata.get("source"):
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

    def _fuse_results(self, *lists: list[SearchResult], top_k: int) -> list[SearchResult]:
        scores: dict[str, float] = {}
        representatives: dict[str, SearchResult] = {}
        for result_list in lists:
            for rank, result in enumerate(result_list):
                scores[result.article_id] = scores.get(result.article_id, 0.0) + 1.0 / (self._RRF_K + rank)
                if result.article_id not in representatives:
                    representatives[result.article_id] = result
        fused: list[SearchResult] = []
        for article_id, score in scores.items():
            r = representatives[article_id]
            fused.append(
                SearchResult(
                    article_id=article_id,
                    title=r.title,
                    snippet=r.snippet,
                    score=round(score, 6),
                    source_file=r.source_file,
                )
            )
        fused.sort(key=lambda x: x.score, reverse=True)
        return fused[:top_k]

    def _graph_expansion_results(self, initial_results: list[SearchResult], top_k: int) -> list[SearchResult]:
        """Expand search results by following typed relations in the wiki graph."""
        if not initial_results or not self._wiki.edges:
            return initial_results

        expanded: dict[str, SearchResult] = {r.article_id: r for r in initial_results}
        decay_factor = 0.5

        for result in initial_results:
            for edge in self._wiki.edges:
                source_id = edge.get("source")
                target_id = edge.get("target")
                if isinstance(source_id, str) and isinstance(target_id, str):
                    if source_id == result.article_id and target_id not in expanded:
                        node = self._node_by_id.get(target_id)
                        if node:
                            expanded[target_id] = SearchResult(
                                article_id=target_id,
                                title=node.title,
                                snippet=node.content[:300] + "..." if node.content else "",
                                score=result.score * decay_factor,
                                source_file=str(node.metadata.get("source")) if node.metadata and node.metadata.get("source") else None,
                            )
                    elif target_id == result.article_id and source_id not in expanded:
                        node = self._node_by_id.get(source_id)
                        if node:
                            expanded[source_id] = SearchResult(
                                article_id=source_id,
                                title=node.title,
                                snippet=node.content[:300] + "..." if node.content else "",
                                score=result.score * decay_factor,
                                source_file=str(node.metadata.get("source")) if node.metadata and node.metadata.get("source") else None,
                            )

        sorted_results = sorted(expanded.values(), key=lambda x: x.score, reverse=True)
        return sorted_results[:top_k]

    @staticmethod
    def _raw_bm25_results(query: str, sources_dir: Path, top_k: int) -> list[SearchResult]:
        """Retrieve from raw source files using BM25."""
        query_tokens = _tokenize(query)
        if not query_tokens or not sources_dir.is_dir():
            return []

        docs: list[tuple[Path, str, list[str]]] = []
        for md_file in sources_dir.rglob("*.md"):
            try:
                content = md_file.read_text(encoding="utf-8", errors="replace")
            except Exception as exc:
                logger.warning("Skipping unreadable source %s: %s", md_file, exc)
                continue
            tokens = _tokenize(content)
            if tokens:
                docs.append((md_file, content, tokens))

        if not docs:
            return []

        corpus = [doc[2] for doc in docs]
        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(query_tokens)

        scored: list[tuple[float, Path, str]] = []
        for i, score in enumerate(scores):
            # BM25 can return negative scores for non-matching docs in small corpora.
            # We only care about documents that actually contain at least one query token.
            if any(token in docs[i][2] for token in query_tokens):
                scored.append((float(score), docs[i][0], docs[i][1]))

        scored.sort(key=lambda pair: pair[0], reverse=True)

        query_set = set(query_tokens)
        results: list[SearchResult] = []
        for score, path, content in scored[:top_k]:
            snippet = _extract_snippet(content, query_set) if content else path.name
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


def _tokenize(text: str) -> list[str]:
    tokens: list[str] = _TOKEN_SPLIT_RE.split(text.lower())
    return [t for t in tokens if len(t) > 1]


def _extract_snippet(content: str, query_tokens: set[str], max_chars: int = 300) -> str:
    best_score: int = 0
    best_first_match: int = 0
    words: list[str] = _TOKEN_SPLIT_RE.split(content.lower())

    for i in range(len(words)):
        score: int = 0
        first_match: int | None = None
        for j in range(i, min(i + 60, len(words))):
            if words[j] in query_tokens:
                score += 1
                if first_match is None:
                    first_match = j
        if score > best_score:
            best_score = score
            best_first_match = first_match if first_match is not None else i

    if best_score == 0:
        return content[:max_chars].rsplit(" ", 1)[0] + "..."

    start_char: int = _word_to_char_index(content, best_first_match)
    return content[start_char : start_char + max_chars].rsplit(" ", 1)[0] + "..."


def _word_to_char_index(text: str, word_index: int) -> int:
    pos: int = 0
    for _ in range(word_index):
        match: re.Match[str] | None = re.search(r"\S+\s*", text[pos:])
        if not match:
            break
        pos += match.end()
    return pos
