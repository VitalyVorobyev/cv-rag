from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from cv_rag.embeddings import OllamaEmbedClient
from cv_rag.exceptions import RetrievalError
from cv_rag.qdrant_store import QdrantStore
from cv_rag.sqlite_store import SQLiteStore


@dataclass(slots=True)
class RetrievedChunk:
    chunk_id: str
    arxiv_id: str
    title: str
    section_title: str
    text: str
    fused_score: float
    vector_score: float | None = None
    keyword_score: float | None = None
    sources: set[str] = field(default_factory=set)


QUERY_TOKEN_RE = re.compile(r"[a-z0-9]+")
QUESTION_ALNUM_RE = re.compile(r"[A-Za-z0-9]+")
SECTION_PRIORITY_RE = re.compile(
    r"(method|approach|supervision|loss|training|optimal matching)",
    re.IGNORECASE,
)
ENTITY_TOKEN_WHITELIST = {"loftr", "superglue"}
TIER_SCORE_BOOSTS = {
    0: 0.25,
    1: 0.10,
    2: 0.0,
}
COMMON_QUERY_TERMS = {
    "about",
    "against",
    "between",
    "compare",
    "difference",
    "differences",
    "does",
    "each",
    "explain",
    "fails",
    "from",
    "idea",
    "into",
    "key",
    "method",
    "objective",
    "paper",
    "show",
    "shows",
    "summarize",
    "their",
    "them",
    "these",
    "this",
    "training",
    "when",
    "with",
}


class NoRelevantSourcesError(RetrievalError):
    def __init__(self, candidates: list[RetrievedChunk]) -> None:
        super().__init__("no relevant sources")
        self.candidates = candidates


def format_citation(arxiv_id: str, section_title: str) -> str:
    section = section_title.strip() or "Untitled"
    return f"arXiv:{arxiv_id} §{section}"


def _is_camel_caseish(token: str) -> bool:
    return any(char.isupper() for char in token) and any(char.islower() for char in token)


def extract_entity_like_tokens(question: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for match in QUESTION_ALNUM_RE.finditer(question):
        raw = match.group(0)
        lower = raw.lower()
        include = False
        if len(lower) >= 5:
            include = True
        if any(char.isdigit() for char in raw):
            include = True
        if _is_camel_caseish(raw):
            include = True
        if lower in ENTITY_TOKEN_WHITELIST:
            include = True
        if not include:
            continue
        if lower in seen:
            continue
        seen.add(lower)
        out.append(lower)
    return out


def filter_chunks_by_entity_tokens(chunks: list[RetrievedChunk], entity_tokens: list[str]) -> list[RetrievedChunk]:
    if not entity_tokens:
        return chunks
    token_patterns = [re.compile(rf"\b{re.escape(token)}\b", re.IGNORECASE) for token in entity_tokens]
    filtered: list[RetrievedChunk] = []
    for chunk in chunks:
        haystack = f"{chunk.title}\n{chunk.section_title}\n{chunk.text}"
        if any(pattern.search(haystack) for pattern in token_patterns):
            filtered.append(chunk)
    return filtered


class HybridRetriever:
    def __init__(
        self,
        embedder: OllamaEmbedClient,
        qdrant_store: QdrantStore,
        sqlite_store: SQLiteStore,
        rrf_k: int = 60,
    ) -> None:
        self.embedder = embedder
        self.qdrant_store = qdrant_store
        self.sqlite_store = sqlite_store
        self.rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        top_k: int = 8,
        vector_k: int = 12,
        keyword_k: int = 12,
        require_relevance: bool = False,
        vector_score_threshold: float = 0.45,
        max_per_doc: int = 4,
        section_boost: float = 0.0,
    ) -> list[RetrievedChunk]:
        query_vector = self.embedder.embed_texts([query])[0]
        vector_hits = self.qdrant_store.search(query_vector, limit=vector_k)
        keyword_hits = self.sqlite_store.keyword_search(query, limit=keyword_k)

        by_id: dict[str, RetrievedChunk] = {}

        self._merge_hits(by_id, vector_hits, source="vector", score_field="score")
        self._merge_hits(by_id, keyword_hits, source="keyword", score_field="score")

        if section_boost > 0:
            for item in by_id.values():
                if self._matches_priority_section(item):
                    item.fused_score += section_boost

        if by_id and hasattr(self.sqlite_store, "get_paper_tiers"):
            tiers_by_arxiv = self.sqlite_store.get_paper_tiers(
                sorted({chunk.arxiv_id for chunk in by_id.values()})
            )
            for item in by_id.values():
                item.fused_score += TIER_SCORE_BOOSTS.get(tiers_by_arxiv.get(item.arxiv_id, 2), 0.0)

        ranked = sorted(by_id.values(), key=lambda item: item.fused_score, reverse=True)
        deduped = self._dedupe_ranked_chunks(ranked)
        quota_limited = self._apply_doc_quota(deduped, max_per_doc=max_per_doc)
        shortlist = quota_limited[: max(top_k, 3)]

        if require_relevance and self._is_irrelevant_result(query, shortlist, vector_score_threshold):
            raise NoRelevantSourcesError(shortlist[:3])

        return quota_limited[:top_k]

    def _merge_hits(
        self,
        by_id: dict[str, RetrievedChunk],
        hits: list[dict[str, Any]],
        source: str,
        score_field: str,
    ) -> None:
        for rank_idx, hit in enumerate(hits, start=1):
            chunk_id = hit.get("chunk_id", "")
            if not chunk_id:
                continue

            fused = 1.0 / (self.rrf_k + rank_idx)
            item = by_id.get(chunk_id)
            if item is None:
                item = RetrievedChunk(
                    chunk_id=chunk_id,
                    arxiv_id=str(hit.get("arxiv_id", "")),
                    title=str(hit.get("title", "")),
                    section_title=str(hit.get("section_title", "")),
                    text=str(hit.get("text", "")),
                    fused_score=0.0,
                )
                by_id[chunk_id] = item

            item.fused_score += fused
            item.sources.add(source)
            raw_score = hit.get(score_field)
            if source == "vector" and raw_score is not None:
                item.vector_score = float(raw_score)
            if source == "keyword" and raw_score is not None:
                item.keyword_score = float(raw_score)

    @staticmethod
    def _dedupe_ranked_chunks(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        deduped: list[RetrievedChunk] = []
        seen: set[tuple[str, str, str]] = set()
        for chunk in chunks:
            key = (
                chunk.arxiv_id,
                chunk.section_title.strip().casefold(),
                chunk.chunk_id,
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(chunk)
        return deduped

    @staticmethod
    def _apply_doc_quota(chunks: list[RetrievedChunk], max_per_doc: int) -> list[RetrievedChunk]:
        if max_per_doc <= 0:
            return chunks
        out: list[RetrievedChunk] = []
        per_doc_counts: dict[str, int] = {}
        for chunk in chunks:
            count = per_doc_counts.get(chunk.arxiv_id, 0)
            if count >= max_per_doc:
                continue
            per_doc_counts[chunk.arxiv_id] = count + 1
            out.append(chunk)
        return out

    @staticmethod
    def _matches_priority_section(chunk: RetrievedChunk) -> bool:
        haystack = f"{chunk.title}\n{chunk.section_title}"
        return SECTION_PRIORITY_RE.search(haystack) is not None

    @staticmethod
    def _extract_rare_query_terms(query: str) -> list[str]:
        terms = QUERY_TOKEN_RE.findall(query.casefold())
        out: list[str] = []
        seen: set[str] = set()
        for term in terms:
            if term in COMMON_QUERY_TERMS:
                continue
            if len(term) < 5 and not any(char.isdigit() for char in term):
                continue
            if term in seen:
                continue
            seen.add(term)
            out.append(term)
        return out

    def _is_irrelevant_result(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        vector_score_threshold: float,
    ) -> bool:
        if not candidates:
            return True

        rare_terms = self._extract_rare_query_terms(query)
        if not rare_terms:
            return False

        overlap_found = False
        for chunk in candidates:
            haystack = f"{chunk.title}\n{chunk.text}".casefold()
            if any(term in haystack for term in rare_terms):
                overlap_found = True
                break

        max_vector = max(
            (chunk.vector_score for chunk in candidates if chunk.vector_score is not None),
            default=-1.0,
        )
        return (not overlap_found) and (max_vector < vector_score_threshold)
