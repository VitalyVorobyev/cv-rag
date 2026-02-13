from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any

from cv_rag.embeddings import OllamaEmbedClient
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
SECTION_PRIORITY_RE = re.compile(r"(method|approach|architecture|supervision|loss|training)", re.IGNORECASE)
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


class NoRelevantSourcesError(RuntimeError):
    def __init__(self, candidates: list[RetrievedChunk]) -> None:
        super().__init__("no relevant sources")
        self.candidates = candidates


def format_citation(arxiv_id: str, section_title: str) -> str:
    section = section_title.strip() or "Untitled"
    return f"arXiv:{arxiv_id} §{section}"


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
        max_chunks_per_doc: int = 4,
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

        ranked = sorted(by_id.values(), key=lambda item: item.fused_score, reverse=True)
        deduped = self._dedupe_ranked_chunks(ranked)
        quota_limited = self._apply_doc_quota(deduped, max_chunks_per_doc=max_chunks_per_doc)
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
    def _apply_doc_quota(chunks: list[RetrievedChunk], max_chunks_per_doc: int) -> list[RetrievedChunk]:
        if max_chunks_per_doc <= 0:
            return chunks
        out: list[RetrievedChunk] = []
        per_doc_counts: dict[str, int] = {}
        for chunk in chunks:
            count = per_doc_counts.get(chunk.arxiv_id, 0)
            if count >= max_chunks_per_doc:
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


def build_answer_prompt(query: str, chunks: list[RetrievedChunk]) -> str:
    lines: list[str] = []
    lines.append("You are answering a computer vision research question using paper excerpts.")
    lines.append("Use only the provided context. If context is insufficient, say so.")
    lines.append("Cite claims with inline citations in this format: arXiv:ID §Section.")
    lines.append("")
    lines.append(f"Question: {query}")
    lines.append("")
    lines.append("Context:")

    for idx, chunk in enumerate(chunks, start=1):
        citation = format_citation(chunk.arxiv_id, chunk.section_title)
        lines.append(f"[{idx}] {citation}")
        lines.append(chunk.text)
        lines.append("")

    lines.append("Answer:")
    return "\n".join(lines)


def build_strict_answer_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    lines: list[str] = []
    lines.append("You are a careful computer vision research assistant.")
    lines.append("")
    lines.append("Question:")
    lines.append(question)
    lines.append("")
    lines.append("Sources:")

    for idx, chunk in enumerate(chunks, start=1):
        section = chunk.section_title.strip() or "Untitled"
        snippet = " ".join(chunk.text.split())
        lines.append(f"[S{idx}]")
        lines.append(f"arxiv_id: {chunk.arxiv_id}")
        lines.append(f"title: {chunk.title}")
        lines.append(f"section: {section}")
        lines.append(f"text: {snippet}")
        lines.append("")

    lines.append("Rules:")
    lines.append("1. Only use information supported by the sources.")
    lines.append("2. Every non-trivial claim must include citations like [S3][S7].")
    lines.append("3. If sources are insufficient, state what is missing and ask one clarifying question.")
    lines.append("4. Prefer comparisons and what each paper claims or shows.")
    lines.append("5. Do not fabricate details, metrics, or citations.")
    lines.append("")
    lines.append("Answer:")
    return "\n".join(lines)
