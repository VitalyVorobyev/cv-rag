from __future__ import annotations

from typing import Any

from cv_rag.embeddings import OllamaEmbedClient
from cv_rag.retrieval.models import (
    NoRelevantSourcesError,
    RetrievedChunk,
)
from cv_rag.retrieval.models import (
    format_citation as _format_citation,
)
from cv_rag.retrieval.relevance import (
    is_irrelevant_result,
    matches_priority_section,
)
from cv_rag.storage.qdrant import QdrantStore
from cv_rag.storage.sqlite import SQLiteStore

TIER_SCORE_BOOSTS = {
    0: 0.25,
    1: 0.10,
    2: 0.0,
}


def format_citation(arxiv_id: str, section_title: str) -> str:
    """Compatibility re-export for call sites importing format_citation from hybrid."""
    return _format_citation(arxiv_id, section_title)


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
                if matches_priority_section(item):
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
    def _is_irrelevant_result(
        query: str,
        candidates: list[RetrievedChunk],
        vector_score_threshold: float,
    ) -> bool:
        return is_irrelevant_result(query, candidates, vector_score_threshold)
