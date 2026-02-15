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

DEFAULT_PROVENANCE_BOOSTS = {
    "curated": 0.08,
    "canonical_api": 0.05,
    "scraped": 0.02,
}


def format_citation(arxiv_id: str, section_title: str) -> str:
    """Compatibility re-export for call sites importing format_citation from hybrid."""
    return _format_citation(arxiv_id, section_title)


def get_provenance_boost(
    *,
    doc_id: str,
    sqlite_store: SQLiteStore,
    provenance_boosts: dict[str, float] | None = None,
) -> float:
    boosts = provenance_boosts or DEFAULT_PROVENANCE_BOOSTS
    by_doc = sqlite_store.get_doc_provenance_kinds([doc_id])
    provenance_kind = by_doc.get(doc_id)
    if provenance_kind is None:
        return 0.0
    return boosts.get(provenance_kind, 0.0)


class HybridRetriever:
    def __init__(
        self,
        embedder: OllamaEmbedClient,
        qdrant_store: QdrantStore,
        sqlite_store: SQLiteStore,
        rrf_k: int = 60,
        provenance_boosts: dict[str, float] | None = None,
    ) -> None:
        self.embedder = embedder
        self.qdrant_store = qdrant_store
        self.sqlite_store = sqlite_store
        self.rrf_k = rrf_k
        self.provenance_boosts = provenance_boosts or dict(DEFAULT_PROVENANCE_BOOSTS)

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

        if by_id and hasattr(self.sqlite_store, "get_doc_provenance_kinds"):
            doc_ids = sorted({chunk.doc_id for chunk in by_id.values() if chunk.doc_id})
            if doc_ids:
                provenance_by_doc = self.sqlite_store.get_doc_provenance_kinds(doc_ids)
                for item in by_id.values():
                    if not item.doc_id:
                        continue
                    provenance_kind = provenance_by_doc.get(item.doc_id)
                    if provenance_kind is None:
                        continue
                    item.fused_score += self.provenance_boosts.get(provenance_kind, 0.0)

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
                    doc_id=str(hit["doc_id"]) if hit.get("doc_id") else None,
                )
                by_id[chunk_id] = item

            item.fused_score += fused
            item.sources.add(source)
            if not item.doc_id and hit.get("doc_id"):
                item.doc_id = str(hit["doc_id"])
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
            doc_key = chunk.doc_id or chunk.arxiv_id
            key = (
                doc_key,
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
            doc_key = chunk.doc_id or chunk.arxiv_id
            count = per_doc_counts.get(doc_key, 0)
            if count >= max_per_doc:
                continue
            per_doc_counts[doc_key] = count + 1
            out.append(chunk)
        return out

    @staticmethod
    def _is_irrelevant_result(
        query: str,
        candidates: list[RetrievedChunk],
        vector_score_threshold: float,
    ) -> bool:
        return is_irrelevant_result(query, candidates, vector_score_threshold)
