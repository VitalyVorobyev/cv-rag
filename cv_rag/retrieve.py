from __future__ import annotations

from dataclasses import dataclass, field
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
    ) -> list[RetrievedChunk]:
        query_vector = self.embedder.embed_texts([query])[0]
        vector_hits = self.qdrant_store.search(query_vector, limit=vector_k)
        keyword_hits = self.sqlite_store.keyword_search(query, limit=keyword_k)

        by_id: dict[str, RetrievedChunk] = {}

        self._merge_hits(by_id, vector_hits, source="vector", score_field="score")
        self._merge_hits(by_id, keyword_hits, source="keyword", score_field="score")

        ranked = sorted(by_id.values(), key=lambda item: item.fused_score, reverse=True)
        return ranked[:top_k]

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
