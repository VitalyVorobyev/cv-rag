from __future__ import annotations

import pytest

from cv_rag.retrieve import HybridRetriever, NoRelevantSourcesError, RetrievedChunk


class DummyEmbedder:
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]


class DummyQdrantStore:
    def __init__(self, hits: list[dict[str, object]]) -> None:
        self.hits = hits

    def search(self, query_vector: list[float], limit: int) -> list[dict[str, object]]:
        return self.hits[:limit]


class DummySQLiteStore:
    def __init__(self, hits: list[dict[str, object]]) -> None:
        self.hits = hits

    def keyword_search(self, query: str, limit: int) -> list[dict[str, object]]:
        return self.hits[:limit]


def test_retrieve_raises_when_query_terms_not_covered_and_scores_low() -> None:
    qdrant_hits = [
        {
            "chunk_id": "x:0",
            "arxiv_id": "2602.00001",
            "title": "Diffusion for textures",
            "section_title": "Abstract",
            "text": "This paper studies texture refinement.",
            "score": 0.12,
        }
    ]

    retriever = HybridRetriever(
        embedder=DummyEmbedder(),
        qdrant_store=DummyQdrantStore(qdrant_hits),
        sqlite_store=DummySQLiteStore([]),
    )

    with pytest.raises(NoRelevantSourcesError):
        retriever.retrieve(
            query="Explain LoFTR versus SuperGlue",
            top_k=5,
            vector_k=5,
            keyword_k=5,
            require_relevance=True,
            vector_score_threshold=0.45,
        )


def test_retrieve_allows_results_when_rare_term_overlaps() -> None:
    qdrant_hits = [
        {
            "chunk_id": "x:0",
            "arxiv_id": "2104.00680",
            "title": "LoFTR: Detector-Free Local Feature Matching",
            "section_title": "Method",
            "text": "LoFTR uses coarse-to-fine matching.",
            "score": 0.12,
        }
    ]

    retriever = HybridRetriever(
        embedder=DummyEmbedder(),
        qdrant_store=DummyQdrantStore(qdrant_hits),
        sqlite_store=DummySQLiteStore([]),
    )

    chunks = retriever.retrieve(
        query="Explain LoFTR versus SuperGlue",
        top_k=5,
        vector_k=5,
        keyword_k=5,
        require_relevance=True,
        vector_score_threshold=0.45,
    )

    assert len(chunks) == 1
    assert chunks[0].arxiv_id == "2104.00680"


def test_dedupe_by_arxiv_section_chunk_id() -> None:
    chunks = [
        RetrievedChunk(
            chunk_id="2104.00680:5",
            arxiv_id="2104.00680",
            title="LoFTR",
            section_title="Method",
            text="a",
            fused_score=0.4,
        ),
        RetrievedChunk(
            chunk_id="2104.00680:5",
            arxiv_id="2104.00680",
            title="LoFTR",
            section_title="Method",
            text="b",
            fused_score=0.3,
        ),
    ]

    deduped = HybridRetriever._dedupe_ranked_chunks(chunks)
    assert len(deduped) == 1
