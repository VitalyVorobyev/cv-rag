from __future__ import annotations

import pytest

from cv_rag.retrieval.hybrid import HybridRetriever, NoRelevantSourcesError, RetrievedChunk


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
        self.tiers: dict[str, int] = {}

    def keyword_search(self, query: str, limit: int) -> list[dict[str, object]]:
        return self.hits[:limit]

    def get_paper_tiers(self, arxiv_ids: list[str]) -> dict[str, int]:
        return {arxiv_id: self.tiers[arxiv_id] for arxiv_id in arxiv_ids if arxiv_id in self.tiers}


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


def test_section_boost_promotes_method_section() -> None:
    qdrant_hits = [
        {
            "chunk_id": "a:0",
            "arxiv_id": "1000.00001",
            "title": "Paper A",
            "section_title": "Introduction",
            "text": "General overview",
            "score": 0.95,
        },
        {
            "chunk_id": "b:0",
            "arxiv_id": "1000.00002",
            "title": "Paper B",
            "section_title": "Optimal Matching",
            "text": "Our approach details",
            "score": 0.90,
        },
    ]

    retriever = HybridRetriever(
        embedder=DummyEmbedder(),
        qdrant_store=DummyQdrantStore(qdrant_hits),
        sqlite_store=DummySQLiteStore([]),
    )

    without_boost = retriever.retrieve(query="matching", top_k=2, vector_k=2, keyword_k=0, section_boost=0.0)
    with_boost = retriever.retrieve(query="matching", top_k=2, vector_k=2, keyword_k=0, section_boost=0.01)

    assert without_boost[0].arxiv_id == "1000.00001"
    assert with_boost[0].arxiv_id == "1000.00002"


def test_max_chunks_per_doc_quota_is_enforced() -> None:
    qdrant_hits: list[dict[str, object]] = []
    for i in range(6):
        qdrant_hits.append(
            {
                "chunk_id": f"a:{i}",
                "arxiv_id": "2000.00001",
                "title": "Paper A",
                "section_title": "Method",
                "text": f"chunk {i}",
                "score": 0.9 - i * 0.01,
            }
        )
    qdrant_hits.extend(
        [
            {
                "chunk_id": "b:0",
                "arxiv_id": "2000.00002",
                "title": "Paper B",
                "section_title": "Method",
                "text": "chunk b",
                "score": 0.6,
            },
            {
                "chunk_id": "c:0",
                "arxiv_id": "2000.00003",
                "title": "Paper C",
                "section_title": "Method",
                "text": "chunk c",
                "score": 0.5,
            },
        ]
    )

    retriever = HybridRetriever(
        embedder=DummyEmbedder(),
        qdrant_store=DummyQdrantStore(qdrant_hits),
        sqlite_store=DummySQLiteStore([]),
    )

    chunks = retriever.retrieve(query="matching", top_k=8, vector_k=8, keyword_k=0, max_per_doc=2)
    counts: dict[str, int] = {}
    for chunk in chunks:
        counts[chunk.arxiv_id] = counts.get(chunk.arxiv_id, 0) + 1

    assert counts["2000.00001"] == 2


def test_tier_boost_promotes_tier0_paper() -> None:
    qdrant_hits = [
        {
            "chunk_id": "a:0",
            "arxiv_id": "1000.00001",
            "title": "Paper A",
            "section_title": "Introduction",
            "text": "General overview",
            "score": 0.95,
        },
        {
            "chunk_id": "b:0",
            "arxiv_id": "1000.00002",
            "title": "Paper B",
            "section_title": "Introduction",
            "text": "General overview",
            "score": 0.94,
        },
    ]
    sqlite_store = DummySQLiteStore([])
    sqlite_store.tiers = {
        "1000.00001": 2,
        "1000.00002": 0,
    }

    retriever = HybridRetriever(
        embedder=DummyEmbedder(),
        qdrant_store=DummyQdrantStore(qdrant_hits),
        sqlite_store=sqlite_store,
    )

    chunks = retriever.retrieve(query="overview", top_k=2, vector_k=2, keyword_k=0)
    assert chunks[0].arxiv_id == "1000.00002"
