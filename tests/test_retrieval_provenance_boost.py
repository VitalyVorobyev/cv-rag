from __future__ import annotations

from pathlib import Path

from cv_rag.retrieval.hybrid import HybridRetriever, get_provenance_boost
from cv_rag.storage.sqlite import SQLiteStore


class DummyEmbedder:
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]


class DummyQdrantStore:
    def __init__(self, hits: list[dict[str, object]]) -> None:
        self.hits = hits

    def search(self, query_vector: list[float], limit: int) -> list[dict[str, object]]:
        _ = query_vector
        return self.hits[:limit]


class DummySQLiteStore:
    def __init__(self, provenance_by_doc: dict[str, str]) -> None:
        self.provenance_by_doc = provenance_by_doc

    def keyword_search(self, query: str, limit: int) -> list[dict[str, object]]:
        _ = (query, limit)
        return []

    def get_paper_tiers(self, arxiv_ids: list[str]) -> dict[str, int]:
        _ = arxiv_ids
        return {}

    def get_doc_provenance_kinds(self, doc_ids: list[str]) -> dict[str, str]:
        return {
            doc_id: self.provenance_by_doc[doc_id]
            for doc_id in doc_ids
            if doc_id in self.provenance_by_doc
        }


def test_get_provenance_boost_reads_kind_from_sqlite_store(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "cv_rag.sqlite3")
    try:
        store.create_schema()
        store.conn.execute(
            """
            INSERT INTO documents (
                doc_id, arxiv_id, arxiv_id_with_version, doi, pdf_url, status,
                resolution_confidence, provenance_kind, best_source_kind, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "axv:2104.00680v2",
                "2104.00680",
                "2104.00680v2",
                None,
                "https://arxiv.org/pdf/2104.00680v2.pdf",
                "ready",
                1.0,
                "curated",
                "curated_repo",
                100,
                100,
            ),
        )
        store.conn.commit()

        boost = get_provenance_boost(doc_id="axv:2104.00680v2", sqlite_store=store)
        assert boost == 0.08
    finally:
        store.close()


def test_provenance_boost_changes_retrieval_order() -> None:
    qdrant_hits = [
        {
            "chunk_id": "a:0",
            "doc_id": "axv:doc-a",
            "arxiv_id": "doc-a",
            "title": "Doc A",
            "section_title": "Method",
            "text": "A",
            "score": 0.90,
        },
        {
            "chunk_id": "b:0",
            "doc_id": "axv:doc-b",
            "arxiv_id": "doc-b",
            "title": "Doc B",
            "section_title": "Method",
            "text": "B",
            "score": 0.89,
        },
    ]
    sqlite_store = DummySQLiteStore(
        provenance_by_doc={
            "axv:doc-a": "scraped",
            "axv:doc-b": "curated",
        }
    )

    no_boost = HybridRetriever(
        embedder=DummyEmbedder(),
        qdrant_store=DummyQdrantStore(qdrant_hits),
        sqlite_store=sqlite_store,
        provenance_boosts={"curated": 0.0, "canonical_api": 0.0, "scraped": 0.0},
    )
    with_boost = HybridRetriever(
        embedder=DummyEmbedder(),
        qdrant_store=DummyQdrantStore(qdrant_hits),
        sqlite_store=sqlite_store,
        provenance_boosts={"curated": 0.08, "canonical_api": 0.05, "scraped": 0.02},
    )

    without = no_boost.retrieve(query="matching", top_k=2, vector_k=2, keyword_k=0)
    boosted = with_boost.retrieve(query="matching", top_k=2, vector_k=2, keyword_k=0)

    assert without[0].doc_id == "axv:doc-a"
    assert boosted[0].doc_id == "axv:doc-b"
