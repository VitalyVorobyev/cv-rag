from __future__ import annotations

import json
from pathlib import Path

from cv_rag.arxiv_sync import PaperMetadata
from cv_rag.sqlite_store import SQLiteStore


def test_keyword_search_handles_trailing_punctuation(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "test.sqlite3")
    try:
        store.create_schema()
        store.upsert_chunks(
            [
                {
                    "chunk_id": "2401.00001:0",
                    "arxiv_id": "2401.00001",
                    "title": "LoFTR and SuperGlue",
                    "section_title": "Comparison",
                    "chunk_index": 0,
                    "text": "SuperGlue is sparse and LoFTR is dense.",
                }
            ]
        )

        hits = store.keyword_search("Compare with SuperGlue.", limit=5)
        assert hits
    finally:
        store.close()


def test_get_ingested_versioned_ids_returns_non_empty_versions(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "test.sqlite3")
    try:
        store.create_schema()
        store.upsert_paper(
            paper=PaperMetadata(
                arxiv_id="2104.00680",
                arxiv_id_with_version="2104.00680v1",
                version="v1",
                title="LoFTR",
                summary="",
                published=None,
                updated=None,
                authors=["A. Author"],
                pdf_url="https://arxiv.org/pdf/2104.00680v1.pdf",
                abs_url="https://arxiv.org/abs/2104.00680v1",
            ),
            pdf_path=None,
            tei_path=None,
        )
        store.upsert_paper(
            paper=PaperMetadata(
                arxiv_id="1911.11763",
                arxiv_id_with_version="1911.11763v2",
                version="v2",
                title="SuperGlue",
                summary="",
                published=None,
                updated=None,
                authors=["B. Author"],
                pdf_url="https://arxiv.org/pdf/1911.11763v2.pdf",
                abs_url="https://arxiv.org/abs/1911.11763v2",
            ),
            pdf_path=None,
            tei_path=None,
        )

        assert store.get_ingested_versioned_ids() == {"2104.00680v1", "1911.11763v2"}
    finally:
        store.close()


def test_paper_metrics_upsert_and_read(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "test.sqlite3")
    try:
        store.create_schema()
        store.upsert_paper(
            paper=PaperMetadata(
                arxiv_id="2104.00680",
                arxiv_id_with_version="2104.00680v1",
                version="v1",
                title="LoFTR",
                summary="",
                published=None,
                updated=None,
                authors=["A. Author"],
                pdf_url="https://arxiv.org/pdf/2104.00680v1.pdf",
                abs_url="https://arxiv.org/abs/2104.00680v1",
            ),
            pdf_path=None,
            tei_path=None,
        )
        store.upsert_paper_metrics(
            [
                {
                    "arxiv_id": "2104.00680",
                    "citation_count": 321,
                    "year": 2021,
                    "venue": "CVPR",
                    "publication_types": json.dumps(["Conference"]),
                    "is_peer_reviewed": 1,
                    "score": 7.5,
                    "tier": 0,
                    "updated_at": 1_700_000_000,
                }
            ]
        )

        tiers = store.get_paper_tiers(["2104.00680"])
        timestamps = store.get_paper_metric_timestamps(["2104.00680"])
        distribution = store.get_paper_metrics_tier_distribution()

        assert tiers == {"2104.00680": 0}
        assert timestamps == {"2104.00680": 1_700_000_000}
        assert distribution == {0: 1}
    finally:
        store.close()
