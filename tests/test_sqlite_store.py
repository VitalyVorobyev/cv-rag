from __future__ import annotations

from pathlib import Path

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
