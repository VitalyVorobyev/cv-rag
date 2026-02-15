from __future__ import annotations

from pathlib import Path

from cv_rag.storage.repositories import ReferenceRecord, ResolvedReference
from cv_rag.storage.sqlite import SQLiteStore


def test_doi_resolution_to_arxiv_merges_to_canonical_axv_doc(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "cv_rag.sqlite3")
    try:
        store.create_schema()

        store.upsert_reference_graph(
            refs=[
                ReferenceRecord(
                    ref_type="arxiv",
                    normalized_value="2104.00680v2",
                    source_kind="curated_repo",
                    source_ref="https://github.com/org/repo",
                    discovered_at_unix=100,
                )
            ],
            resolved=[],
            run_id="run1",
            candidate_retry_days=14,
            candidate_max_retries=5,
            now_unix=100,
        )

        doi = "10.1145/3366423.3380211"
        store.upsert_reference_graph(
            refs=[
                ReferenceRecord(
                    ref_type="doi",
                    normalized_value=doi,
                    source_kind="curated_repo",
                    source_ref="https://github.com/org/repo",
                    discovered_at_unix=101,
                )
            ],
            resolved=[
                ResolvedReference(
                    doc_id="axv:2104.00680v2",
                    arxiv_id="2104.00680",
                    arxiv_id_with_version="2104.00680v2",
                    doi=doi,
                    pdf_url="https://arxiv.org/pdf/2104.00680v2.pdf",
                    resolution_confidence=0.95,
                    source_kind="openalex_resolved",
                    resolved_at_unix=101,
                )
            ],
            run_id="run2",
            candidate_retry_days=14,
            candidate_max_retries=5,
            now_unix=101,
        )

        canonical = store.conn.execute(
            "SELECT doc_id, doi, status FROM documents WHERE doc_id = ?",
            ("axv:2104.00680v2",),
        ).fetchone()
        alias_candidate = store.conn.execute(
            "SELECT status, last_error FROM ingest_candidates WHERE doc_id = ?",
            (f"doi:{doi}",),
        ).fetchone()
        ready_ids = {item.doc_id for item in store.list_ready_candidates(limit=10, now_unix=101)}

        assert canonical is not None
        assert canonical["doi"] == doi
        assert canonical["status"] == "ready"

        assert alias_candidate is not None
        assert alias_candidate["status"] == "resolved"
        assert str(alias_candidate["last_error"]).startswith("merged_to:axv:2104.00680v2")

        assert "axv:2104.00680v2" in ready_ids
        assert f"doi:{doi}" not in ready_ids
    finally:
        store.close()


def test_doi_with_oa_pdf_becomes_ready_candidate(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "cv_rag.sqlite3")
    try:
        store.create_schema()
        doi = "10.48550/arxiv.1706.03762"
        now_unix = 1_700_000_000
        store.upsert_reference_graph(
            refs=[
                ReferenceRecord(
                    ref_type="doi",
                    normalized_value=doi,
                    source_kind="curated_repo",
                    source_ref="https://github.com/org/repo",
                    discovered_at_unix=now_unix,
                )
            ],
            resolved=[
                ResolvedReference(
                    doc_id=f"doi:{doi}",
                    arxiv_id=None,
                    arxiv_id_with_version=None,
                    doi=doi,
                    pdf_url="https://example.org/paper.pdf",
                    resolution_confidence=0.75,
                    source_kind="openalex_resolved",
                    resolved_at_unix=now_unix,
                )
            ],
            run_id="run-ready",
            candidate_retry_days=14,
            candidate_max_retries=5,
            now_unix=now_unix,
        )

        row = store.conn.execute(
            "SELECT status, best_pdf_url FROM ingest_candidates WHERE doc_id = ?",
            (f"doi:{doi}",),
        ).fetchone()
        ready_ids = {item.doc_id for item in store.list_ready_candidates(limit=10, now_unix=now_unix)}

        assert row is not None
        assert row["status"] == "ready"
        assert row["best_pdf_url"] == "https://example.org/paper.pdf"
        assert f"doi:{doi}" in ready_ids
    finally:
        store.close()


def test_doi_without_fulltext_becomes_blocked_with_retry_schedule(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "cv_rag.sqlite3")
    try:
        store.create_schema()
        doi = "10.1000/no-fulltext"
        now_unix = 2_000
        retry_days = 10

        store.upsert_reference_graph(
            refs=[
                ReferenceRecord(
                    ref_type="doi",
                    normalized_value=doi,
                    source_kind="curated_repo",
                    source_ref="https://github.com/org/repo",
                    discovered_at_unix=now_unix,
                )
            ],
            resolved=[
                ResolvedReference(
                    doc_id=f"doi:{doi}",
                    arxiv_id=None,
                    arxiv_id_with_version=None,
                    doi=doi,
                    pdf_url=None,
                    resolution_confidence=0.30,
                    source_kind="openalex_resolved",
                    resolved_at_unix=now_unix,
                )
            ],
            run_id="run-blocked",
            candidate_retry_days=retry_days,
            candidate_max_retries=5,
            now_unix=now_unix,
        )

        row = store.conn.execute(
            "SELECT status, next_retry_unix FROM ingest_candidates WHERE doc_id = ?",
            (f"doi:{doi}",),
        ).fetchone()
        assert row is not None
        assert row["status"] == "blocked"
        assert int(row["next_retry_unix"]) == now_unix + (retry_days * 86400)

        assert store.list_ready_candidates(limit=10, now_unix=now_unix) == []
    finally:
        store.close()
