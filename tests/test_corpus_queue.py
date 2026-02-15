from __future__ import annotations

from pathlib import Path

import pytest

from cv_rag.storage.repositories import ResolvedReference, compute_candidate_priority
from cv_rag.storage.sqlite import SQLiteStore


def test_list_ready_candidates_orders_by_priority_and_filters_blocked(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "cv_rag.sqlite3")
    try:
        store.create_schema()
        now_unix = 10_000

        store.upsert_reference_graph(
            refs=[],
            resolved=[
                ResolvedReference(
                    doc_id="axv:2104.00680v2",
                    arxiv_id="2104.00680",
                    arxiv_id_with_version="2104.00680v2",
                    doi=None,
                    pdf_url="https://arxiv.org/pdf/2104.00680v2.pdf",
                    resolution_confidence=1.0,
                    source_kind="curated_repo",
                    resolved_at_unix=now_unix,
                ),
                ResolvedReference(
                    doc_id="doi:10.1000/ready-doi",
                    arxiv_id=None,
                    arxiv_id_with_version=None,
                    doi="10.1000/ready-doi",
                    pdf_url="https://example.org/a.pdf",
                    resolution_confidence=0.75,
                    source_kind="openalex_resolved",
                    resolved_at_unix=now_unix,
                ),
                ResolvedReference(
                    doc_id="url:raw-pdf",
                    arxiv_id=None,
                    arxiv_id_with_version=None,
                    doi=None,
                    pdf_url="https://example.org/raw.pdf",
                    resolution_confidence=0.60,
                    source_kind="scraped_pdf",
                    resolved_at_unix=now_unix,
                ),
            ],
            run_id="run1",
            candidate_retry_days=14,
            candidate_max_retries=5,
            now_unix=now_unix,
        )

        store.mark_candidate_result(
            doc_id="doi:10.1000/ready-doi",
            status="blocked",
            reason="temporary_error",
            candidate_retry_days=14,
            candidate_max_retries=5,
            now_unix=now_unix,
        )

        ready = store.list_ready_candidates(limit=10, now_unix=now_unix)
        assert [candidate.doc_id for candidate in ready] == ["axv:2104.00680v2", "url:raw-pdf"]
    finally:
        store.close()


def test_mark_candidate_result_applies_retry_limit_and_final_failed_state(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "cv_rag.sqlite3")
    try:
        store.create_schema()
        now_unix = 5_000
        store.upsert_reference_graph(
            refs=[],
            resolved=[
                ResolvedReference(
                    doc_id="doi:10.1000/retry-me",
                    arxiv_id=None,
                    arxiv_id_with_version=None,
                    doi="10.1000/retry-me",
                    pdf_url="https://example.org/retry.pdf",
                    resolution_confidence=0.70,
                    source_kind="openalex_resolved",
                    resolved_at_unix=now_unix,
                )
            ],
            run_id="run-retry",
            candidate_retry_days=3,
            candidate_max_retries=2,
            now_unix=now_unix,
        )

        store.mark_candidate_result(
            doc_id="doi:10.1000/retry-me",
            status="failed",
            reason="pipeline_failed",
            candidate_retry_days=3,
            candidate_max_retries=2,
            now_unix=now_unix + 1,
        )
        first = store.conn.execute(
            "SELECT status, retry_count, next_retry_unix, last_error FROM ingest_candidates WHERE doc_id = ?",
            ("doi:10.1000/retry-me",),
        ).fetchone()

        assert first is not None
        assert first["status"] == "blocked"
        assert int(first["retry_count"]) == 1
        assert int(first["next_retry_unix"]) == (now_unix + 1) + (3 * 86400)
        assert first["last_error"] == "pipeline_failed"

        store.mark_candidate_result(
            doc_id="doi:10.1000/retry-me",
            status="failed",
            reason="pipeline_failed_again",
            candidate_retry_days=3,
            candidate_max_retries=2,
            now_unix=now_unix + 2,
        )
        second = store.conn.execute(
            "SELECT status, retry_count, next_retry_unix, last_error FROM ingest_candidates WHERE doc_id = ?",
            ("doi:10.1000/retry-me",),
        ).fetchone()
        doc_row = store.conn.execute(
            "SELECT status FROM documents WHERE doc_id = ?",
            ("doi:10.1000/retry-me",),
        ).fetchone()

        assert second is not None
        assert second["status"] == "failed"
        assert int(second["retry_count"]) == 2
        assert second["next_retry_unix"] is None
        assert second["last_error"] == "pipeline_failed_again"

        assert doc_row is not None
        assert doc_row["status"] == "failed"
    finally:
        store.close()


def test_compute_candidate_priority_matches_weight_formula() -> None:
    score = compute_candidate_priority(
        source_kind="curated_repo",
        resolution_confidence=1.0,
        age_days=0,
        retry_count=2,
        resolution_kind="arxiv_versioned",
    )

    # 1.00 + 0.40 + 0.20 + 0.05 - 0.30
    assert score == pytest.approx(1.35)
