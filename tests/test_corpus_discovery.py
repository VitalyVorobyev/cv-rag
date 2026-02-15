from __future__ import annotations

from pathlib import Path

from cv_rag.seeding import awesome as awesome_module
from cv_rag.storage.repositories import ReferenceRecord
from cv_rag.storage.sqlite import SQLiteStore


def test_discover_awesome_references_writes_run_artifact_and_upserts_graph(
    monkeypatch: object,
    tmp_path: Path,
) -> None:
    sources_path = tmp_path / "awesome_sources.txt"
    sources_path.write_text("owner/repo\n", encoding="utf-8")

    def fake_fetch_repo_readme(
        *,
        client: object,
        repo: str,
        max_retries: int,
        backoff_start_seconds: float,
        backoff_cap_seconds: float,
    ) -> object:
        _ = (client, repo, max_retries, backoff_start_seconds, backoff_cap_seconds)
        return awesome_module._FetchedContent(
            text=(
                "See https://arxiv.org/abs/2104.00680v2 "
                "and DOI:10.1145/3366423.3380211."
            ),
            found_in="README.md",
        )

    monkeypatch.setattr(awesome_module, "_fetch_repo_readme", fake_fetch_repo_readme)

    sqlite_path = tmp_path / "cv_rag.sqlite3"
    runs_dir = tmp_path / "runs"
    refs = awesome_module.discover_awesome_references(
        sources_path=sources_path,
        run_id="run123",
        user_agent="cv-rag/test",
        runs_dir=runs_dir,
        sqlite_path=sqlite_path,
        delay_seconds=0.0,
    )

    assert {ref.ref_type for ref in refs} == {"arxiv", "doi"}
    assert (runs_dir / "run123" / "awesome_references.jsonl").exists()

    store = SQLiteStore(sqlite_path)
    try:
        store.create_schema()
        rows = store.conn.execute(
            "SELECT COUNT(*) AS count FROM reference_events WHERE run_id = ?",
            ("run123",),
        ).fetchone()
        assert rows is not None
        assert int(rows["count"]) == len(refs)
    finally:
        store.close()


def test_discover_awesome_references_uses_append_only_run_directories(
    monkeypatch: object,
    tmp_path: Path,
) -> None:
    sources_path = tmp_path / "awesome_sources.txt"
    sources_path.write_text("owner/repo\n", encoding="utf-8")

    def fake_fetch_repo_readme(
        *,
        client: object,
        repo: str,
        max_retries: int,
        backoff_start_seconds: float,
        backoff_cap_seconds: float,
    ) -> object:
        _ = (client, repo, max_retries, backoff_start_seconds, backoff_cap_seconds)
        return awesome_module._FetchedContent(
            text="arXiv:2104.00680v2 DOI:10.1145/3366423.3380211",
            found_in="README.md",
        )

    monkeypatch.setattr(awesome_module, "_fetch_repo_readme", fake_fetch_repo_readme)

    runs_dir = tmp_path / "runs"
    awesome_module.discover_awesome_references(
        sources_path=sources_path,
        run_id="runA",
        user_agent="cv-rag/test",
        runs_dir=runs_dir,
        sqlite_path=None,
        delay_seconds=0.0,
    )
    run_a_artifact = runs_dir / "runA" / "awesome_references.jsonl"
    before = run_a_artifact.read_text(encoding="utf-8")

    awesome_module.discover_awesome_references(
        sources_path=sources_path,
        run_id="runB",
        user_agent="cv-rag/test",
        runs_dir=runs_dir,
        sqlite_path=None,
        delay_seconds=0.0,
    )
    run_b_artifact = runs_dir / "runB" / "awesome_references.jsonl"

    assert run_a_artifact.exists()
    assert run_b_artifact.exists()
    assert run_a_artifact.read_text(encoding="utf-8") == before


def test_upsert_reference_graph_dedupes_canonical_doc_ids_across_sources(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "cv_rag.sqlite3")
    try:
        store.create_schema()
        refs = [
            ReferenceRecord(
                ref_type="arxiv",
                normalized_value="2104.00680v2",
                source_kind="curated_repo",
                source_ref="https://github.com/a/repo",
                discovered_at_unix=100,
            ),
            ReferenceRecord(
                ref_type="arxiv",
                normalized_value="2104.00680v2",
                source_kind="visionbib_page",
                source_ref="https://visionbib.example/page",
                discovered_at_unix=100,
            ),
            ReferenceRecord(
                ref_type="doi",
                normalized_value="10.1145/3366423.3380211",
                source_kind="curated_repo",
                source_ref="https://github.com/a/repo",
                discovered_at_unix=100,
            ),
            ReferenceRecord(
                ref_type="doi",
                normalized_value="10.1145/3366423.3380211",
                source_kind="visionbib_page",
                source_ref="https://visionbib.example/page",
                discovered_at_unix=100,
            ),
        ]
        store.upsert_reference_graph(
            refs=refs,
            resolved=[],
            run_id="run123",
            candidate_retry_days=14,
            candidate_max_retries=5,
            now_unix=100,
        )

        doc_count = store.conn.execute("SELECT COUNT(*) AS count FROM documents").fetchone()
        source_count = store.conn.execute("SELECT COUNT(*) AS count FROM document_sources").fetchone()
        assert doc_count is not None
        assert source_count is not None
        assert int(doc_count["count"]) == 2
        assert int(source_count["count"]) == 4
    finally:
        store.close()
