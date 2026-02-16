from __future__ import annotations

from pathlib import Path

from cv_rag.ingest import pdf_pipeline as pdf_pipeline_module
from cv_rag.ingest import service as ingest_service_module
from cv_rag.ingest.models import PaperMetadata
from cv_rag.ingest.pdf_pipeline import IngestResult
from cv_rag.ingest.tei_extract import Section
from cv_rag.shared.settings import Settings
from cv_rag.storage.repositories import IngestCandidate, ReferenceRecord, ResolvedReference
from cv_rag.storage.sqlite import SQLiteStore


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        pdf_dir=tmp_path / "data" / "pdfs",
        tei_dir=tmp_path / "data" / "tei",
        metadata_dir=tmp_path / "data" / "metadata",
        metadata_json_path=tmp_path / "data" / "metadata" / "arxiv_cs_cv.json",
        sqlite_path=tmp_path / "cv_rag.sqlite3",
    )


def test_ingest_candidates_transitions_ingested_failed_and_blocked(
    monkeypatch: object,
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    settings.ensure_directories()

    store = SQLiteStore(settings.sqlite_path)
    try:
        store.create_schema()
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
                ),
                ResolvedReference(
                    doc_id="url:raw-doc",
                    arxiv_id=None,
                    arxiv_id_with_version=None,
                    doi=None,
                    pdf_url="https://example.org/raw.pdf",
                    resolution_confidence=0.6,
                    source_kind="scraped_pdf",
                ),
            ],
            run_id="run1",
            candidate_retry_days=14,
            candidate_max_retries=5,
            now_unix=1_000,
        )
        store.upsert_reference_graph(
            refs=[
                ReferenceRecord(
                    ref_type="doi",
                    normalized_value="10.1000/missing",
                    source_kind="curated_repo",
                    source_ref="https://github.com/org/repo",
                    discovered_at_unix=1_000,
                )
            ],
            resolved=[],
            run_id="run2",
            candidate_retry_days=14,
            candidate_max_retries=5,
            now_unix=1_000,
        )
    finally:
        store.close()

    captured_doc_ids: list[str] = []

    class FakePipeline:
        def __init__(self, settings_arg: Settings) -> None:
            _ = settings_arg

        def run(
            self,
            papers: list[PaperMetadata],
            metadata_json_path: Path,
            force_grobid: bool = False,
            embed_batch_size: int | None = None,
            cache_only: bool = False,
            on_progress: object | None = None,
        ) -> IngestResult:
            _ = (metadata_json_path, force_grobid, embed_batch_size, cache_only, on_progress)
            captured_doc_ids.extend([paper.resolved_doc_id() for paper in papers])
            return IngestResult(
                papers_processed=len(papers),
                total_chunks=4,
                failed_papers=[f"{papers[0].arxiv_id}: pipeline failure"],
            )

    monkeypatch.setattr(ingest_service_module, "IngestPipeline", FakePipeline)

    stats = ingest_service_module.ingest_candidates(
        settings=settings,
        candidates=[
            IngestCandidate(
                doc_id="url:raw-doc",
                status="ready",
                best_pdf_url="https://example.org/raw.pdf",
                priority_score=1.0,
                retry_count=0,
                next_retry_unix=None,
            ),
            IngestCandidate(
                doc_id="axv:2104.00680v2",
                status="ready",
                best_pdf_url="https://arxiv.org/pdf/2104.00680v2.pdf",
                priority_score=0.9,
                retry_count=0,
                next_retry_unix=None,
            ),
            IngestCandidate(
                doc_id="doi:10.1000/missing",
                status="ready",
                best_pdf_url=None,
                priority_score=0.8,
                retry_count=0,
                next_retry_unix=None,
            ),
        ],
        force_grobid=False,
        embed_batch_size=None,
    )

    assert stats == {
        "selected": 3,
        "queued": 2,
        "ingested": 1,
        "failed": 1,
        "blocked": 1,
    }
    assert captured_doc_ids == ["url:raw-doc", "axv:2104.00680v2"]

    store = SQLiteStore(settings.sqlite_path)
    try:
        failed_row = store.conn.execute(
            "SELECT status, retry_count, last_error FROM ingest_candidates WHERE doc_id = ?",
            ("url:raw-doc",),
        ).fetchone()
        success_row = store.conn.execute(
            "SELECT status FROM ingest_candidates WHERE doc_id = ?",
            ("axv:2104.00680v2",),
        ).fetchone()
        missing_row = store.conn.execute(
            "SELECT status, last_error FROM ingest_candidates WHERE doc_id = ?",
            ("doi:10.1000/missing",),
        ).fetchone()

        assert failed_row is not None
        assert failed_row["status"] == "blocked"
        assert int(failed_row["retry_count"]) == 1
        assert failed_row["last_error"] == "pipeline_failed"

        assert success_row is not None
        assert success_row["status"] == "ingested"

        assert missing_row is not None
        assert missing_row["status"] == "blocked"
        assert missing_row["last_error"] == "missing_pdf_url"
    finally:
        store.close()


def test_pipeline_indexes_url_doc_chunks_with_doc_id(
    monkeypatch: object,
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    settings.ensure_directories()

    captured_points: list[object] = []

    def fake_download_pdf(
        *,
        paper: PaperMetadata,
        pdf_dir: Path,
        timeout_seconds: float,
        user_agent: str,
        cache_only: bool = False,
    ) -> Path:
        _ = (timeout_seconds, user_agent, cache_only)
        pdf_dir.mkdir(parents=True, exist_ok=True)
        path = pdf_dir / f"{paper.safe_file_stem()}.pdf"
        path.write_bytes(b"%PDF-1.4\n")
        return path

    def fake_load_tei_or_parse(
        pdf_path: Path,
        tei_path: Path,
        settings: Settings,
        force: bool,
    ) -> str:
        _ = (pdf_path, tei_path, settings, force)
        return "<TEI/>"

    class FakeEmbedClient:
        def __init__(
            self,
            base_url: str,
            model: str,
            timeout_seconds: float,
        ) -> None:
            _ = (base_url, model, timeout_seconds)

        def embed_in_batches(self, texts: list[str], batch_size: int) -> list[list[float]]:
            _ = batch_size
            return [[0.1, 0.2, 0.3] for _ in texts]

    class FakeQdrantStore:
        def __init__(self, url: str, collection_name: str) -> None:
            _ = (url, collection_name)

        def ensure_collection(self, vector_size: int) -> None:
            _ = vector_size

        def upsert(self, points: list[object]) -> None:
            captured_points.extend(points)

    monkeypatch.setattr(pdf_pipeline_module, "download_pdf", fake_download_pdf)
    monkeypatch.setattr(pdf_pipeline_module, "load_tei_or_parse", fake_load_tei_or_parse)
    monkeypatch.setattr(
        pdf_pipeline_module,
        "extract_sections",
        lambda tei_xml: [Section(title="Method", text="Dense matching with robust features")],
    )
    monkeypatch.setattr(pdf_pipeline_module, "OllamaEmbedClient", FakeEmbedClient)
    monkeypatch.setattr(pdf_pipeline_module, "QdrantStore", FakeQdrantStore)

    pipeline = pdf_pipeline_module.IngestPipeline(settings)
    paper = PaperMetadata(
        arxiv_id="url:raw-doc",
        arxiv_id_with_version="url:raw-doc",
        version=None,
        doc_id="url:raw-doc",
        title="Raw PDF",
        summary="",
        published=None,
        updated=None,
        authors=[],
        pdf_url="https://example.org/raw.pdf",
        abs_url="https://example.org/raw.pdf",
    )

    result = pipeline.run(
        papers=[paper],
        metadata_json_path=settings.metadata_dir / "corpus_candidates_test.json",
        force_grobid=False,
    )

    assert result.papers_processed == 1
    assert result.total_chunks == 1
    assert result.failed_papers == []

    store = SQLiteStore(settings.sqlite_path)
    try:
        chunk_row = store.conn.execute(
            "SELECT doc_id, arxiv_id FROM chunks LIMIT 1",
        ).fetchone()
        paper_row = store.conn.execute(
            "SELECT doc_id, arxiv_id FROM papers WHERE doc_id = ?",
            ("url:raw-doc",),
        ).fetchone()
        hits = store.keyword_search("matching", limit=5)

        assert chunk_row is not None
        assert chunk_row["doc_id"] == "url:raw-doc"
        assert chunk_row["arxiv_id"] == "url:raw-doc"

        assert paper_row is not None
        assert paper_row["doc_id"] == "url:raw-doc"
        assert paper_row["arxiv_id"] == "url:raw-doc"

        assert hits
        assert hits[0]["doc_id"] == "url:raw-doc"

        assert captured_points
        assert captured_points[0].payload["doc_id"] == "url:raw-doc"
    finally:
        store.close()
