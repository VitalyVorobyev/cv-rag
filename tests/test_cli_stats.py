from __future__ import annotations

import json
from pathlib import Path
import re

from typer.testing import CliRunner

import cv_rag.cli as cli_module
from cv_rag.arxiv_sync import PaperMetadata
from cv_rag.config import Settings
from cv_rag.sqlite_store import SQLiteStore


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        pdf_dir=tmp_path / "data" / "pdfs",
        tei_dir=tmp_path / "data" / "tei",
        metadata_dir=tmp_path / "data" / "metadata",
        metadata_json_path=tmp_path / "data" / "metadata" / "arxiv_cs_cv.json",
        sqlite_path=tmp_path / "cv_rag.sqlite3",
    )


def _paper(arxiv_id: str, arxiv_id_with_version: str, title: str) -> PaperMetadata:
    return PaperMetadata(
        arxiv_id=arxiv_id,
        arxiv_id_with_version=arxiv_id_with_version,
        version=None,
        title=title,
        summary="",
        published=None,
        updated=None,
        authors=["A. Author"],
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id_with_version}.pdf",
        abs_url=f"https://arxiv.org/abs/{arxiv_id_with_version}",
    )


def test_stats_command_errors_when_sqlite_missing(monkeypatch: object, tmp_path: Path) -> None:
    runner = CliRunner()
    settings = _settings(tmp_path)
    monkeypatch.setattr(cli_module, "get_settings", lambda: settings)

    result = runner.invoke(cli_module.app, ["stats"])

    assert result.exit_code == 1
    assert "SQLite database not found" in result.output


def test_stats_command_reports_database_statistics(monkeypatch: object, tmp_path: Path) -> None:
    runner = CliRunner()
    settings = _settings(tmp_path)
    settings.ensure_directories()

    indexed_pdf_1 = settings.pdf_dir / "1111.1111v1.pdf"
    indexed_pdf_2 = settings.pdf_dir / "2222.2222v1.pdf"
    orphan_pdf = settings.pdf_dir / "orphan.pdf"
    tei_file = settings.tei_dir / "1111.1111v1.tei.xml"
    indexed_pdf_1.write_text("pdf", encoding="utf-8")
    indexed_pdf_2.write_text("pdf", encoding="utf-8")
    orphan_pdf.write_text("pdf", encoding="utf-8")
    tei_file.write_text("tei", encoding="utf-8")

    store = SQLiteStore(settings.sqlite_path)
    try:
        store.create_schema()
        store.upsert_paper(
            paper=_paper("1111.1111", "1111.1111v1", "Paper A"),
            pdf_path=indexed_pdf_1,
            tei_path=tei_file,
        )
        store.upsert_paper(
            paper=_paper("2222.2222", "2222.2222v1", "Paper B"),
            pdf_path=indexed_pdf_2,
            tei_path=None,
        )
        store.upsert_chunks(
            [
                {
                    "chunk_id": "1111.1111:0",
                    "arxiv_id": "1111.1111",
                    "title": "Paper A",
                    "section_title": "Method",
                    "chunk_index": 0,
                    "text": "text a",
                },
                {
                    "chunk_id": "2222.2222:0",
                    "arxiv_id": "2222.2222",
                    "title": "Paper B",
                    "section_title": "Method",
                    "chunk_index": 0,
                    "text": "text b",
                },
            ]
        )
        store.upsert_paper_metrics(
            [
                {
                    "arxiv_id": "1111.1111",
                    "citation_count": 100,
                    "year": 2020,
                    "venue": "CVPR",
                    "publication_types": json.dumps(["Conference"]),
                    "is_peer_reviewed": 1,
                    "score": 5.0,
                    "tier": 0,
                    "updated_at": 1_700_000_000,
                }
            ]
        )
    finally:
        store.close()

    monkeypatch.setattr(cli_module, "get_settings", lambda: settings)
    result = runner.invoke(cli_module.app, ["stats", "--top-venues", "3"])

    assert result.exit_code == 0
    assert "Database Stats" in result.output
    assert "papers rows" in result.output
    assert "paper_metrics rows" in result.output
    assert "papers without metrics" in result.output
    assert "orphan pdf files (disk - indexed)" in result.output
    assert re.search(r"Tier\s+Distribution", result.output)
    assert re.search(r"Top\s+Venues\s+\(top\s+3\)", result.output)
