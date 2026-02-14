from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

import cv_rag.cli as cli_module
from cv_rag.arxiv_sync import PaperMetadata
from cv_rag.config import Settings


def _paper(arxiv_id_with_version: str, title: str = "Paper") -> PaperMetadata:
    base_id = arxiv_id_with_version.split("v", 1)[0]
    return PaperMetadata(
        arxiv_id=base_id,
        arxiv_id_with_version=arxiv_id_with_version,
        version=None,
        title=title,
        summary="summary",
        published=None,
        updated=None,
        authors=["A. Author"],
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id_with_version}.pdf",
        abs_url=f"https://arxiv.org/abs/{arxiv_id_with_version}",
    )


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        pdf_dir=tmp_path / "data" / "pdfs",
        tei_dir=tmp_path / "data" / "tei",
        metadata_dir=tmp_path / "data" / "metadata",
        metadata_json_path=tmp_path / "data" / "metadata" / "arxiv_cs_cv.json",
        sqlite_path=tmp_path / "cv_rag.sqlite3",
    )


def test_ingest_passes_skip_set_by_default(monkeypatch: object, tmp_path: Path) -> None:
    runner = CliRunner()
    settings = _settings(tmp_path)
    captured: dict[str, object] = {}

    def fake_fetch(
        limit: int,
        arxiv_api_url: str,
        timeout_seconds: float,
        user_agent: str,
        max_retries: int = 5,
        backoff_start_seconds: float = 2.0,
        backoff_cap_seconds: float = 30.0,
        skip_arxiv_id_with_version: set[str] | None = None,
        max_scan_results: int | None = None,
        stats: dict[str, int] | None = None,
    ) -> list[PaperMetadata]:
        captured["skip_set"] = skip_arxiv_id_with_version
        if stats is not None:
            stats.update({"requested": limit, "selected": 1, "skipped": 1, "scanned": 2})
        return [_paper("2602.12177v1")]

    def fake_run_ingest(
        papers: list[PaperMetadata],
        metadata_json_path: Path,
        force_grobid: bool,
        embed_batch_size: int | None,
    ) -> None:
        captured["papers_len"] = len(papers)

    monkeypatch.setattr(cli_module, "get_settings", lambda: settings)
    monkeypatch.setattr(cli_module.SQLiteStore, "get_ingested_versioned_ids", lambda self: {"2104.00680v1"})
    monkeypatch.setattr(cli_module, "fetch_cs_cv_papers", fake_fetch)
    monkeypatch.setattr(cli_module, "_run_ingest", fake_run_ingest)

    result = runner.invoke(cli_module.app, ["ingest", "--limit", "1"])

    assert result.exit_code == 0
    assert captured["skip_set"] == {"2104.00680v1"}
    assert captured["papers_len"] == 1


def test_ingest_no_skip_disables_skip_set(monkeypatch: object, tmp_path: Path) -> None:
    runner = CliRunner()
    settings = _settings(tmp_path)
    captured: dict[str, object] = {}

    def fail_get_ingested(self: object) -> set[str]:
        raise AssertionError("get_ingested_versioned_ids should not be called when --no-skip-ingested is set")

    def fake_fetch(
        limit: int,
        arxiv_api_url: str,
        timeout_seconds: float,
        user_agent: str,
        max_retries: int = 5,
        backoff_start_seconds: float = 2.0,
        backoff_cap_seconds: float = 30.0,
        skip_arxiv_id_with_version: set[str] | None = None,
        max_scan_results: int | None = None,
        stats: dict[str, int] | None = None,
    ) -> list[PaperMetadata]:
        captured["skip_set"] = skip_arxiv_id_with_version
        if stats is not None:
            stats.update({"requested": limit, "selected": 1, "skipped": 0, "scanned": 1})
        return [_paper("2602.12178v1")]

    monkeypatch.setattr(cli_module, "get_settings", lambda: settings)
    monkeypatch.setattr(cli_module.SQLiteStore, "get_ingested_versioned_ids", fail_get_ingested)
    monkeypatch.setattr(cli_module, "fetch_cs_cv_papers", fake_fetch)
    monkeypatch.setattr(
        cli_module,
        "_run_ingest",
        lambda papers, metadata_json_path, force_grobid, embed_batch_size: None,
    )

    result = runner.invoke(cli_module.app, ["ingest", "--limit", "1", "--no-skip-ingested"])

    assert result.exit_code == 0
    assert captured["skip_set"] is None


def test_ingest_no_new_papers_exits_zero_with_message(monkeypatch: object, tmp_path: Path) -> None:
    runner = CliRunner()
    settings = _settings(tmp_path)

    def fake_fetch(
        limit: int,
        arxiv_api_url: str,
        timeout_seconds: float,
        user_agent: str,
        max_retries: int = 5,
        backoff_start_seconds: float = 2.0,
        backoff_cap_seconds: float = 30.0,
        skip_arxiv_id_with_version: set[str] | None = None,
        max_scan_results: int | None = None,
        stats: dict[str, int] | None = None,
    ) -> list[PaperMetadata]:
        if stats is not None:
            stats.update({"requested": limit, "selected": 0, "skipped": 3, "scanned": 3})
        return []

    def fail_run_ingest(
        papers: list[PaperMetadata],
        metadata_json_path: Path,
        force_grobid: bool,
        embed_batch_size: int | None,
    ) -> None:
        raise AssertionError("_run_ingest should not be called when there are no new papers")

    monkeypatch.setattr(cli_module, "get_settings", lambda: settings)
    monkeypatch.setattr(cli_module.SQLiteStore, "get_ingested_versioned_ids", lambda self: {"2602.12177v1"})
    monkeypatch.setattr(cli_module, "fetch_cs_cv_papers", fake_fetch)
    monkeypatch.setattr(cli_module, "_run_ingest", fail_run_ingest)

    result = runner.invoke(cli_module.app, ["ingest", "--limit", "3"])

    assert result.exit_code == 0
    assert "No new cs.CV papers to ingest after skipping already ingested versions." in result.output
